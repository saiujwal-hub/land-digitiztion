"""
postgres_store.py - Transactional PostgreSQL Persistence Engine for OneBhoomi

Provides a thread-safe, transactional PostgreSQL repository layer that exactly
preserves the signatures, return shapes, and semantics of existing flat-file
persistence functions in OneBhoomi.
"""

import atexit
import hashlib
import json
import logging
import os
import re
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)

def is_postgres_backend() -> bool:
    """
    Returns True if PostgreSQL is configured as the active persistence backend.
    Explicitly checks ONEBHOOMI_PERSISTENCE_BACKEND ('postgres' or 'json').
    If unset, checks legacy ONEBHOOMI_USE_POSTGRES ('true' or 'false').
    Default is True (PostgreSQL is production default).
    """
    backend = os.environ.get("ONEBHOOMI_PERSISTENCE_BACKEND", "").lower().strip()
    if backend:
        return backend == "postgres"
    legacy = os.environ.get("ONEBHOOMI_USE_POSTGRES", "true").lower().strip()
    return legacy in ("true", "1", "yes")


USE_POSTGRES = is_postgres_backend()

_db_initialized = False
_init_lock = threading.Lock()

_pool: Optional[ConnectionPool] = None
_pool_config: Optional[Dict[str, Any]] = None
_pool_lock = threading.Lock()


def get_connection_kwargs() -> Dict[str, Any]:
    """Resolves connection parameters from environment dynamically without hardcoded credentials."""
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return {"conninfo": db_url}
    kwargs = {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "dbname": os.environ.get("POSTGRES_DB", "onebhoomi"),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
    }
    password = os.environ.get("POSTGRES_PASSWORD")
    if password is not None:
        kwargs["password"] = password
    return kwargs


def get_pool() -> ConnectionPool:
    """Returns or initializes the singleton thread-safe ConnectionPool."""
    global _pool, _pool_config
    current_kwargs = get_connection_kwargs()
    if _pool is not None and not _pool.closed and _pool_config == current_kwargs:
        return _pool
    with _pool_lock:
        if _pool is not None and not _pool.closed and _pool_config == current_kwargs:
            return _pool
        if _pool is not None and not _pool.closed:
            try:
                _pool.close()
            except Exception:
                pass
            _pool = None

        _pool_config = dict(current_kwargs)
        kwargs = dict(current_kwargs)
        conninfo = kwargs.pop("conninfo", None)

        min_size = int(os.environ.get("POSTGRES_POOL_MIN_SIZE", "2"))
        max_size = int(os.environ.get("POSTGRES_POOL_MAX_SIZE", "10"))
        timeout = float(os.environ.get("POSTGRES_POOL_TIMEOUT", "30.0"))

        if conninfo:
            _pool = ConnectionPool(
                conninfo=conninfo,
                min_size=min_size,
                max_size=max_size,
                timeout=timeout,
                open=True,
                kwargs={"row_factory": dict_row, **kwargs},
            )
        else:
            _pool = ConnectionPool(
                min_size=min_size,
                max_size=max_size,
                timeout=timeout,
                open=True,
                kwargs={"row_factory": dict_row, **kwargs},
            )
        return _pool


def close_pool() -> None:
    """Safely closes the connection pool if open."""
    global _pool, _pool_config
    with _pool_lock:
        if _pool is not None and not _pool.closed:
            try:
                _pool.close()
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL connection pool: {e}")
            finally:
                _pool = None
                _pool_config = None


def reset_pool() -> None:
    """Closes existing pool and forces recreation on next use."""
    close_pool()



def get_pool_stats() -> Dict[str, Any]:
    """Returns operational statistics of the connection pool."""
    pool = get_pool()
    return pool.get_stats()


atexit.register(close_pool)


class PooledConnectionContext:
    """
    Context manager wrapping connection acquisition from ConnectionPool.
    Ensures safe transaction commit/rollback and automatic return to pool.
    """
    def __init__(self, pool: ConnectionPool, timeout: float):
        self._cm = pool.connection(timeout=timeout)
        self._conn: Optional[psycopg.Connection] = None

    def __enter__(self) -> psycopg.Connection:
        self._conn = self._cm.__enter__()
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return self._cm.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._conn = None

    def cursor(self, *args, **kwargs):
        if self._conn is None:
            self._conn = self._cm.__enter__()
        return self._conn.cursor(*args, **kwargs)

    def commit(self):
        if self._conn is not None:
            self._conn.commit()

    def rollback(self):
        if self._conn is not None:
            self._conn.rollback()

    def close(self):
        if self._conn is not None:
            self._cm.__exit__(None, None, None)
            self._conn = None


def get_connection() -> PooledConnectionContext:
    """
    Returns a pooled connection context manager.
    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
            conn.commit()
    """
    pool = get_pool()
    timeout = float(os.environ.get("POSTGRES_POOL_TIMEOUT", "30.0"))
    kwargs = get_connection_kwargs()
    conninfo = kwargs.get("conninfo", "")
    if conninfo and "connect_timeout=" in conninfo:
        m = re.search(r"connect_timeout=(\d+)", conninfo)
        if m:
            timeout = min(timeout, float(m.group(1)) + 1.0)
    return PooledConnectionContext(pool, timeout)



# =====================================================================
# STALE-SNAPSHOT TRACKING ENGINE
# =====================================================================

def _record_fingerprint(record: Any) -> str:
    """Computes a deterministic SHA-256 fingerprint of a record dictionary."""
    if not isinstance(record, (dict, list)):
        return hashlib.sha256(str(record).encode("utf-8")).hexdigest()

    def default_serializer(o):
        if isinstance(o, (bytes, bytearray, memoryview)):
            return bytes(o).hex()
        if isinstance(o, (datetime,)):
            return o.isoformat()
        return str(o)

    try:
        serialized = json.dumps(record, sort_keys=True, default=default_serializer)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(str(record).encode("utf-8")).hexdigest()


class SnapshotDict(dict):
    """
    A dict subclass returned by load_*_db functions.
    Preserves initial snapshot state so save_*_db functions can:
    1. Detect which specific records were modified or added.
    2. Detect which specific records were explicitly deleted from the snapshot.
    3. Avoid overwriting concurrent modifications made by other threads to unrelated records.
    4. Avoid deleting records created concurrently by other threads after the snapshot was loaded.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_keys: Set[str] = set()
        self._initial_fingerprints: Dict[str, str] = {}
        self._is_snapshot: bool = False

    def freeze_snapshot(self) -> "SnapshotDict":
        self._initial_keys = set(self.keys())
        self._initial_fingerprints = {k: _record_fingerprint(v) for k, v in self.items()}
        self._is_snapshot = True
        return self


def init_db(schema_path: Optional[str] = None) -> None:
    """Initializes PostgreSQL schema idempotently."""
    global _db_initialized
    with _init_lock:
        if _db_initialized:
            return
        sql_file = Path(schema_path) if schema_path else Path(__file__).parent / "schema.sql"
        if not sql_file.exists():
            raise FileNotFoundError(f"Schema file not found at {sql_file}")
        schema_sql = sql_file.read_text(encoding="utf-8")

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
            conn.commit()
        _db_initialized = True
        logger.info("PostgreSQL database schema initialized successfully.")


def _ensure_init():
    if not _db_initialized:
        try:
            init_db()
        except Exception as e:
            logger.warning(f"Could not auto-initialize DB schema: {e}")


def _parse_iso_utc(ts_str: str) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


# =====================================================================
# 1. VERIFICATION RECORDS PERSISTENCE
# =====================================================================

def _row_to_verification_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a database row back into the exact Python record dict."""
    if not row:
        return {}
    rec = {
        "verification_id": row["verification_id"],
        "status": row["status"],
        "is_land_document": bool(row["is_land_document"]),
        "populated_fields": row["populated_fields"] if row["populated_fields"] is not None else [],
        "file_hash": row["file_hash"],
        "uploaded_by_user_id": row["uploaded_by_user_id"],
        "document_payload": row["document_payload"] if row["document_payload"] is not None else {},
        "raw_ocr": row["raw_ocr"],
        "field_provenance": row["field_provenance"] if row["field_provenance"] is not None else {},
        "checks": row["checks"] if row["checks"] is not None else [],
        "duplicate_info": row["duplicate_info"],
        "clerk_submitted": bool(row["clerk_submitted"]),
        "created_at": row["created_at"],
        "decision": row["decision"],
        "signature": row["signature"],
        "public_key": row["public_key"],
        "qr_code": row["qr_code"],
        "filename": row["filename"],
    }
    if row.get("submitted_at") is not None:
        rec["submitted_at"] = row["submitted_at"]
    if row.get("approved_at") is not None:
        rec["approved_at"] = row["approved_at"]
    if row.get("neural_nlp") is not None:
        rec["neural_nlp"] = row["neural_nlp"]
    if row.get("canonical_sealed_payload") is not None:
        raw_b = row["canonical_sealed_payload"]
        rec["canonical_sealed_payload"] = bytes(raw_b) if isinstance(raw_b, memoryview) else raw_b

    return rec


def pg_load_db() -> Dict[str, Dict[str, Any]]:
    """Reads all verification records from PostgreSQL, returning a SnapshotDict."""
    _ensure_init()
    result = SnapshotDict()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM verification_records ORDER BY created_at ASC")
            rows = cur.fetchall()
            for r in rows:
                rec = _row_to_verification_record(r)
                result[rec["verification_id"]] = rec
    return result.freeze_snapshot()


def pg_get_record(verification_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single verification record by ID from PostgreSQL."""
    if not verification_id:
        return None
    _ensure_init()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM verification_records WHERE verification_id = %s", (verification_id,))
            row = cur.fetchone()
            if not row:
                return None
            return _row_to_verification_record(row)


def pg_save_record(record: Dict[str, Any]) -> None:
    """
    Saves or updates a verification record inside an atomic transaction,
    enforcing immutability once approved, and preserving raw_ocr and field_provenance.
    """
    verification_id = record.get("verification_id")
    if not verification_id:
        raise ValueError("Record is missing a verification_id")

    _ensure_init()

    # Normalize clerk_submitted default
    if "clerk_submitted" not in record:
        rec_status = (record.get("status") or "").upper()
        if rec_status in {"APPROVED", "REJECTED", "DUPLICATE"}:
            record["clerk_submitted"] = True
        else:
            record["clerk_submitted"] = False

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Check existing record with FOR UPDATE
            cur.execute(
                "SELECT status, document_payload, raw_ocr, field_provenance, canonical_sealed_payload "
                "FROM verification_records WHERE verification_id = %s FOR UPDATE",
                (verification_id,)
            )
            existing = cur.fetchone()

            if existing:
                if existing["status"] == "APPROVED":
                    # Check document payload immutability
                    if existing["document_payload"] != record.get("document_payload"):
                        raise ValueError("Immutable approved records cannot be modified.")

                # Immutably preserve raw_ocr and field_provenance
                if existing.get("raw_ocr") is not None:
                    record["raw_ocr"] = existing["raw_ocr"]
                if existing.get("field_provenance") and not record.get("field_provenance"):
                    record["field_provenance"] = existing["field_provenance"]

            # Compute or preserve canonical_sealed_payload
            canonical_bytes = None
            if record.get("canonical_sealed_payload"):
                raw_b = record["canonical_sealed_payload"]
                canonical_bytes = bytes(raw_b) if isinstance(raw_b, memoryview) else raw_b
            elif existing and existing.get("canonical_sealed_payload"):
                canonical_bytes = bytes(existing["canonical_sealed_payload"])
            elif (record.get("status") or "").upper() == "APPROVED" and record.get("document_payload"):
                try:
                    import verification_service
                    canonical_bytes = verification_service.canonicalize_document(record["document_payload"])
                except Exception:
                    canonical_bytes = None

            cur.execute(
                """
                INSERT INTO verification_records (
                    verification_id, status, is_land_document, file_hash, filename,
                    uploaded_by_user_id, clerk_submitted, created_at, submitted_at, approved_at,
                    signature, public_key, qr_code, decision, populated_fields,
                    document_payload, canonical_sealed_payload, raw_ocr, field_provenance,
                    checks, duplicate_info, neural_nlp
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (verification_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    is_land_document = EXCLUDED.is_land_document,
                    file_hash = EXCLUDED.file_hash,
                    filename = EXCLUDED.filename,
                    uploaded_by_user_id = EXCLUDED.uploaded_by_user_id,
                    clerk_submitted = EXCLUDED.clerk_submitted,
                    submitted_at = EXCLUDED.submitted_at,
                    approved_at = EXCLUDED.approved_at,
                    signature = EXCLUDED.signature,
                    public_key = EXCLUDED.public_key,
                    qr_code = EXCLUDED.qr_code,
                    decision = EXCLUDED.decision,
                    populated_fields = EXCLUDED.populated_fields,
                    document_payload = EXCLUDED.document_payload,
                    canonical_sealed_payload = COALESCE(EXCLUDED.canonical_sealed_payload, verification_records.canonical_sealed_payload),
                    raw_ocr = COALESCE(verification_records.raw_ocr, EXCLUDED.raw_ocr),
                    field_provenance = COALESCE(verification_records.field_provenance, EXCLUDED.field_provenance),
                    checks = EXCLUDED.checks,
                    duplicate_info = EXCLUDED.duplicate_info,
                    neural_nlp = EXCLUDED.neural_nlp;
                """,
                (
                    verification_id,
                    record.get("status", "READY_FOR_APPROVAL"),
                    record.get("is_land_document", False),
                    record.get("file_hash"),
                    record.get("filename"),
                    record.get("uploaded_by_user_id"),
                    record.get("clerk_submitted", False),
                    record.get("created_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    record.get("submitted_at"),
                    record.get("approved_at"),
                    record.get("signature"),
                    record.get("public_key"),
                    record.get("qr_code"),
                    Jsonb(record.get("decision")) if record.get("decision") is not None else None,
                    Jsonb(record.get("populated_fields") or []),
                    Jsonb(record.get("document_payload") or {}),
                    canonical_bytes,
                    Jsonb(record["raw_ocr"]) if record.get("raw_ocr") is not None else None,
                    Jsonb(record.get("field_provenance") or {}),
                    Jsonb(record.get("checks") or []),
                    Jsonb(record.get("duplicate_info")) if record.get("duplicate_info") is not None else None,
                    Jsonb(record.get("neural_nlp")) if record.get("neural_nlp") is not None else None,
                )
            )
        conn.commit()


def pg_save_db(db: Dict[str, Any]) -> None:
    """
    Reconciles the entire verification_records table inside a single atomic transaction.
    Stale-snapshot safe:
    - If db is empty (len == 0), deletes all records.
    - If db is a SnapshotDict:
        * Deletes ONLY records explicitly removed from this snapshot (present in _initial_keys but absent now).
        * Never deletes concurrent records created by other transactions after snapshot load.
        * Upserts ONLY records that were modified or newly added in this snapshot.
        * Unmodified records are NOT written, preventing stale-snapshot overwrites of other threads' changes.
    - If db is a plain dict:
        * Upserts all records present in db without deleting absent records.
    """
    _ensure_init()

    with get_connection() as conn:
        with conn.cursor() as cur:
            if len(db) == 0:
                cur.execute("DELETE FROM verification_records")
                conn.commit()
                return

            if isinstance(db, SnapshotDict) and db._is_snapshot:
                current_keys = set(db.keys())
                deleted_keys = db._initial_keys - current_keys
                if deleted_keys:
                    cur.execute(
                        "DELETE FROM verification_records WHERE verification_id = ANY(%s)",
                        (list(deleted_keys),)
                    )

                to_save = {}
                for vid, record in db.items():
                    if not isinstance(record, dict):
                        continue
                    if vid not in db._initial_keys or _record_fingerprint(record) != db._initial_fingerprints.get(vid):
                        to_save[vid] = record
            else:
                to_save = {vid: rec for vid, rec in db.items() if isinstance(rec, dict)}

            for vid, record in to_save.items():
                record["verification_id"] = vid
                if "clerk_submitted" not in record:
                    rec_status = (record.get("status") or "").upper()
                    record["clerk_submitted"] = rec_status in {"APPROVED", "REJECTED", "DUPLICATE"}

                canonical_bytes = None
                if record.get("canonical_sealed_payload"):
                    raw_b = record["canonical_sealed_payload"]
                    canonical_bytes = bytes(raw_b) if isinstance(raw_b, memoryview) else raw_b
                elif (record.get("status") or "").upper() == "APPROVED" and record.get("document_payload"):
                    try:
                        import verification_service
                        canonical_bytes = verification_service.canonicalize_document(record["document_payload"])
                    except Exception:
                        canonical_bytes = None

                cur.execute(
                    """
                    INSERT INTO verification_records (
                        verification_id, status, is_land_document, file_hash, filename,
                        uploaded_by_user_id, clerk_submitted, created_at, submitted_at, approved_at,
                        signature, public_key, qr_code, decision, populated_fields,
                        document_payload, canonical_sealed_payload, raw_ocr, field_provenance,
                        checks, duplicate_info, neural_nlp
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s
                    )
                    ON CONFLICT (verification_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        is_land_document = EXCLUDED.is_land_document,
                        file_hash = EXCLUDED.file_hash,
                        filename = EXCLUDED.filename,
                        uploaded_by_user_id = EXCLUDED.uploaded_by_user_id,
                        clerk_submitted = EXCLUDED.clerk_submitted,
                        submitted_at = EXCLUDED.submitted_at,
                        approved_at = EXCLUDED.approved_at,
                        signature = EXCLUDED.signature,
                        public_key = EXCLUDED.public_key,
                        qr_code = EXCLUDED.qr_code,
                        decision = EXCLUDED.decision,
                        populated_fields = EXCLUDED.populated_fields,
                        document_payload = EXCLUDED.document_payload,
                        canonical_sealed_payload = COALESCE(EXCLUDED.canonical_sealed_payload, verification_records.canonical_sealed_payload),
                        raw_ocr = COALESCE(verification_records.raw_ocr, EXCLUDED.raw_ocr),
                        field_provenance = COALESCE(verification_records.field_provenance, EXCLUDED.field_provenance),
                        checks = EXCLUDED.checks,
                        duplicate_info = EXCLUDED.duplicate_info,
                        neural_nlp = EXCLUDED.neural_nlp;
                    """,
                    (
                        vid,
                        record.get("status", "READY_FOR_APPROVAL"),
                        record.get("is_land_document", False),
                        record.get("file_hash"),
                        record.get("filename"),
                        record.get("uploaded_by_user_id"),
                        record.get("clerk_submitted", False),
                        record.get("created_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        record.get("submitted_at"),
                        record.get("approved_at"),
                        record.get("signature"),
                        record.get("public_key"),
                        record.get("qr_code"),
                        Jsonb(record.get("decision")) if record.get("decision") is not None else None,
                        Jsonb(record.get("populated_fields") or []),
                        Jsonb(record.get("document_payload") or {}),
                        canonical_bytes,
                        Jsonb(record["raw_ocr"]) if record.get("raw_ocr") is not None else None,
                        Jsonb(record.get("field_provenance") or {}),
                        Jsonb(record.get("checks") or []),
                        Jsonb(record.get("duplicate_info")) if record.get("duplicate_info") is not None else None,
                        Jsonb(record.get("neural_nlp")) if record.get("neural_nlp") is not None else None,
                    )
                )
        conn.commit()


# =====================================================================
# 2. USERS PERSISTENCE
# =====================================================================

def _row_to_user(row: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a database row into the exact user dict structure."""
    if not row:
        return {}
    u = {
        "user_id": row["user_id"],
        "name": row["name"],
        "role": row["role"],
        "identities": row["identities"] if row["identities"] is not None else [],
        "created_at": row["created_at"],
    }
    if row.get("assigned_officer_id") is not None:
        u["assigned_officer_id"] = row["assigned_officer_id"]
    return u


def pg_load_users_db() -> Dict[str, Dict[str, Any]]:
    """Reads all users from PostgreSQL, returning a SnapshotDict."""
    _ensure_init()
    result = SnapshotDict()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users ORDER BY created_at ASC")
            for r in cur.fetchall():
                result[r["user_id"]] = _row_to_user(r)
    return result.freeze_snapshot()


def pg_get_user(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single user by user_id from PostgreSQL."""
    if not user_id:
        return None
    _ensure_init()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if not row:
                return None
            return _row_to_user(row)


def pg_get_user_by_identity(identity_type: str, identifier: str) -> Optional[Dict[str, Any]]:
    """
    Finds a user linked with a specific auth identity (e.g. email, phone, google).
    Checks both users.identities and auth_credentials.
    """
    if not identity_type or not identifier:
        return None
    _ensure_init()
    target_type = identity_type.strip().lower()
    target_id = identifier.strip().lower()

    with get_connection() as conn:
        with conn.cursor() as cur:
            # 1. First check auth_credentials for quick indexed lookup
            cred_key = f"{target_type}:{target_id}"
            cur.execute("SELECT user_id FROM auth_credentials WHERE identity_key = %s", (cred_key,))
            cred_row = cur.fetchone()
            if cred_row:
                cur.execute("SELECT * FROM users WHERE user_id = %s", (cred_row["user_id"],))
                u_row = cur.fetchone()
                if u_row:
                    return _row_to_user(u_row)

            # 2. Check users.identities JSONB
            cur.execute("SELECT * FROM users")
            for r in cur.fetchall():
                user = _row_to_user(r)
                identities = user.get("identities", [])
                if not isinstance(identities, list):
                    continue
                for identity in identities:
                    if isinstance(identity, dict):
                        itype = str(identity.get("type", "")).strip().lower()
                        ival = str(identity.get("identifier") or identity.get("value") or "").strip().lower()
                        if itype == target_type and ival == target_id:
                            return user
                    elif isinstance(identity, str) and target_type == "email":
                        if identity.strip().lower() == target_id:
                            return user
    return None


def pg_save_user(user: Dict[str, Any]) -> None:
    """Saves or updates a user in PostgreSQL inside an atomic transaction."""
    user_id = user.get("user_id")
    if not user_id:
        raise ValueError("User must have a user_id")
    role = user.get("role")
    import accounts_store
    if role not in accounts_store.VALID_ROLES:
        raise ValueError(f"Invalid user role: {role}. Must be one of {accounts_store.VALID_ROLES}")

    _ensure_init()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (user_id, name, role, assigned_officer_id, identities, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    role = EXCLUDED.role,
                    assigned_officer_id = EXCLUDED.assigned_officer_id,
                    identities = EXCLUDED.identities;
                """,
                (
                    user_id,
                    user.get("name", "").strip(),
                    role,
                    user.get("assigned_officer_id"),
                    Jsonb(user.get("identities") or []),
                    user.get("created_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            )
        conn.commit()


def pg_create_user(
    name: str,
    role: str,
    identities: Optional[List[Dict[str, Any]]] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Creates and stores a new user record in PostgreSQL."""
    import accounts_store
    if role not in accounts_store.VALID_ROLES:
        raise ValueError(f"Invalid user role '{role}'. Must be one of: {', '.join(sorted(accounts_store.VALID_ROLES))}")

    uid = user_id or f"usr_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    normalized_identities: List[Dict[str, Any]] = []
    if identities:
        for ident in identities:
            if isinstance(ident, dict):
                normalized_identities.append(ident)
            elif isinstance(ident, str):
                normalized_identities.append({"type": "email", "identifier": ident})

    user = {
        "user_id": uid,
        "name": name.strip(),
        "role": role,
        "identities": normalized_identities,
        "created_at": now_iso,
    }
    pg_save_user(user)
    return user


def pg_link_identity_to_user(
    user_id: str,
    identity_type: str,
    identifier: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Links an additional auth identity (email, phone, google) to an existing user in PostgreSQL."""
    _ensure_init()
    target_type = identity_type.strip().lower()
    target_id = identifier.strip().lower()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE user_id = %s FOR UPDATE", (user_id,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"User with id '{user_id}' does not exist")

            user = _row_to_user(row)
            identities = user.setdefault("identities", [])

            already_linked = False
            for ident in identities:
                if isinstance(ident, dict):
                    if (
                        str(ident.get("type", "")).strip().lower() == target_type
                        and str(ident.get("identifier") or ident.get("value") or "").strip().lower() == target_id
                    ):
                        already_linked = True
                        break

            if not already_linked:
                new_identity = {"type": target_type, "identifier": target_id}
                if metadata:
                    new_identity.update(metadata)
                identities.append(new_identity)

                cur.execute(
                    "UPDATE users SET identities = %s WHERE user_id = %s",
                    (Jsonb(identities), user_id)
                )
        conn.commit()
    return user


def pg_save_users_db(users_db: Dict[str, Dict[str, Any]]) -> None:
    """
    Reconciles users table inside one atomic transaction.
    Stale-snapshot safe:
    - If empty, clears non-admin users.
    - If SnapshotDict:
        * Deletes ONLY users explicitly removed from this snapshot.
        * Upserts ONLY users modified or newly added in this snapshot.
        * Unmodified users are not written, preventing stale-snapshot overwrites.
    - If plain dict:
        * Upserts all users present in dict without deleting absent users.
    """
    _ensure_init()

    with get_connection() as conn:
        with conn.cursor() as cur:
            if len(users_db) == 0:
                cur.execute("DELETE FROM users WHERE user_id != 'usr_admin_master'")
                conn.commit()
                return

            if isinstance(users_db, SnapshotDict) and users_db._is_snapshot:
                current_keys = set(users_db.keys())
                deleted_keys = users_db._initial_keys - current_keys
                if deleted_keys:
                    cur.execute("DELETE FROM users WHERE user_id = ANY(%s)", (list(deleted_keys),))

                to_save = {}
                for uid, user in users_db.items():
                    if not isinstance(user, dict):
                        continue
                    if uid not in users_db._initial_keys or _record_fingerprint(user) != users_db._initial_fingerprints.get(uid):
                        to_save[uid] = user
            else:
                to_save = {uid: u for uid, u in users_db.items() if isinstance(u, dict)}

            for uid, user in to_save.items():
                cur.execute(
                    """
                    INSERT INTO users (user_id, name, role, assigned_officer_id, identities, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        role = EXCLUDED.role,
                        assigned_officer_id = EXCLUDED.assigned_officer_id,
                        identities = EXCLUDED.identities;
                    """,
                    (
                        uid,
                        user.get("name", "").strip(),
                        user.get("role", "user"),
                        user.get("assigned_officer_id"),
                        Jsonb(user.get("identities") or []),
                        user.get("created_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    )
                )
        conn.commit()


# =====================================================================
# 3. SESSIONS PERSISTENCE
# =====================================================================

def pg_load_sessions_db() -> Dict[str, Dict[str, Any]]:
    """Reads all sessions from PostgreSQL, returning a SnapshotDict."""
    _ensure_init()
    result = SnapshotDict()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM sessions")
            for r in cur.fetchall():
                result[r["session_token"]] = {
                    "session_token": r["session_token"],
                    "user_id": r["user_id"],
                    "created_at": r["created_at"],
                    "expires_at": r["expires_at"],
                }
    return result.freeze_snapshot()


def pg_create_session(
    user_id: str,
    duration_days: int = 7,
    custom_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Creates a new authenticated session in PostgreSQL."""
    _ensure_init()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=duration_days)
    token = custom_token or secrets.token_urlsafe(32)

    session = {
        "session_token": token,
        "user_id": user_id,
        "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at": expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (session_token, user_id, created_at, expires_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (session_token) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    expires_at = EXCLUDED.expires_at;
                """,
                (token, user_id, session["created_at"], session["expires_at"])
            )
        conn.commit()
    return session


def pg_get_session(session_token: str) -> Optional[Dict[str, Any]]:
    """Retrieves session if valid and unexpired; auto-deletes if expired."""
    if not session_token:
        return None
    _ensure_init()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM sessions WHERE session_token = %s", (session_token,))
            row = cur.fetchone()
            if not row:
                return None

            expires_at = _parse_iso_utc(row["expires_at"])
            if not expires_at or datetime.now(timezone.utc) >= expires_at:
                # Expired - delete atomically
                cur.execute("DELETE FROM sessions WHERE session_token = %s", (session_token,))
                conn.commit()
                return None

            return {
                "session_token": row["session_token"],
                "user_id": row["user_id"],
                "created_at": row["created_at"],
                "expires_at": row["expires_at"],
            }


def pg_delete_session(session_token: str) -> bool:
    """Deletes an active session from PostgreSQL."""
    if not session_token:
        return False
    _ensure_init()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE session_token = %s", (session_token,))
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted


def pg_save_sessions_db(sessions_db: Dict[str, Dict[str, Any]]) -> None:
    """
    Reconciles sessions in PostgreSQL inside one transaction.
    Stale-snapshot safe:
    - If empty, clears all sessions.
    - If SnapshotDict:
        * Deletes ONLY sessions explicitly removed from this snapshot.
        * Upserts ONLY sessions modified or newly added in this snapshot.
        * Unmodified sessions are not written.
    - If plain dict:
        * Upserts all sessions present without deleting absent ones.
    """
    _ensure_init()

    with get_connection() as conn:
        with conn.cursor() as cur:
            if len(sessions_db) == 0:
                cur.execute("DELETE FROM sessions")
                conn.commit()
                return

            if isinstance(sessions_db, SnapshotDict) and sessions_db._is_snapshot:
                current_tokens = set(sessions_db.keys())
                deleted_tokens = sessions_db._initial_keys - current_tokens
                if deleted_tokens:
                    cur.execute("DELETE FROM sessions WHERE session_token = ANY(%s)", (list(deleted_tokens),))

                to_save = {}
                for token, sess in sessions_db.items():
                    if not isinstance(sess, dict):
                        continue
                    if token not in sessions_db._initial_keys or _record_fingerprint(sess) != sessions_db._initial_fingerprints.get(token):
                        to_save[token] = sess
            else:
                to_save = {token: sess for token, sess in sessions_db.items() if isinstance(sess, dict)}

            for token, sess in to_save.items():
                cur.execute(
                    """
                    INSERT INTO sessions (session_token, user_id, created_at, expires_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (session_token) DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        expires_at = EXCLUDED.expires_at;
                    """,
                    (
                        token,
                        sess.get("user_id"),
                        sess.get("created_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        sess.get("expires_at") or (datetime.now(timezone.utc) + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    )
                )
        conn.commit()


# =====================================================================
# 4. AUTH CREDENTIALS PERSISTENCE
# =====================================================================

def pg_load_credentials_db() -> Dict[str, Dict[str, Any]]:
    """Reads all credentials from PostgreSQL, returning a SnapshotDict."""
    _ensure_init()
    result = SnapshotDict()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM auth_credentials")
            for r in cur.fetchall():
                result[r["identity_key"]] = {
                    "user_id": r["user_id"],
                    "password_hash": r["password_hash"],
                    "created_at": r["created_at"],
                }
    return result.freeze_snapshot()


def pg_save_credentials_db(creds: Dict[str, Dict[str, Any]]) -> None:
    """
    Reconciles auth_credentials in PostgreSQL inside one transaction.
    Stale-snapshot safe:
    - If empty, clears all credentials.
    - If SnapshotDict:
        * Deletes ONLY credentials explicitly removed from this snapshot.
        * Upserts ONLY credentials modified or newly added in this snapshot.
        * Unmodified credentials are not written.
    - If plain dict:
        * Upserts all credentials present without deleting absent ones.
    """
    _ensure_init()

    with get_connection() as conn:
        with conn.cursor() as cur:
            if len(creds) == 0:
                cur.execute("DELETE FROM auth_credentials")
                conn.commit()
                return

            if isinstance(creds, SnapshotDict) and creds._is_snapshot:
                current_keys = set(creds.keys())
                deleted_keys = creds._initial_keys - current_keys
                if deleted_keys:
                    cur.execute("DELETE FROM auth_credentials WHERE identity_key = ANY(%s)", (list(deleted_keys),))

                to_save = {}
                for key, c in creds.items():
                    if not isinstance(c, dict) or not c.get("user_id"):
                        continue
                    if key not in creds._initial_keys or _record_fingerprint(c) != creds._initial_fingerprints.get(key):
                        to_save[key] = c
            else:
                to_save = {key: c for key, c in creds.items() if isinstance(c, dict) and c.get("user_id")}

            for key, c in to_save.items():
                cur.execute(
                    """
                    INSERT INTO auth_credentials (identity_key, user_id, password_hash, created_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (identity_key) DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        password_hash = EXCLUDED.password_hash;
                    """,
                    (
                        key,
                        c["user_id"],
                        c.get("password_hash", ""),
                        c.get("created_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    )
                )
        conn.commit()


def pg_delete_user_account(user_id: str) -> Tuple[bool, str]:
    """
    Completely and atomically deletes a user account, their credentials,
    active sessions, and uploaded documents from PostgreSQL.
    Foreign key ON DELETE CASCADE handles sessions and credentials.
    """
    if not user_id:
        return False, "User ID is required."

    _ensure_init()
    user = pg_get_user(user_id)
    if not user:
        return False, f"User '{user_id}' not found."

    identities = user.get("identities", [])
    user_emails = [
        i.get("identifier") or i.get("value")
        for i in identities
        if isinstance(i, dict) and i.get("type") == "email"
    ]
    if user_id == "usr_admin_master" or "admin@admin.com" in user_emails or user.get("role") == "admin":
        return False, "The master administrator account (admin@admin.com) cannot be deleted."

    user_name = user.get("name") or user_id

    with get_connection() as conn:
        with conn.cursor() as cur:
            # 1. Delete verification records uploaded by this user
            cur.execute("DELETE FROM verification_records WHERE uploaded_by_user_id = %s", (user_id,))
            # 2. Delete active sessions for this user
            cur.execute("DELETE FROM sessions WHERE user_id = %s", (user_id,))
            # 3. Delete authentication credentials for this user
            cur.execute("DELETE FROM auth_credentials WHERE user_id = %s", (user_id,))
            # 4. Delete user record
            cur.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
        conn.commit()

    return True, f"Successfully deleted user '{user_name}' ({user_id}) and purged all associated data."
