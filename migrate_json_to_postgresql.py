#!/usr/bin/env python3
"""
migrate_json_to_postgresql.py - OneBhoomi Lossless JSON -> PostgreSQL Migration Engine

Migrates:
  1. users_db.json -> users
  2. auth_credentials.json -> auth_credentials
  3. sessions_db.json -> sessions
  4. verification_db.json -> verification_records

Guarantees:
  - Lossless nested preservation
  - Byte-identical preservation of canonical sealed payloads (BYTEA)
  - Preserves exact data types (e.g. "480" string remains "480")
  - Idempotent and restartable (UPSERT with conflict detection)
  - Never deletes or modifies source JSON files
  - Full transactional commit per store
  - Built-in verification / validation mode
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from psycopg.types.json import Jsonb

import postgres_store
import verification_service

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Safely loads and validates a JSON database file."""
    if not file_path.exists():
        print(f"[WARN] Source JSON file not found: {file_path}")
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Expected top-level JSON dict, got {type(data).__name__}")
            return data
    except Exception as exc:
        raise RuntimeError(f"Failed to read/parse {file_path}: {exc}")


def migrate_users(conn: psycopg.Connection, users_data: Dict[str, Any]) -> Tuple[int, int, int]:
    """Migrates users_db.json into users table idempotently."""
    inserted = 0
    updated = 0
    skipped = 0

    with conn.cursor() as cur:
        for uid, u in users_data.items():
            if not isinstance(u, dict) or not uid:
                skipped += 1
                continue

            name = (u.get("name") or uid).strip()
            role = u.get("role", "user")
            assigned_officer_id = u.get("assigned_officer_id")
            identities = u.get("identities") or []
            created_at = u.get("created_at") or "2026-09-12T00:00:00Z"

            cur.execute("SELECT name, role, identities FROM users WHERE user_id = %s", (uid,))
            existing = cur.fetchone()

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
                (uid, name, role, assigned_officer_id, Jsonb(identities), created_at)
            )
            if existing:
                updated += 1
            else:
                inserted += 1

    return inserted, updated, skipped


def migrate_credentials(conn: psycopg.Connection, creds_data: Dict[str, Any]) -> Tuple[int, int, int]:
    """Migrates auth_credentials.json into auth_credentials table idempotently."""
    inserted = 0
    updated = 0
    skipped = 0

    with conn.cursor() as cur:
        for ident_key, c in creds_data.items():
            if not isinstance(c, dict) or not ident_key:
                skipped += 1
                continue
            uid = c.get("user_id")
            pwd_hash = c.get("password_hash")
            if not uid or not pwd_hash:
                skipped += 1
                continue

            # Ensure referenced user exists in users table first
            cur.execute("SELECT 1 FROM users WHERE user_id = %s", (uid,))
            if not cur.fetchone():
                # Synthesize user if missing
                cur.execute(
                    """
                    INSERT INTO users (user_id, name, role, identities, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO NOTHING;
                    """,
                    (uid, uid, "user", Jsonb([]), c.get("created_at") or "2026-09-12T00:00:00Z")
                )

            cur.execute("SELECT 1 FROM auth_credentials WHERE identity_key = %s", (ident_key,))
            existing = cur.fetchone()

            cur.execute(
                """
                INSERT INTO auth_credentials (identity_key, user_id, password_hash, created_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (identity_key) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    password_hash = EXCLUDED.password_hash;
                """,
                (ident_key, uid, pwd_hash, c.get("created_at") or "2026-09-12T00:00:00Z")
            )
            if existing:
                updated += 1
            else:
                inserted += 1

    return inserted, updated, skipped


def migrate_sessions(conn: psycopg.Connection, sessions_data: Dict[str, Any]) -> Tuple[int, int, int]:
    """Migrates sessions_db.json into sessions table idempotently."""
    inserted = 0
    updated = 0
    skipped = 0

    with conn.cursor() as cur:
        for token, s in sessions_data.items():
            if not isinstance(s, dict) or not token:
                skipped += 1
                continue
            uid_raw = s.get("user_id")
            if not uid_raw:
                skipped += 1
                continue
            if isinstance(uid_raw, dict):
                uid = uid_raw.get("user_id") or "usr_unknown"
                user_name = uid_raw.get("name") or uid
                user_role = uid_raw.get("role") or "user"
            else:
                uid = str(uid_raw)

            cur.execute("SELECT 1 FROM sessions WHERE session_token = %s", (token,))
            existing = cur.fetchone()

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
                    uid,
                    s.get("created_at") or "2026-09-12T00:00:00Z",
                    s.get("expires_at") or "2026-09-19T00:00:00Z",
                )
            )
            if existing:
                updated += 1
            else:
                inserted += 1

    return inserted, updated, skipped


def migrate_verification_records(conn: psycopg.Connection, verif_data: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Migrates verification_db.json into verification_records table idempotently.
    Computes exact canonical sealed bytes for approved/signed records and stores as BYTEA.
    """
    inserted = 0
    updated = 0
    skipped = 0

    with conn.cursor() as cur:
        for vid, r in verif_data.items():
            if not isinstance(r, dict) or not vid:
                skipped += 1
                continue

            # Status and clerk_submitted logic
            status = r.get("status", "READY_FOR_APPROVAL")
            clerk_submitted = r.get("clerk_submitted")
            if clerk_submitted is None:
                clerk_submitted = status.upper() in {"APPROVED", "REJECTED", "DUPLICATE"}

            # Canonical sealed payload extraction
            canonical_bytes = None
            if status.upper() == "APPROVED" and r.get("signature") and r.get("document_payload"):
                canonical_bytes = verification_service.canonicalize_document(r["document_payload"])

            cur.execute("SELECT 1 FROM verification_records WHERE verification_id = %s", (vid,))
            existing = cur.fetchone()

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
                    status,
                    bool(r.get("is_land_document", False)),
                    r.get("file_hash"),
                    r.get("filename"),
                    r.get("uploaded_by_user_id"),
                    bool(clerk_submitted),
                    r.get("created_at") or "2026-09-12T00:00:00Z",
                    r.get("submitted_at"),
                    r.get("approved_at"),
                    r.get("signature"),
                    r.get("public_key"),
                    r.get("qr_code"),
                    Jsonb(r.get("decision")) if r.get("decision") is not None else None,
                    Jsonb(r.get("populated_fields") or []),
                    Jsonb(r.get("document_payload") or {}),
                    canonical_bytes,
                    Jsonb(r["raw_ocr"]) if r.get("raw_ocr") is not None else None,
                    Jsonb(r.get("field_provenance") or {}),
                    Jsonb(r.get("checks") or []),
                    Jsonb(r.get("duplicate_info")) if r.get("duplicate_info") is not None else None,
                    Jsonb(r.get("neural_nlp")) if r.get("neural_nlp") is not None else None,
                )
            )
            if existing:
                updated += 1
            else:
                inserted += 1

    return inserted, updated, skipped


def run_migration(base_dir: Path) -> Dict[str, Any]:
    """Executes the complete migration from JSON files to PostgreSQL."""
    print("=" * 70)
    print("ONEBHOOMI JSON -> POSTGRESQL MIGRATION ENGINE")
    print("=" * 70)

    # 1. Initialize schema
    postgres_store.init_db()

    # 2. Source files
    users_file = base_dir / "users_db.json"
    creds_file = base_dir / "auth_credentials.json"
    sessions_file = base_dir / "sessions_db.json"
    verif_file = base_dir / "verification_db.json"

    print(f"Reading source JSON files from: {base_dir.resolve()}")
    users_data = load_json_file(users_file)
    creds_data = load_json_file(creds_file)
    sessions_data = load_json_file(sessions_file)
    verif_data = load_json_file(verif_file)

    counts = {}

    with postgres_store.get_connection() as conn:
        # Atomic transaction across all entity migrations
        print("\n--> Migrating Users...")
        u_ins, u_upd, u_skp = migrate_users(conn, users_data)
        counts["users"] = {"inserted": u_ins, "updated": u_upd, "skipped": u_skp, "total": len(users_data)}
        print(f"    Users: {u_ins} inserted, {u_upd} updated, {u_skp} skipped (source: {len(users_data)})")

        print("--> Migrating Auth Credentials...")
        c_ins, c_upd, c_skp = migrate_credentials(conn, creds_data)
        counts["credentials"] = {"inserted": c_ins, "updated": c_upd, "skipped": c_skp, "total": len(creds_data)}
        print(f"    Credentials: {c_ins} inserted, {c_upd} updated, {c_skp} skipped (source: {len(creds_data)})")

        print("--> Migrating Sessions...")
        s_ins, s_upd, s_skp = migrate_sessions(conn, sessions_data)
        counts["sessions"] = {"inserted": s_ins, "updated": s_upd, "skipped": s_skp, "total": len(sessions_data)}
        print(f"    Sessions: {s_ins} inserted, {s_upd} updated, {s_skp} skipped (source: {len(sessions_data)})")

        print("--> Migrating Verification Records...")
        v_ins, v_upd, v_skp = migrate_verification_records(conn, verif_data)
        counts["verification_records"] = {"inserted": v_ins, "updated": v_upd, "skipped": v_skp, "total": len(verif_data)}
        print(f"    Verification Records: {v_ins} inserted, {v_upd} updated, {v_skp} skipped (source: {len(verif_data)})")

        conn.commit()

    print("\n[SUCCESS] Migration committed atomically to PostgreSQL.")
    return counts


def validate_migration(base_dir: Path) -> bool:
    """
    Validates logical equivalence between original JSON data and PostgreSQL reconstructed data.
    """
    print("\n" + "=" * 70)
    print("MIGRATION LOGICAL EQUIVALENCE VALIDATION")
    print("=" * 70)

    verif_file = base_dir / "verification_db.json"
    users_file = base_dir / "users_db.json"
    sessions_file = base_dir / "sessions_db.json"
    creds_file = base_dir / "auth_credentials.json"

    orig_verif = load_json_file(verif_file)
    orig_users = load_json_file(users_file)
    orig_sessions = load_json_file(sessions_file)
    orig_creds = load_json_file(creds_file)

    pg_verif = postgres_store.pg_load_db()
    pg_users = postgres_store.pg_load_users_db()
    pg_sessions = postgres_store.pg_load_sessions_db()
    pg_creds = postgres_store.pg_load_credentials_db()

    all_passed = True

    # 1. Validate Users
    print(f"\n1. Validating Users ({len(orig_users)} source vs {len(pg_users)} PG)...")
    for uid, u in orig_users.items():
        if uid not in pg_users:
            print(f"   [FAIL] User {uid} missing in PostgreSQL!")
            all_passed = False
            continue
        pg_u = pg_users[uid]
        assert pg_u["name"] == u["name"], f"Name mismatch for {uid}"
        assert pg_u["role"] == u["role"], f"Role mismatch for {uid}"
        assert pg_u["identities"] == u.get("identities", []), f"Identities mismatch for {uid}"
    print(f"   [PASS] All {len(orig_users)} users logically identical.")

    # 2. Validate Credentials
    print(f"\n2. Validating Credentials ({len(orig_creds)} source vs {len(pg_creds)} PG)...")
    for k, c in orig_creds.items():
        if k not in pg_creds:
            print(f"   [FAIL] Credential {k} missing in PostgreSQL!")
            all_passed = False
            continue
        pg_c = pg_creds[k]
        assert pg_c["user_id"] == c["user_id"], f"User ID mismatch for credential {k}"
        assert pg_c["password_hash"] == c["password_hash"], f"Password hash mismatch for {k}"
    print(f"   [PASS] All {len(orig_creds)} credentials logically identical.")

    # 3. Validate Sessions
    print(f"\n3. Validating Sessions ({len(orig_sessions)} source vs {len(pg_sessions)} PG)...")
    for tok, s in orig_sessions.items():
        if tok not in pg_sessions:
            print(f"   [FAIL] Session {tok} missing in PostgreSQL!")
            all_passed = False
            continue
        pg_s = pg_sessions[tok]
        orig_uid = s.get("user_id")
        expected_uid = orig_uid.get("user_id") if isinstance(orig_uid, dict) else orig_uid
        assert pg_s["user_id"] == expected_uid, f"Session user_id mismatch for {tok}"
        assert pg_s["expires_at"] == s["expires_at"], f"Session expires_at mismatch for {tok}"
    print(f"   [PASS] All {len(orig_sessions)} sessions logically identical.")

    # 4. Validate Verification Records
    print(f"\n4. Validating Verification Records ({len(orig_verif)} source vs {len(pg_verif)} PG)...")
    for vid, r in orig_verif.items():
        if vid not in pg_verif:
            print(f"   [FAIL] Record {vid} missing in PostgreSQL!")
            all_passed = False
            continue
        pg_r = pg_verif[vid]

        # Key fields comparison
        assert pg_r["status"] == r.get("status"), f"Status mismatch for {vid}"
        assert pg_r["is_land_document"] == r.get("is_land_document", False), f"is_land_doc mismatch for {vid}"
        assert pg_r["file_hash"] == r.get("file_hash"), f"file_hash mismatch for {vid}"
        assert pg_r["filename"] == r.get("filename"), f"filename mismatch for {vid}"
        assert pg_r["uploaded_by_user_id"] == r.get("uploaded_by_user_id"), f"uploaded_by mismatch for {vid}"
        assert pg_r["signature"] == r.get("signature"), f"signature mismatch for {vid}"
        assert pg_r["public_key"] == r.get("public_key"), f"public_key mismatch for {vid}"

        # Deep payload check
        assert pg_r["document_payload"] == (r.get("document_payload") or {}), f"document_payload mismatch for {vid}"

        # Raw OCR check (immutability)
        assert pg_r["raw_ocr"] == r.get("raw_ocr"), f"raw_ocr mismatch for {vid}"

        # Field provenance check
        assert pg_r["field_provenance"] == (r.get("field_provenance") or {}), f"field_provenance mismatch for {vid}"

        # Checks array
        assert pg_r["checks"] == (r.get("checks") or []), f"checks mismatch for {vid}"

    print(f"   [PASS] All {len(orig_verif)} verification records logically identical.")

    # 5. Validate Sealed Record Byte Identity (Phase 8 & Point 3 & Point 11)
    print("\n5. Validating Sealed Record Canonical Byte Identity (Phase 8)...")
    approved_count = 0
    for vid, r in orig_verif.items():
        if (r.get("status") or "").upper() == "APPROVED" and r.get("signature"):
            approved_count += 1
            pg_r = pg_verif[vid]

            # Original canonical bytes
            orig_bytes = verification_service.canonicalize_document(r["document_payload"])

            # Stored PostgreSQL BYTEA
            pg_bytes = pg_r.get("canonical_sealed_payload")

            assert pg_bytes is not None, f"Sealed record {vid} has no canonical_sealed_payload in PostgreSQL!"
            assert orig_bytes == pg_bytes, f"Byte mismatch for sealed record {vid}!"

            # Verify RSA-PSS signature against reconstructed PostgreSQL payload if real key present
            if pg_r.get("public_key") and pg_r.get("signature") and pg_r["signature"] != "mock-signature":
                sig_valid = verification_service.verify_document_signature(
                    pg_r["document_payload"],
                    pg_r["signature"],
                    pg_r["public_key"]
                )
                assert sig_valid is True, f"RSA signature failed on PG reconstructed payload for {vid}!"
                print(f"   [PASS] Record {vid}: BYTE-IDENTICAL ({len(orig_bytes)} bytes), Signature Verified: TRUE")
            else:
                print(f"   [PASS] Record {vid}: BYTE-IDENTICAL ({len(orig_bytes)} bytes), Mock test record preserved")

    print(f"   [PASS] All {approved_count} sealed records confirmed byte-identical.")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="OneBhoomi JSON to PostgreSQL Migration")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing data without re-migrating")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if not args.validate_only:
        counts = run_migration(base_dir)

    ok = validate_migration(base_dir)
    if ok:
        print("\n=======================================================")
        print("  🎉 MIGRATION & VALIDATION COMPLETED WITH 100% PASS")
        print("=======================================================\n")
        sys.exit(0)
    else:
        print("\n=======================================================")
        print("  ❌ MIGRATION VALIDATION DETECTED MISMATCHES")
        print("=======================================================\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
