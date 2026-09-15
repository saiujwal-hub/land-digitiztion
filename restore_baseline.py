"""
restore_baseline.py - Restore 368-record baseline from pre-migration backup.

Guarantees:
1. Authoritative baseline source: backups/onebhoomi_pre_migration_backup.sql (368 records).
2. All populated sensitive payloads (document_payload, raw_ocr, field_provenance,
   neural_nlp, canonical_sealed_payload) are encrypted using active storage_encryption.key.
3. Logical plaintext is verified identical to backup plaintext before insertion.
4. Single atomic PostgreSQL transaction.
5. Exact ID-set comparison against the 368 backup IDs.
"""

import json
import os
import sys
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

import postgres_store
import storage_encryption

BACKUP_FILE = Path("backups/onebhoomi_pre_migration_backup.sql")

COLS = [
    "verification_id", "status", "is_land_document", "file_hash", "filename",
    "uploaded_by_user_id", "clerk_submitted", "created_at", "submitted_at",
    "approved_at", "signature", "public_key", "qr_code", "decision",
    "populated_fields", "document_payload", "canonical_sealed_payload",
    "raw_ocr", "field_provenance", "checks", "duplicate_info", "neural_nlp"
]

SENSITIVE_JSON_COLS = {
    "document_payload", "raw_ocr", "field_provenance", "neural_nlp"
}

OTHER_JSON_COLS = {
    "decision", "populated_fields", "checks", "duplicate_info"
}

BOOL_COLS = {
    "is_land_document", "clerk_submitted"
}


def unescape_pg(val: str):
    if val == r"\N":
        return None
    res = []
    i = 0
    n = len(val)
    while i < n:
        if val[i] == "\\" and i + 1 < n:
            c = val[i + 1]
            if c == "n":
                res.append("\n")
            elif c == "r":
                res.append("\r")
            elif c == "t":
                res.append("\t")
            elif c == "\\":
                res.append("\\")
            else:
                res.append(c)
            i += 2
        else:
            res.append(val[i])
            i += 1
    return "".join(res)


def parse_backup_records(backup_path: Path):
    with open(backup_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_copy = False
    raw_rows = []
    for line in lines:
        if line.startswith("COPY public.verification_records "):
            in_copy = True
            continue
        if in_copy:
            if line.strip() == r"\.":
                break
            parts = line.rstrip("\r\n").split("\t")
            assert len(parts) == len(COLS), f"Expected {len(COLS)} cols, got {len(parts)}"
            row_dict = {col: unescape_pg(raw) for col, raw in zip(COLS, parts)}
            raw_rows.append(row_dict)

    return raw_rows


def process_record(raw_row: dict) -> dict:
    processed = {}
    for col, val in raw_row.items():
        if val is None:
            processed[col] = None
        elif col in BOOL_COLS:
            processed[col] = (val == "t")
        elif col in OTHER_JSON_COLS:
            processed[col] = Jsonb(json.loads(val))
        elif col in SENSITIVE_JSON_COLS:
            obj = json.loads(val)
            if storage_encryption.is_encrypted_json(obj):
                processed[col] = Jsonb(obj)
            else:
                enc = storage_encryption.encrypt_json(obj)
                dec = storage_encryption.decrypt_json(enc)
                if dec != obj:
                    raise ValueError(f"Roundtrip check failed on {col} for {raw_row['verification_id']}")
                processed[col] = Jsonb(enc)
        elif col == "canonical_sealed_payload":
            if val.startswith("\\x"):
                raw_bytes = bytes.fromhex(val[2:])
            else:
                raw_bytes = val.encode("utf-8")
            if storage_encryption.is_encrypted_bytes(raw_bytes):
                processed[col] = raw_bytes
            else:
                enc_b = storage_encryption.encrypt_bytes(raw_bytes)
                dec_b = storage_encryption.decrypt_bytes(enc_b)
                if dec_b != raw_bytes:
                    raise ValueError(f"Roundtrip check failed on canonical_sealed_payload for {raw_row['verification_id']}")
                processed[col] = enc_b
        else:
            processed[col] = val
    return processed


def main():
    print(f"[1/4] Parsing backup file {BACKUP_FILE}...")
    raw_rows = parse_backup_records(BACKUP_FILE)
    print(f"      Parsed {len(raw_rows)} records from backup.")
    assert len(raw_rows) == 368, f"Expected 368 records, found {len(raw_rows)}"

    backup_ids = [r["verification_id"] for r in raw_rows]
    backup_id_set = set(backup_ids)
    assert len(backup_id_set) == 368, "Duplicate IDs found in backup"

    print("[2/4] Encrypting sensitive columns for baseline restoration...")
    processed_rows = []
    encrypted_payload_count = 0
    for r in raw_rows:
        proc = process_record(r)
        processed_rows.append(proc)
        for sc in SENSITIVE_JSON_COLS:
            if proc[sc] is not None:
                encrypted_payload_count += 1
        if proc["canonical_sealed_payload"] is not None:
            encrypted_payload_count += 1

    print(f"      Prepared {len(processed_rows)} rows ({encrypted_payload_count} encrypted sensitive payloads).")

    print("[3/4] Executing single atomic transaction restoration in PostgreSQL...")
    cols_str = ", ".join(COLS)
    placeholders = ", ".join(["%s"] * len(COLS))
    update_set = ", ".join([f"{c} = EXCLUDED.{c}" for c in COLS if c != "verification_id"])
    upsert_sql = f"""
        INSERT INTO verification_records ({cols_str})
        VALUES ({placeholders})
        ON CONFLICT (verification_id) DO UPDATE SET {update_set}
    """

    with postgres_store.get_connection() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                # Remove any extraneous records not in the authoritative 368 baseline
                cur.execute(
                    "DELETE FROM verification_records WHERE NOT (verification_id = ANY(%s))",
                    (backup_ids,)
                )
                deleted_extraneous = cur.rowcount
                print(f"      Removed {deleted_extraneous} extraneous/untracked record(s).")

                # Upsert all 368 rows
                for row_dict in processed_rows:
                    params = [row_dict[c] for c in COLS]
                    cur.execute(upsert_sql, params)

                cur.execute("SELECT COUNT(*) FROM verification_records")
                final_count = cur.fetchone()["count"]
                print(f"      Verification records in transaction: {final_count}")
                assert final_count == 368, f"Expected 368 records, got {final_count}"

    print("[4/4] Verifying restored database against backup baseline...")
    with postgres_store.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT verification_id FROM verification_records")
            current_ids = [r["verification_id"] for r in cur.fetchall()]

    current_id_set = set(current_ids)
    missing_ids = backup_id_set - current_id_set
    unexpected_ids = current_id_set - backup_id_set

    print(f"      Backup IDs: {len(backup_id_set)}")
    print(f"      Current IDs: {len(current_id_set)}")
    print(f"      Missing IDs: {len(missing_ids)}")
    print(f"      Unexpected IDs: {len(unexpected_ids)}")
    print(f"      Exact ID-Set Match: {'PASS' if backup_id_set == current_id_set else 'FAIL'}")

    assert backup_id_set == current_id_set, "ID set mismatch after restore!"
    print("\n[SUCCESS] RESTORATION SUCCESSFUL: 368 baseline records restored and verified!")


if __name__ == "__main__":
    main()
