"""
migrate_storage_to_encryption.py - In-Place Storage Encryption Migration CLI

Safely and idempotently migrates:
1. Persistent scanned document previews (scratch/preview_cache/*.html).
2. PostgreSQL verification_records sensitive payloads (document_payload, raw_ocr,
   field_provenance, neural_nlp, canonical_sealed_payload).

SAFETY & IDEMPOTENCY GUARANTEES:
- Detects legacy plaintext payloads vs. already encrypted envelopes.
- Never double-encrypts payloads.
- Atomic replacement for filesystem files via temporary files (.tmp_enc -> os.replace).
- Atomic transaction for PostgreSQL: entire DB migration is wrapped in a single transaction
  with automatic rollback on any failure.
- Verifies ciphertext roundtrip before committing any modification.
- Preserves all RSA-PSS signatures and verification keys completely untouched.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import postgres_store
import storage_encryption
from psycopg.types.json import Jsonb


def is_legacy_plaintext_file(raw_bytes: bytes) -> bool:
    """Checks if a document file is legacy plaintext (not starting with envelope magic)."""
    return not storage_encryption.is_encrypted_bytes(raw_bytes)


def migrate_preview_files(preview_dir: Path, dry_run: bool = False) -> Tuple[int, int, int]:
    """
    Migrates document preview cache files to storage encryption.
    Returns (inspected_count, migrated_count, skipped_count).
    """
    if not preview_dir.exists():
        print(f"[FILES] Directory {preview_dir} does not exist. Skipping.")
        return 0, 0, 0

    files = list(preview_dir.glob("*.html"))
    inspected = len(files)
    migrated = 0
    skipped = 0

    for p in files:
        try:
            raw_bytes = p.read_bytes()
            if not is_legacy_plaintext_file(raw_bytes):
                skipped += 1
                continue

            encrypted = storage_encryption.encrypt_bytes(raw_bytes)
            # Verification assertion before writing to disk
            decrypted = storage_encryption.decrypt_bytes(encrypted)
            if decrypted != raw_bytes:
                raise ValueError(f"Self-verification failed for file {p.name}")

            if not dry_run:
                tmp_file = p.with_suffix(".tmp_enc")
                tmp_file.write_bytes(encrypted)
                os.replace(tmp_file, p)
            migrated += 1
            action = "[DRY-RUN WOULD ENCRYPT]" if dry_run else "[MIGRATED]"
            print(f"  {action} {p.name} ({len(raw_bytes)} bytes -> {len(encrypted)} bytes)")
        except Exception as e:
            print(f"  [ERROR] Failed to migrate file {p.name}: {e}", file=sys.stderr)
            raise

    return inspected, migrated, skipped


def migrate_database_records(dry_run: bool = False) -> Tuple[int, int, int]:
    """
    Migrates sensitive payloads in PostgreSQL verification_records table.
    Wrapped in a single atomic transaction: rolls back completely on any error.
    Returns (inspected_count, migrated_count, skipped_count).
    """
    postgres_store.init_db()

    inspected = 0
    migrated = 0
    skipped = 0

    with postgres_store.get_connection() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT verification_id, document_payload, raw_ocr, field_provenance, neural_nlp, canonical_sealed_payload "
                    "FROM verification_records"
                )
                rows = cur.fetchall()
                inspected = len(rows)

                for r in rows:
                    vid = r["verification_id"]
                    doc_payload = r.get("document_payload")
                    raw_ocr = r.get("raw_ocr")
                    field_prov = r.get("field_provenance")
                    neural_nlp = r.get("neural_nlp")
                    canon_raw = r.get("canonical_sealed_payload")

                    needs_migration = False

                    new_doc = doc_payload
                    if doc_payload is not None and not storage_encryption.is_encrypted_json(doc_payload):
                        new_doc = storage_encryption.encrypt_json(doc_payload)
                        if storage_encryption.decrypt_json(new_doc) != doc_payload:
                            raise ValueError(f"Roundtrip check failed on document_payload for {vid}")
                        needs_migration = True

                    new_raw_ocr = raw_ocr
                    if raw_ocr is not None and not storage_encryption.is_encrypted_json(raw_ocr):
                        new_raw_ocr = storage_encryption.encrypt_json(raw_ocr)
                        if storage_encryption.decrypt_json(new_raw_ocr) != raw_ocr:
                            raise ValueError(f"Roundtrip check failed on raw_ocr for {vid}")
                        needs_migration = True

                    new_field_prov = field_prov
                    if field_prov is not None and not storage_encryption.is_encrypted_json(field_prov):
                        new_field_prov = storage_encryption.encrypt_json(field_prov)
                        if storage_encryption.decrypt_json(new_field_prov) != field_prov:
                            raise ValueError(f"Roundtrip check failed on field_provenance for {vid}")
                        needs_migration = True

                    new_neural = neural_nlp
                    if neural_nlp is not None and not storage_encryption.is_encrypted_json(neural_nlp):
                        new_neural = storage_encryption.encrypt_json(neural_nlp)
                        if storage_encryption.decrypt_json(new_neural) != neural_nlp:
                            raise ValueError(f"Roundtrip check failed on neural_nlp for {vid}")
                        needs_migration = True

                    new_canon = canon_raw
                    if canon_raw is not None:
                        b = bytes(canon_raw) if isinstance(canon_raw, memoryview) else canon_raw
                        if not storage_encryption.is_encrypted_bytes(b):
                            new_canon = storage_encryption.encrypt_bytes(b)
                            if storage_encryption.decrypt_bytes(new_canon) != b:
                                raise ValueError(f"Roundtrip check failed on canonical_sealed_payload for {vid}")
                            needs_migration = True

                    if needs_migration:
                        if not dry_run:
                            cur.execute(
                                """
                                UPDATE verification_records
                                SET document_payload = %s,
                                    raw_ocr = %s,
                                    field_provenance = %s,
                                    neural_nlp = %s,
                                    canonical_sealed_payload = %s
                                WHERE verification_id = %s
                                """,
                                (
                                    Jsonb(new_doc) if new_doc is not None else None,
                                    Jsonb(new_raw_ocr) if new_raw_ocr is not None else None,
                                    Jsonb(new_field_prov) if new_field_prov is not None else None,
                                    Jsonb(new_neural) if new_neural is not None else None,
                                    new_canon,
                                    vid,
                                ),
                            )
                        migrated += 1
                        action = "[DRY-RUN WOULD ENCRYPT]" if dry_run else "[MIGRATED]"
                        print(f"  {action} DB record {vid}")
                    else:
                        skipped += 1

                if not dry_run:
                    conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"[ERROR] Database migration aborted and rolled back: {e}", file=sys.stderr)
            raise

    return inspected, migrated, skipped


def main():
    parser = argparse.ArgumentParser(description="OneBhoomi Storage Encryption Migration Utility")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and report without modifying data")
    parser.add_argument("--files-only", action="store_true", help="Only migrate stored preview files")
    parser.add_argument("--db-only", action="store_true", help="Only migrate database records")
    parser.add_argument("--preview-dir", default="scratch/preview_cache", help="Path to preview cache directory")
    args = parser.parse_args()

    print("=" * 70)
    print("ONEBHOOMI STORAGE ENCRYPTION MIGRATION")
    print("Mode:", "DRY RUN (Read-Only)" if args.dry_run else "EXECUTE (In-Place Migration)")
    print("=" * 70)

    storage_encryption.ensure_storage_encryption_key()
    print(f"Active Encryption Key: {storage_encryption.get_storage_encryption_key_path()}")

    total_files_migrated = 0
    total_db_migrated = 0

    if not args.db_only:
        preview_path = Path(args.preview_dir)
        print(f"\n[1/2] Migrating Document Files ({preview_path})...")
        inspected, migrated, skipped = migrate_preview_files(preview_path, dry_run=args.dry_run)
        total_files_migrated = migrated
        print(f"  Files summary: {inspected} inspected, {migrated} migrated, {skipped} already encrypted")

    if not args.files_only:
        print("\n[2/2] Migrating PostgreSQL Database Records...")
        inspected, migrated, skipped = migrate_database_records(dry_run=args.dry_run)
        total_db_migrated = migrated
        print(f"  Database summary: {inspected} inspected, {migrated} migrated, {skipped} already encrypted")

    print("\n" + "=" * 70)
    print(f"MIGRATION COMPLETE: {total_files_migrated} files, {total_db_migrated} DB records migrated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
