"""
test_storage_encryption.py - Comprehensive Unit, Regression, and Migration Test Suite
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from cryptography.fernet import InvalidToken

import migrate_storage_to_encryption
import postgres_store
import storage_encryption
import verification_service
import web_app


class TestStorageEncryption(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["ONEBHOOMI_USE_POSTGRES"] = "true"
        postgres_store.init_db()
        storage_encryption.ensure_storage_encryption_key()

    # -------------------------------------------------------------
    # 1. CORE ENCRYPTION & FAIL-CLOSED TESTS
    # -------------------------------------------------------------
    def test_01_bytes_roundtrip(self):
        original = b"Confidential document bytes payload"
        enc = storage_encryption.encrypt_bytes(original)
        self.assertTrue(storage_encryption.is_encrypted_bytes(enc))
        self.assertNotEqual(original, enc)
        dec = storage_encryption.decrypt_bytes(enc)
        self.assertEqual(original, dec)

    def test_02_unexpected_plaintext_fails_closed(self):
        # Normal persistent storage reads must strictly reject plaintext
        with self.assertRaises(ValueError):
            storage_encryption.decrypt_bytes(b"<html>Plaintext Document</html>")

    def test_03_tampered_ciphertext_fails_closed(self):
        enc = bytearray(storage_encryption.encrypt_bytes(b"Sensitive deed"))
        enc[-4] ^= 0xFF
        with self.assertRaises(InvalidToken):
            storage_encryption.decrypt_bytes(bytes(enc))

    def test_04_empty_bytes_behavior(self):
        self.assertEqual(storage_encryption.encrypt_bytes(b""), b"")
        self.assertEqual(storage_encryption.decrypt_bytes(b""), b"")

    def test_05_binary_document_bytes_with_nulls(self):
        binary_data = b"%PDF-1.4\x00\x01\x02\xff\xfe\x00\x00BinaryStream\x00End"
        enc = storage_encryption.encrypt_bytes(binary_data)
        dec = storage_encryption.decrypt_bytes(enc)
        self.assertEqual(binary_data, dec)

    def test_06_large_document_payload(self):
        large_data = os.urandom(1024 * 1024)
        enc = storage_encryption.encrypt_bytes(large_data)
        dec = storage_encryption.decrypt_bytes(enc)
        self.assertEqual(large_data, dec)

    # -------------------------------------------------------------
    # 2. KEY SEPARATION TESTS
    # -------------------------------------------------------------
    def test_07_encryption_key_exists_independently(self):
        key_path = storage_encryption.get_storage_encryption_key_path()
        self.assertTrue(key_path.exists())
        key = storage_encryption.get_storage_encryption_key()
        self.assertEqual(len(key), 44)

    def test_08_encryption_key_is_not_rsa_key(self):
        key = storage_encryption.get_storage_encryption_key()
        rsa_path = verification_service.PRIVATE_KEY_PATH
        if rsa_path.exists():
            rsa_bytes = rsa_path.read_bytes()
            self.assertNotIn(key, rsa_bytes)
            self.assertFalse(key.startswith(b"-----BEGIN"))

    def test_09_rsa_signing_still_uses_existing_rsa_key(self):
        doc = {"document_number": "DOC-999", "property": {"survey_number": "100"}}
        sig = verification_service.sign_document(doc)
        pub_key = verification_service.get_public_verification_key()
        valid = verification_service.verify_document_signature(doc, sig, pub_key)
        self.assertTrue(valid)

    def test_10_encryption_module_does_not_import_rsa(self):
        import sys
        mod = sys.modules["storage_encryption"]
        self.assertFalse(hasattr(mod, "sign_document"))
        self.assertFalse(hasattr(mod, "verify_document_signature"))
        self.assertFalse(hasattr(mod, "rsa"))

    # -------------------------------------------------------------
    # 3. DOCUMENT STORAGE WRAPPER TESTS
    # -------------------------------------------------------------
    def test_11_stored_preview_is_ciphertext_on_disk(self):
        test_vid = "test_preview_enc_disk"
        test_html = "<html><body>Confidential Preview Image Base64 Data</body></html>"
        web_app.save_preview_html(test_vid, test_html)

        disk_file = web_app.PREVIEW_CACHE_DIR / f"{test_vid}.html"
        self.assertTrue(disk_file.exists())
        raw_on_disk = disk_file.read_bytes()
        self.assertTrue(storage_encryption.is_encrypted_bytes(raw_on_disk))
        self.assertNotIn(b"Confidential Preview Image", raw_on_disk)

    def test_12_normal_preview_read_returns_plaintext(self):
        test_vid = "test_preview_enc_read"
        test_html = "<html><body>Plaintext Document Content 12345</body></html>"
        web_app.save_preview_html(test_vid, test_html)

        with web_app._PREVIEW_LOCK:
            web_app.PREVIEW_CACHE.pop(test_vid, None)

        read_back = web_app.get_preview_html(test_vid)
        self.assertEqual(read_back, test_html)

    # -------------------------------------------------------------
    # 4. DATABASE ENCRYPTION TESTS
    # -------------------------------------------------------------
    def test_14_database_payloads_encrypted_at_rest(self):
        test_vid = "test_db_enc_rec_persist"
        record = {
            "verification_id": test_vid,
            "status": "READY_FOR_APPROVAL",
            "document_payload": {
                "document_number": "SECURE_DOC_888",
                "property": {"district": "Rangareddy", "survey_number": "42/B"},
            },
            "raw_ocr": {"full_text": "Sensitive transcribed land deed text"},
            "field_provenance": {"survey_number": {"confidence": 0.98}},
            "neural_nlp": {"entities": ["Rangareddy"]},
        }
        postgres_store.pg_save_record(record)

        with postgres_store.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT document_payload, raw_ocr FROM verification_records WHERE verification_id = %s",
                    (test_vid,),
                )
                row = cur.fetchone()
                self.assertTrue(storage_encryption.is_encrypted_json(row["document_payload"]))
                self.assertNotIn("SECURE_DOC_888", json.dumps(row["document_payload"]))
                self.assertTrue(storage_encryption.is_encrypted_json(row["raw_ocr"]))

    def test_15_database_read_returns_exact_plaintext_object(self):
        test_vid = "test_db_enc_rec_persist"
        loaded = postgres_store.pg_get_record(test_vid)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["document_payload"]["document_number"], "SECURE_DOC_888")
        self.assertEqual(loaded["document_payload"]["property"]["district"], "Rangareddy")
        self.assertEqual(loaded["raw_ocr"]["full_text"], "Sensitive transcribed land deed text")

    def test_16_database_read_fails_closed_on_raw_plaintext(self):
        # If an unencrypted record is inserted directly into DB, normal read fails closed
        test_vid = "test_raw_plaintext_tamper"
        with postgres_store.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO verification_records (verification_id, status, created_at, document_payload) "
                    "VALUES (%s, %s, %s, %s) ON CONFLICT (verification_id) DO UPDATE SET document_payload = EXCLUDED.document_payload",
                    (test_vid, "PENDING", "2026-09-15T00:00:00Z", '{"unencrypted": "raw_data"}'),
                )
            conn.commit()

        with self.assertRaises(ValueError):
            postgres_store.pg_get_record(test_vid)

    # -------------------------------------------------------------
    # 5. PREVIOUSLY SEALED RECORD MIGRATION & REGRESSION (POINTS 5 & 6)
    # -------------------------------------------------------------
    def test_18_previously_sealed_record_migration_regression(self):
        target_vid = "72059847-8ab2-4dd9-affa-f1019ccf7b7e"

        # 1. Capture BEFORE state directly from database
        with postgres_store.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM verification_records WHERE verification_id = %s", (target_vid,))
                row_before = cur.fetchone()

        if not row_before:
            self.skipTest(f"Sealed record {target_vid} not in database")

        orig_sig = row_before["signature"]
        orig_pub = row_before["public_key"]
        raw_doc_before = row_before["document_payload"]

        # Determine canonical plaintext payload before migration
        if storage_encryption.is_encrypted_json(raw_doc_before):
            orig_doc_payload = storage_encryption.decrypt_json(raw_doc_before)
        else:
            orig_doc_payload = raw_doc_before

        orig_canonical_bytes = verification_service.canonicalize_document(orig_doc_payload)

        # Verify signature BEFORE migration
        ok_before = verification_service.verify_document_signature(orig_doc_payload, orig_sig, orig_pub)
        self.assertTrue(ok_before, "Existing sealed record signature must verify before migration")

        # 2. Execute migration through actual migration path
        inspected, migrated, skipped = migrate_storage_to_encryption.migrate_database_records()

        # 3. Capture AFTER state directly from database
        with postgres_store.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM verification_records WHERE verification_id = %s", (target_vid,))
                row_after = cur.fetchone()

        # Verify ciphertext is now stored in the database column
        self.assertTrue(storage_encryption.is_encrypted_json(row_after["document_payload"]))

        # Direct assertions: signature string and public key are completely unchanged across migration
        self.assertEqual(row_after["signature"], orig_sig, "Signature string must not change during migration")
        self.assertEqual(row_after["public_key"], orig_pub, "Public key must not change during migration")

        # 4. Read through normal application persistence layer
        loaded_record = postgres_store.pg_get_record(target_vid)
        self.assertIsNotNone(loaded_record)

        loaded_canonical_bytes = verification_service.canonicalize_document(loaded_record["document_payload"])

        # Direct assertion: canonical plaintext payload is 100% identical
        self.assertEqual(orig_canonical_bytes, loaded_canonical_bytes, "Canonical plaintext payload must be identical")

        # 5. Verify the existing signature against the decrypted payload WITHOUT re-signing
        ok_after = verification_service.verify_document_signature(
            loaded_record["document_payload"], loaded_record["signature"], loaded_record["public_key"]
        )
        self.assertTrue(ok_after, "Existing sealed record must verify with original signature after migration")

    # -------------------------------------------------------------
    # 6. MIGRATION IDEMPOTENCY & SAFETY
    # -------------------------------------------------------------
    def test_19_migration_idempotency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            p = tmp_path / "test_preview.html"
            p.write_text("<html>Plaintext Preview Content</html>", encoding="utf-8")

            # Pass 1: Migrates file
            inspected, migrated, skipped = migrate_storage_to_encryption.migrate_preview_files(tmp_path)
            self.assertEqual(migrated, 1)
            self.assertEqual(skipped, 0)

            # Pass 2: Idempotent - must skip already encrypted file
            inspected2, migrated2, skipped2 = migrate_storage_to_encryption.migrate_preview_files(tmp_path)
            self.assertEqual(migrated2, 0)
            self.assertEqual(skipped2, 1)


if __name__ == "__main__":
    unittest.main()
