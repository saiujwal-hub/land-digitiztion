"""
test_sealed_record_bytes.py - Rigorous Byte-Level Sealed Record Identity Test

Tests real sealed record '743f8cab-6947-453e-b14c-bce450d39ccd':
1. Extracts canonical signed bytes from original verification_db.json
2. Retrieves the record through existing verification_service.get_record() from PostgreSQL
3. Compares the reconstructed bytes against the original bytes: before_bytes == after_bytes
4. Verifies the existing RSA-PSS signature against the reconstructed payload without re-signing.
"""

import json
import os
import unittest
from pathlib import Path

import postgres_store
import verification_service


class TestSealedRecordBytes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["ONEBHOOMI_USE_POSTGRES"] = "true"
        postgres_store.init_db()

    def test_real_sealed_record_byte_identity(self):
        json_path = Path("verification_db.json")
        self.assertTrue(json_path.exists(), "verification_db.json must exist as migration source")

        with open(json_path, "r", encoding="utf-8") as f:
            source_db = json.load(f)

        target_vid = "743f8cab-6947-453e-b14c-bce450d39ccd"
        if target_vid not in source_db:
            self.skipTest(f"Legacy record {target_vid} not present in verification_db.json (ledger cleared)")


        orig_rec = source_db[target_vid]
        self.assertEqual(orig_rec["status"], "APPROVED")
        self.assertIsNotNone(orig_rec.get("signature"))
        self.assertIsNotNone(orig_rec.get("public_key"))

        # Step 1: Canonicalize original document payload from JSON source
        before_bytes = verification_service.canonicalize_document(orig_rec["document_payload"])
        self.assertEqual(len(before_bytes), 1055, "Known canonical length for record 743f8cab must be 1055 bytes")

        # Step 2: Retrieve through existing application persistence function from PostgreSQL
        pg_rec = verification_service.get_record(target_vid)
        self.assertIsNotNone(pg_rec, "Record must be retrievable from PostgreSQL")

        # Step 3: Canonicalize reconstructed document payload from PostgreSQL
        after_bytes = verification_service.canonicalize_document(pg_rec["document_payload"])

        # Also retrieve the exact stored BYTEA from PostgreSQL
        pg_stored_bytea = pg_rec.get("canonical_sealed_payload")

        print("\n" + "=" * 75)
        print("SEALED RECORD BYTE-IDENTITY VERIFICATION")
        print("=" * 75)
        print(f"Record ID: {target_vid}")
        print(f"Original Byte Length     : {len(before_bytes)}")
        print(f"Reconstructed Byte Length: {len(after_bytes)}")
        print(f"PostgreSQL BYTEA Length  : {len(pg_stored_bytea) if pg_stored_bytea else 'N/A'}")
        print("\nBEFORE:")
        print(before_bytes.decode("utf-8"))
        print("\nAFTER:")
        print(after_bytes.decode("utf-8"))
        print("\nBYTE IDENTICAL:")
        print(before_bytes == after_bytes)
        print("=" * 75 + "\n")

        # Step 4: Prove byte identity
        self.assertEqual(before_bytes, after_bytes, "Before and after canonical bytes MUST be identical!")
        if pg_stored_bytea is not None:
            self.assertEqual(before_bytes, bytes(pg_stored_bytea), "BYTEA stored in PG must match canonical bytes!")

        # Step 5: Verify original RSA-PSS signature against reconstructed PostgreSQL payload (NO RE-SIGNING)
        sig_valid = verification_service.verify_document_signature(
            pg_rec["document_payload"],
            pg_rec["signature"],
            pg_rec["public_key"]
        )
        self.assertTrue(sig_valid, "Original RSA-PSS signature must verify successfully against reconstructed PG record!")


if __name__ == "__main__":
    unittest.main()
