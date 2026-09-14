"""
test_postgres_persistence.py - PostgreSQL Persistence CRUD and Behavioral Equivalence Test
"""

import os
import unittest
import uuid
from datetime import datetime, timezone

import psycopg

import accounts_store
import auth_service
import postgres_store
import verification_service


class TestPostgresPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["ONEBHOOMI_USE_POSTGRES"] = "true"
        postgres_store.init_db()

    def test_01_verification_record_crud(self):
        vid = f"test_verif_{uuid.uuid4().hex[:8]}"
        record = {
            "verification_id": vid,
            "status": "READY_FOR_APPROVAL",
            "is_land_document": True,
            "file_hash": "hash_abc_123",
            "filename": "sample_deed.pdf",
            "uploaded_by_user_id": "usr_test_1",
            "document_payload": {
                "document_number": "1234/2024",
                "document_type": "Sale Deed",
                "property": {
                    "village": "Aushapur",
                    "area": "480",  # Preserve exact string representation!
                },
            },
            "raw_ocr": {"pages": [{"page_number": 1, "text": "SALE DEED"}]},
            "field_provenance": {"document_type": {"confidence": 0.95}},
            "checks": [{"name": "document_type", "status": "PASS"}],
            "duplicate_info": None,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        # 1. Save
        verification_service.save_record(record)

        # 2. Retrieve
        loaded = verification_service.get_record(vid)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["verification_id"], vid)
        self.assertEqual(loaded["status"], "READY_FOR_APPROVAL")
        self.assertEqual(loaded["document_payload"]["property"]["area"], "480")
        self.assertEqual(loaded["raw_ocr"]["pages"][0]["text"], "SALE DEED")
        self.assertFalse(loaded["clerk_submitted"])

        # 3. Update status to NEEDS_REVIEW with clerk submitted
        loaded["status"] = "NEEDS_REVIEW"
        loaded["clerk_submitted"] = True
        verification_service.save_record(loaded)

        updated = verification_service.get_record(vid)
        self.assertEqual(updated["status"], "NEEDS_REVIEW")
        self.assertTrue(updated["clerk_submitted"])

        # 4. Immutability test on APPROVED
        updated["status"] = "APPROVED"
        verification_service.save_record(updated)

        approved = verification_service.get_record(vid)
        self.assertEqual(approved["status"], "APPROVED")

        # Attempting to tamper with document_payload of approved record must raise ValueError
        tampered = dict(approved)
        tampered["document_payload"] = {"document_number": "TAMPERED_999"}
        with self.assertRaises(ValueError):
            verification_service.save_record(tampered)

    def test_02_users_and_credentials_crud(self):
        uid = f"usr_test_{uuid.uuid4().hex[:8]}"
        email = f"{uid}@telangana.gov.in"

        # 1. Create user
        user = accounts_store.create_user(
            name="Test Officer",
            role="officer",
            identities=[{"type": "email", "identifier": email}],
            user_id=uid,
        )
        self.assertEqual(user["user_id"], uid)
        self.assertEqual(user["role"], "officer")

        # 2. Retrieve user
        loaded = accounts_store.get_user(uid)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["name"], "Test Officer")

        # 3. Retrieve by identity
        by_ident = accounts_store.get_user_by_identity("email", email)
        self.assertIsNotNone(by_ident)
        self.assertEqual(by_ident["user_id"], uid)

        # 4. Link identity
        accounts_store.link_identity_to_user(uid, "phone", "+919876543210")
        with_phone = accounts_store.get_user(uid)
        ident_types = [i["type"] for i in with_phone["identities"]]
        self.assertIn("phone", ident_types)

        # 5. Credentials persistence
        creds = auth_service.load_credentials_db()
        creds[f"email:{email}"] = {
            "user_id": uid,
            "password_hash": auth_service.hash_password("SecretPass123!"),
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        auth_service.save_credentials_db(creds)

        reloaded_creds = auth_service.load_credentials_db()
        self.assertIn(f"email:{email}", reloaded_creds)
        self.assertTrue(auth_service.verify_password(reloaded_creds[f"email:{email}"]["password_hash"], "SecretPass123!"))

    def test_03_sessions_crud_and_expiry(self):
        uid = f"usr_sess_{uuid.uuid4().hex[:8]}"
        accounts_store.create_user(name="Session User", role="clerk", user_id=uid)

        # 1. Create session
        sess = accounts_store.create_session(user_id=uid, duration_days=7)
        tok = sess["session_token"]
        self.assertIsNotNone(tok)

        # 2. Get active session
        loaded = accounts_store.get_session(tok)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["user_id"], uid)

        # 3. Simulate expired session
        expired_sess = accounts_store.create_session(user_id=uid, duration_days=-1)
        exp_tok = expired_sess["session_token"]
        # Retrieval should lazily delete and return None
        res = accounts_store.get_session(exp_tok)
        self.assertIsNone(res)

        # 4. Explicit delete session
        ok = accounts_store.delete_session(tok)
        self.assertTrue(ok)
        self.assertIsNone(accounts_store.get_session(tok))


if __name__ == "__main__":
    unittest.main()
