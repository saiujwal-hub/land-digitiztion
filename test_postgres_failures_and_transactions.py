"""
test_postgres_failures_and_transactions.py
Rigorous tests for:
1. No silent PostgreSQL -> JSON fallback (explicit failure when PostgreSQL fails).
2. Verification that PostgreSQL failures do NOT write to JSON files.
3. Explicit backend selection (ONEBHOOMI_PERSISTENCE_BACKEND=postgres vs json).
4. Transaction rollback on forced exception (database remains completely unchanged).
5. Atomic account deletion (delete_user_account): all user, session, creds, and records purged in one transaction, or none if aborted.
6. Concurrent account deletion with cascade.
"""

import concurrent.futures
import os
import unittest
import uuid
from pathlib import Path
import psycopg

import accounts_store
import auth_service
import postgres_store
import verification_service


class TestPostgresFailuresAndTransactions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ["ONEBHOOMI_PERSISTENCE_BACKEND"] = "postgres"
        postgres_store.init_db()

    def test_01_transaction_rollback_on_error(self):
        """A transaction that encounters an exception must roll back completely without partial state."""
        vid = f"rb_test_{uuid.uuid4().hex[:8]}"
        rec = {
            "verification_id": vid,
            "status": "READY_FOR_APPROVAL",
            "filename": "initial.pdf",
            "checks": []
        }
        verification_service.save_record(rec)

        # Confirm saved
        saved = verification_service.get_record(vid)
        self.assertEqual(saved["filename"], "initial.pdf")

        # Now attempt a transaction that updates the record and then raises an exception
        try:
            with postgres_store.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE verification_records SET filename = %s WHERE verification_id = %s",
                        ("corrupted_after_rollback.pdf", vid)
                    )
                    # Simulate forced failure before commit
                    raise RuntimeError("Simulated mid-transaction failure")
                conn.commit()
        except RuntimeError:
            pass

        # Verify the record still has the original value and rollback was 100% effective
        rechecked = verification_service.get_record(vid)
        self.assertEqual(rechecked["filename"], "initial.pdf", "Database MUST roll back on exception")

    def test_02_explicit_failure_when_postgres_fails_no_silent_fallback(self):
        """When PostgreSQL is the active backend, a DB failure must raise explicitly and NOT fall back to JSON."""
        # Save JSON file state
        orig_json_records = verification_service.load_db()

        # Temporarily point DATABASE_URL to a non-existent port
        real_db_url = os.environ.get("DATABASE_URL", "")
        real_host = os.environ.get("POSTGRES_HOST", "localhost")
        real_port = os.environ.get("POSTGRES_PORT", "5432")

        test_vid = f"fail_mode_{uuid.uuid4().hex[:8]}"
        fail_rec = {"verification_id": test_vid, "status": "READY_FOR_APPROVAL", "filename": "fail_test.pdf"}

        try:
            os.environ["ONEBHOOMI_PERSISTENCE_BACKEND"] = "postgres"
            os.environ["DATABASE_URL"] = "postgresql://postgres:wrongpass@127.0.0.1:54329/nonexistent_db_12345?connect_timeout=1"

            # Attempting save_record must FAIL explicitly (raise exception)
            with self.assertRaises((psycopg.OperationalError, psycopg.Error, Exception)):
                verification_service.save_record(fail_rec)

            # Attempting get_record must FAIL explicitly
            with self.assertRaises((psycopg.OperationalError, psycopg.Error, Exception)):
                verification_service.get_record(test_vid)

            # Attempting user creation must FAIL explicitly
            with self.assertRaises((psycopg.OperationalError, psycopg.Error, Exception)):
                accounts_store.create_user(name="Fail User", role="user")

        finally:
            # Restore valid database connection
            if real_db_url:
                os.environ["DATABASE_URL"] = real_db_url
            else:
                os.environ.pop("DATABASE_URL", None)
            os.environ["POSTGRES_HOST"] = real_host
            os.environ["POSTGRES_PORT"] = real_port

        # CRITICAL VERIFICATION: Verify nothing was silently written to verification_db.json!
        json_path = Path("verification_db.json")
        if json_path.exists():
            import json
            current_json = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertNotIn(test_vid, current_json, "PostgreSQL failure must NEVER silently write to JSON!")

    def test_03_backend_selection_json_vs_postgres(self):
        """ONEBHOOMI_PERSISTENCE_BACKEND determines backend unambiguously without silent switching."""
        os.environ["ONEBHOOMI_PERSISTENCE_BACKEND"] = "postgres"
        self.assertTrue(postgres_store.is_postgres_backend())

        os.environ["ONEBHOOMI_PERSISTENCE_BACKEND"] = "json"
        self.assertFalse(postgres_store.is_postgres_backend())

        # Reset back to postgres
        os.environ["ONEBHOOMI_PERSISTENCE_BACKEND"] = "postgres"
        self.assertTrue(postgres_store.is_postgres_backend())

    def test_04_atomic_cross_store_account_deletion(self):
        """delete_user_account must atomically remove user, credentials, sessions, and records in one transaction."""
        os.environ["ONEBHOOMI_PERSISTENCE_BACKEND"] = "postgres"
        uid = f"usr_del_{uuid.uuid4().hex[:8]}"
        email = f"test_{uid}@telangana.gov.in"

        # 1. Create user
        user = accounts_store.create_user(
            name="Delete Candidate",
            role="clerk",
            identities=[{"type": "email", "identifier": email}],
            user_id=uid
        )

        # 2. Create credential
        auth_service.save_credentials_db({
            f"email:{email}": {
                "user_id": uid,
                "password_hash": "argon2_mock_hash",
                "created_at": "2026-09-14T00:00:00Z"
            }
        })

        # 3. Create session
        sess = accounts_store.create_session(uid)
        token = sess["session_token"]

        # 4. Create document uploaded by this user
        vid = f"doc_del_{uuid.uuid4().hex[:8]}"
        verification_service.save_record({
            "verification_id": vid,
            "status": "READY_FOR_APPROVAL",
            "filename": "candidate.pdf",
            "uploaded_by_user_id": uid
        })

        # Verify all 4 exist
        self.assertIsNotNone(accounts_store.get_user(uid))
        self.assertIsNotNone(accounts_store.get_session(token))
        self.assertIsNotNone(verification_service.get_record(vid))
        creds = auth_service.load_credentials_db()
        self.assertIn(f"email:{email}", creds)

        # Execute atomic delete
        success, msg = auth_service.delete_user_account(uid)
        self.assertTrue(success)

        # Verify all 4 are completely gone
        self.assertIsNone(accounts_store.get_user(uid), "User must be deleted")
        self.assertIsNone(accounts_store.get_session(token), "Session must be purged")
        self.assertIsNone(verification_service.get_record(vid), "Uploaded record must be purged")
        creds_after = auth_service.load_credentials_db()
        self.assertNotIn(f"email:{email}", creds_after, "Credential must be purged")

    def test_05_concurrent_account_deletion(self):
        """Multiple users created with related data and concurrently deleted must not produce deadlocks or partial state."""
        os.environ["ONEBHOOMI_PERSISTENCE_BACKEND"] = "postgres"
        num_users = 4
        candidates = []

        for i in range(num_users):
            u_id = f"usr_cdel_{i}_{uuid.uuid4().hex[:6]}"
            em = f"{u_id}@telangana.gov.in"
            accounts_store.create_user(name=f"Cdel User {i}", role="user", identities=[{"type": "email", "identifier": em}], user_id=u_id)
            s = accounts_store.create_session(u_id)
            v = f"v_cdel_{i}_{uuid.uuid4().hex[:6]}"
            verification_service.save_record({"verification_id": v, "status": "READY_FOR_APPROVAL", "uploaded_by_user_id": u_id})
            candidates.append((u_id, s["session_token"], v, em))

        def delete_worker(candidate):
            u_id, token, vid, em = candidate
            return auth_service.delete_user_account(u_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(delete_worker, c) for c in candidates]
            results = [f.result() for f in futures]

        for ok, msg in results:
            self.assertTrue(ok)

        # Check all candidates cleanly purged
        for u_id, token, vid, em in candidates:
            self.assertIsNone(accounts_store.get_user(u_id))
            self.assertIsNone(accounts_store.get_session(token))
            self.assertIsNone(verification_service.get_record(vid))


if __name__ == "__main__":
    unittest.main()
