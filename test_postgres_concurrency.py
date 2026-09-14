"""
test_postgres_concurrency.py - Multi-Threaded PostgreSQL Concurrency Test

Verifies:
  a. Two different records updated concurrently (no lost updates, no whole-file rewrite race)
  b. Same record updated concurrently (transaction safety, no deadlocks)
  c. Concurrent session creation (atomic inserts, all tokens preserved)
  d. Concurrent user updates (atomic independent writes)
"""

import concurrent.futures
import os
import unittest
import uuid

import accounts_store
import postgres_store
import verification_service


class TestPostgresConcurrency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["ONEBHOOMI_USE_POSTGRES"] = "true"
        postgres_store.init_db()

    def test_a_concurrent_different_records_update(self):
        """Two different records updated concurrently must both persist without overwriting each other."""
        vid_a = f"concur_a_{uuid.uuid4().hex[:8]}"
        vid_b = f"concur_b_{uuid.uuid4().hex[:8]}"

        rec_a = {"verification_id": vid_a, "status": "READY_FOR_APPROVAL", "filename": "doc_a.pdf"}
        rec_b = {"verification_id": vid_b, "status": "READY_FOR_APPROVAL", "filename": "doc_b.pdf"}

        verification_service.save_record(rec_a)
        verification_service.save_record(rec_b)

        def update_record(vid, new_status):
            rec = verification_service.get_record(vid)
            rec["status"] = new_status
            verification_service.save_record(rec)
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_a = executor.submit(update_record, vid_a, "NEEDS_REVIEW")
            fut_b = executor.submit(update_record, vid_b, "REJECTED")

            self.assertTrue(fut_a.result())
            self.assertTrue(fut_b.result())

        # Verify both records have their respective updated statuses
        res_a = verification_service.get_record(vid_a)
        res_b = verification_service.get_record(vid_b)

        self.assertEqual(res_a["status"], "NEEDS_REVIEW")
        self.assertEqual(res_b["status"], "REJECTED")

    def test_b_concurrent_same_record_update(self):
        """Same record updated concurrently must complete transactionally without corrupting the store."""
        vid = f"concur_same_{uuid.uuid4().hex[:8]}"
        rec = {"verification_id": vid, "status": "READY_FOR_APPROVAL", "filename": "doc_same.pdf", "checks": []}
        verification_service.save_record(rec)

        def worker_append(idx):
            for i in range(5):
                r = verification_service.get_record(vid)
                r["filename"] = f"doc_updated_by_{idx}_{i}.pdf"
                verification_service.save_record(r)
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_append, i) for i in range(4)]
            for f in futures:
                self.assertTrue(f.result())

        final_rec = verification_service.get_record(vid)
        self.assertIsNotNone(final_rec)
        self.assertTrue(final_rec["filename"].startswith("doc_updated_by_"))

    def test_c_concurrent_session_creation(self):
        """Concurrent sessions created simultaneously must all persist uniquely."""
        uid = f"usr_concur_{uuid.uuid4().hex[:8]}"
        accounts_store.create_user(name="Concur User", role="user", user_id=uid)

        num_sessions = 16

        def create_sess_worker(idx):
            s = accounts_store.create_session(user_id=uid, duration_days=1)
            return s["session_token"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(create_sess_worker, i) for i in range(num_sessions)]
            tokens = [f.result() for f in futures]

        self.assertEqual(len(set(tokens)), num_sessions, "All tokens must be unique")

        # Verify every session is retrievable from PostgreSQL
        for tok in tokens:
            sess = accounts_store.get_session(tok)
            self.assertIsNotNone(sess, f"Session {tok} must exist in PostgreSQL")
            self.assertEqual(sess["user_id"], uid)

    def test_d_concurrent_user_updates(self):
        """Multiple independent user updates running simultaneously must not drop any update."""
        num_users = 8
        uids = []
        for i in range(num_users):
            uid = f"usr_concur_{i}_{uuid.uuid4().hex[:6]}"
            accounts_store.create_user(name=f"User {i}", role="clerk", user_id=uid)
            uids.append(uid)

        def update_user_worker(uid, new_name):
            u = accounts_store.get_user(uid)
            u["name"] = new_name
            accounts_store.save_user(u)
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(update_user_worker, uids[i], f"Updated Name {i}") for i in range(num_users)]
            for f in futures:
                self.assertTrue(f.result())

        # Verify all users have updated names
        for i in range(num_users):
            reloaded = accounts_store.get_user(uids[i])
            self.assertEqual(reloaded["name"], f"Updated Name {i}")


if __name__ == "__main__":
    unittest.main()
