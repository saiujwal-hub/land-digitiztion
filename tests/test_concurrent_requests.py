"""
tests/test_concurrent_requests.py - 10-Concurrent-Request Concurrency Test Suite

Validates that OneBhoomi's ThreadingHTTPServer safely handles concurrent requests
against the PostgreSQL persistence layer without changing route-handler logic,
preventing race conditions, lost updates, and database corruption.

Test Verification Coverage:
1. Exactly 10 requests were sent.
2. All 10 requests completed.
3. Successful responses were returned.
4. All expected writes exist in PostgreSQL afterward.
5. No expected write was lost.
6. No unexpected duplicate was created.
7. No database corruption occurred.
8. The persisted records contain valid data.
9. Actual concurrency proven via interval overlap and wall-clock vs sequential timing.
10. Test data cleanly purged from PostgreSQL afterward.
"""

import concurrent.futures
import json
import os
import sys
import threading
import time
import unittest
import uuid
from typing import Any, Dict, List

# Ensure virtual environment site-packages are prioritized
_venv_site = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "Lib", "site-packages")
if os.path.exists(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

# Ensure project root is in sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Force PostgreSQL backend
os.environ["ONEBHOOMI_PERSISTENCE_BACKEND"] = "postgres"
os.environ["ONEBHOOMI_USE_POSTGRES"] = "true"

import requests
from http.server import ThreadingHTTPServer
from web_app import LandExtractorHandler
import postgres_store
import auth_service


class TestConcurrentRequests(unittest.TestCase):
    server: ThreadingHTTPServer
    server_thread: threading.Thread
    base_url: str

    @classmethod
    def setUpClass(cls):
        postgres_store.init_db()
        # Bind ThreadingHTTPServer to an ephemeral port
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), LandExtractorHandler)
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

        # Warm-up check
        resp = requests.get(f"{cls.base_url}/auth/signin", timeout=5)
        assert resp.status_code == 200, f"Server failed to start, status={resp.status_code}"

    @classmethod
    def tearDownClass(cls):
        try:
            cls.server.shutdown()
            cls.server.server_close()
        except Exception:
            pass

    def test_ten_concurrent_requests_to_postgres_write_endpoint(self):
        """
        Sends exactly 10 concurrent HTTP POST requests to the /auth/signup write endpoint.
        Each request creates a unique user with credentials and session in PostgreSQL.
        Demonstrates actual concurrency, no lost writes, no duplicates, and clean teardown.
        """
        num_requests = 10
        run_id = uuid.uuid4().hex[:8]
        barrier = threading.Barrier(num_requests)
        endpoint = f"{self.base_url}/auth/signup"

        # Unique identifier per request
        test_payloads: List[Dict[str, Any]] = [
            {
                "index": i,
                "email": f"concur_{run_id}_{i}@revenue.telangana.gov.in",
                "password": f"SecurePassword123!_{i}",
                "name": f"Concurrent Officer {run_id}_{i}",
                "role": "officer",
            }
            for i in range(num_requests)
        ]

        def worker(req_data: Dict[str, Any]) -> Dict[str, Any]:
            # Synchronize all 10 worker threads so they hit the server simultaneously
            barrier.wait(timeout=10.0)

            post_body = {
                "auth_method": "password",
                "email": req_data["email"],
                "password": req_data["password"],
                "name": req_data["name"],
                "role": req_data["role"],
            }

            t_start = time.perf_counter()
            resp = requests.post(
                endpoint,
                json=post_body,
                headers={"Content-Type": "application/json"},
                allow_redirects=False,
                timeout=15.0,
            )
            t_end = time.perf_counter()

            return {
                "index": req_data["index"],
                "email": req_data["email"],
                "name": req_data["name"],
                "role": req_data["role"],
                "status_code": resp.status_code,
                "set_cookie": resp.headers.get("Set-Cookie", ""),
                "start_time": t_start,
                "end_time": t_end,
                "duration": t_end - t_start,
            }

        # Dispatch exactly 10 concurrent requests
        t_batch_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(worker, p) for p in test_payloads]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        t_batch_end = time.perf_counter()

        # -------------------------------------------------------------------
        # Verification 1 & 2: Exactly 10 requests were sent and completed
        # -------------------------------------------------------------------
        self.assertEqual(len(results), num_requests, f"Expected {num_requests} completed responses, got {len(results)}")

        # -------------------------------------------------------------------
        # Verification 3: Successful responses returned
        # -------------------------------------------------------------------
        for res in results:
            self.assertIn(
                res["status_code"],
                [200, 302, 303],
                f"Request {res['index']} returned unexpected HTTP status {res['status_code']}",
            )
            self.assertIn(
                "session_token=",
                res["set_cookie"],
                f"Request {res['index']} did not receive authenticated session cookie",
            )

        # -------------------------------------------------------------------
        # Verification 9: Concurrency proof (Task 7)
        # -------------------------------------------------------------------
        total_wall_time = t_batch_end - t_batch_start
        sum_durations = sum(r["duration"] for r in results)

        # Calculate interval overlaps between concurrent request executions
        overlaps = 0
        sorted_by_start = sorted(results, key=lambda x: x["start_time"])
        for i in range(len(sorted_by_start) - 1):
            if sorted_by_start[i]["end_time"] > sorted_by_start[i + 1]["start_time"]:
                overlaps += 1

        print(f"\n=======================================================")
        print(f"        10-CONCURRENT-REQUEST EXECUTION METRICS")
        print(f"=======================================================")
        print(f"  Requests Dispatched:       {num_requests}")
        print(f"  Requests Completed:        {len(results)}")
        print(f"  Batch Wall-Clock Duration: {total_wall_time * 1000:.2f} ms")
        print(f"  Sum of Request Durations:  {sum_durations * 1000:.2f} ms")
        print(f"  Concurrency Speedup:       {sum_durations / max(total_wall_time, 0.001):.2f}x")
        print(f"  Concurrent Overlaps:       {overlaps} / {num_requests - 1}")
        print(f"=======================================================")

        # Assert actual concurrency: wall time must be significantly less than sequential sum, with overlapping intervals
        self.assertGreater(overlaps, 0, "Expected requests to execute concurrently with overlapping time intervals")
        self.assertLess(total_wall_time, sum_durations, "Concurrent wall time must be less than sum of sequential durations")

        # -------------------------------------------------------------------
        # PostgreSQL Direct Verification (Task 6)
        # -------------------------------------------------------------------
        created_user_ids: List[str] = []
        try:
            with postgres_store.get_connection() as conn:
                with conn.cursor() as cur:
                    # Query users table
                    cur.execute(
                        "SELECT user_id, name, role, identities, created_at FROM users WHERE name LIKE %s ORDER BY created_at ASC",
                        (f"Concurrent Officer {run_id}_%",),
                    )
                    users_rows = cur.fetchall()

                    # Verification 4: All expected writes exist in PostgreSQL
                    self.assertEqual(
                        len(users_rows),
                        num_requests,
                        f"Expected {num_requests} user records in PostgreSQL, found {len(users_rows)}",
                    )

                    persisted_emails = set()
                    for r in users_rows:
                        uid = r["user_id"]
                        created_user_ids.append(uid)

                        # Verification 8: Persisted records contain valid data
                        self.assertTrue(uid.startswith("usr_"), f"Invalid user_id format: {uid}")
                        self.assertEqual(r["role"], "officer")
                        idents = r["identities"]
                        self.assertIsInstance(idents, list)
                        self.assertGreater(len(idents), 0)
                        email = idents[0].get("identifier")
                        self.assertIsNotNone(email)
                        persisted_emails.add(email)

                    # Verification 5: No expected write was lost
                    for expected in test_payloads:
                        self.assertIn(
                            expected["email"],
                            persisted_emails,
                            f"Expected write for {expected['email']} was lost from PostgreSQL!",
                        )

                    # Verification 6: No unexpected duplicate was created
                    self.assertEqual(
                        len(created_user_ids),
                        len(set(created_user_ids)),
                        "Duplicate user_id detected in PostgreSQL!",
                    )

                    # Verification 7 & 8: Verify credentials in auth_credentials table
                    cur.execute(
                        "SELECT identity_key, user_id, password_hash FROM auth_credentials WHERE user_id = ANY(%s)",
                        (created_user_ids,),
                    )
                    creds_rows = cur.fetchall()
                    self.assertEqual(
                        len(creds_rows),
                        num_requests,
                        f"Expected {num_requests} credential records, found {len(creds_rows)}",
                    )
                    for cr in creds_rows:
                        self.assertTrue(cr["identity_key"].startswith("email:concur_"))
                        self.assertTrue(len(cr["password_hash"]) > 20)

                    # Verification 7 & 8: Verify sessions in sessions table
                    cur.execute(
                        "SELECT session_token, user_id, created_at, expires_at FROM sessions WHERE user_id = ANY(%s)",
                        (created_user_ids,),
                    )
                    sess_rows = cur.fetchall()
                    self.assertEqual(
                        len(sess_rows),
                        num_requests,
                        f"Expected {num_requests} session records, found {len(sess_rows)}",
                    )
                    for sr in sess_rows:
                        self.assertTrue(len(sr["session_token"]) >= 32)
                        self.assertIn(sr["user_id"], created_user_ids)

            print(f"[POSTGRESQL PERSISTENCE & INTEGRITY]")
            print(f"  Users Table Writes:        {len(users_rows)} / {num_requests} [OK]")
            print(f"  Auth Credentials Writes:   {len(creds_rows)} / {num_requests} [OK]")
            print(f"  Sessions Table Writes:     {len(sess_rows)} / {num_requests} [OK]")
            print(f"  Database Corruption:       NONE DETECTED [OK]")
            print(f"  Data Validity:             100% VALIDATED [OK]")

        finally:
            # Clean up test data afterwards
            for uid in created_user_ids:
                try:
                    auth_service.delete_user_account(uid)
                except Exception as clean_err:
                    print(f"Warning: Cleanup failed for {uid}: {clean_err}")

            with postgres_store.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) as cnt FROM users WHERE user_id = ANY(%s)", (created_user_ids,))
                    remaining = cur.fetchone()["cnt"]
                    self.assertEqual(remaining, 0, "Test data cleanup incomplete: records remain in database")

            print(f"[CLEANUP]")
            print(f"  Purged Test Records:       {len(created_user_ids)} / {len(created_user_ids)} [100% PURGED]")
            print(f"=======================================================\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
