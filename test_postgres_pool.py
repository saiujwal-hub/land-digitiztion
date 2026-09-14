"""
test_postgres_pool.py - Connection Pool Transaction Integrity, Concurrency, and Sanity Benchmark

Validates:
  1. Transaction rollback with connection pool (Objective 3B).
  2. Pool concurrency with 16 workers against max_size=4 (Objective 3C).
  3. Performance sanity check measuring sequential vs concurrent operations (Objective 9).
"""

import concurrent.futures
import os
import threading
import time
import uuid
from datetime import datetime, timezone
import pytest

os.environ["ONEBHOOMI_PERSISTENCE_BACKEND"] = "postgres"

import accounts_store
import postgres_store
import verification_service


@pytest.fixture(autouse=True)
def ensure_postgres():
    os.environ["ONEBHOOMI_PERSISTENCE_BACKEND"] = "postgres"
    postgres_store.init_db()


def test_pool_transaction_rollback_integrity():
    """
    OBJECTIVE 3B:
    Verify that an exception inside a pooled transaction causes an immediate
    ROLLBACK, leaves no uncommitted data, and does not contaminate subsequent
    connections from the pool.
    """
    test_id = f"test_rollback_{uuid.uuid4().hex[:8]}"

    # 1. Attempt a transaction that raises an error after an INSERT
    with pytest.raises(RuntimeError, match="Simulated crash during transaction"):
        with postgres_store.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO verification_records (verification_id, status, created_at)
                    VALUES (%s, %s, %s)
                    """,
                    (test_id, "PENDING", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                )
            # Do NOT commit; raise exception
            raise RuntimeError("Simulated crash during transaction")

    # 2. Verify that the record was rolled back and does not exist in DB
    rec = postgres_store.pg_get_record(test_id)
    assert rec is None, "Failed transaction was not rolled back; uncommitted data persisted!"

    # 3. Verify next connection from the pool is clean and unaffected
    with postgres_store.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 as val")
            row = cur.fetchone()
            assert row["val"] == 1


def test_pool_concurrency_stress_16_workers_pool_4():
    """
    OBJECTIVE 3C:
    Run concurrency test using more workers than the pool size.
    pool max size = 4
    workers = 16
    Operations:
      - verification reads
      - verification writes
      - user reads
      - session creation
      - session deletion
    Verify:
      - no deadlocks
      - no connection leaks
      - no transaction contamination
      - no corrupted state
      - all expected operations complete
    """
    # Configure pool to small size: max=4
    os.environ["POSTGRES_POOL_MIN_SIZE"] = "2"
    os.environ["POSTGRES_POOL_MAX_SIZE"] = "4"
    os.environ["POSTGRES_POOL_TIMEOUT"] = "15.0"
    postgres_store.reset_pool()

    NUM_WORKERS = 16
    OPERATIONS_PER_WORKER = 5
    total_operations = NUM_WORKERS * OPERATIONS_PER_WORKER * 5  # 5 op types per iteration
    failures = 0

    user_base = f"usr_stress_{uuid.uuid4().hex[:6]}"
    accounts_store.create_user(name="Stress Test User", role="user", user_id=user_base)

    def worker_routine(worker_idx: int):
        nonlocal failures
        worker_failures = 0
        for op in range(OPERATIONS_PER_WORKER):
            try:
                # 1. Verification Write
                v_id = f"verif_pool_{worker_idx}_{op}_{uuid.uuid4().hex[:6]}"
                verification_service.save_record({
                    "verification_id": v_id,
                    "status": "READY_FOR_APPROVAL",
                    "document_payload": {"survey_number": f"WN-{worker_idx}-{op}"},
                })

                # 2. Verification Read
                rec = verification_service.get_record(v_id)
                if not rec or rec["document_payload"].get("survey_number") != f"WN-{worker_idx}-{op}":
                    worker_failures += 1

                # 3. User Read
                u = accounts_store.get_user(user_base)
                if not u or u["name"] != "Stress Test User":
                    worker_failures += 1

                # 4. Session Creation
                tok = f"tok_stress_{worker_idx}_{op}_{uuid.uuid4().hex[:8]}"
                sess = accounts_store.create_session(user_id=user_base, custom_token=tok)
                if not sess or sess["session_token"] != tok:
                    worker_failures += 1

                # 5. Session Deletion
                deleted = accounts_store.delete_session(tok)
                if not deleted:
                    worker_failures += 1

            except Exception as e:
                print(f"Worker {worker_idx} exception on op {op}: {e}")
                worker_failures += 1
        return worker_failures

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(worker_routine, i) for i in range(NUM_WORKERS)]
        for f in concurrent.futures.as_completed(futures):
            failures += f.result()
    duration = time.time() - start_time

    stats = postgres_store.get_pool_stats()

    print("\n--- Connection Pool Concurrency Test Results ---")
    print(f"pool min size: 2")
    print(f"pool max size: 4")
    print(f"workers: {NUM_WORKERS}")
    print(f"operations: {total_operations}")
    print(f"failures: {failures}")
    print(f"duration: {duration:.2f}s")
    print(f"pool stats: {stats}")
    print(f"final state: {'CORRECT' if failures == 0 else 'CORRUPTED'}")

    assert failures == 0, f"Encountered {failures} failures during pool concurrency stress!"
    assert stats.get("requests_waiting", 0) == 0, "Connections were leaked; requests still waiting!"


def test_performance_sanity_check():
    """
    OBJECTIVE 9:
    Performance sanity check measuring actual sequential and concurrent persistence operations.
    """
    N = 50

    # 1. Sequential persistence operations
    start_seq = time.perf_counter()
    for i in range(N):
        vid = f"bench_seq_{i}_{uuid.uuid4().hex[:6]}"
        postgres_store.pg_save_record({
            "verification_id": vid,
            "status": "PENDING",
            "document_payload": {"index": i},
        })
        _ = postgres_store.pg_get_record(vid)
    seq_time = time.perf_counter() - start_seq
    seq_ops_per_sec = (N * 2) / seq_time

    # 2. Concurrent persistence operations (10 threads)
    def concurrent_op(i):
        vid = f"bench_conc_{i}_{uuid.uuid4().hex[:6]}"
        postgres_store.pg_save_record({
            "verification_id": vid,
            "status": "PENDING",
            "document_payload": {"index": i},
        })
        return postgres_store.pg_get_record(vid)

    start_conc = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(concurrent_op, range(N)))
    conc_time = time.perf_counter() - start_conc
    conc_ops_per_sec = (N * 2) / conc_time

    print(f"\n--- Performance Sanity Check ---")
    print(f"N: {N} records (write + read = {N * 2} ops)")
    print(f"Sequential Duration: {seq_time:.3f}s ({seq_ops_per_sec:.1f} ops/sec)")
    print(f"Concurrent Duration (10 threads): {conc_time:.3f}s ({conc_ops_per_sec:.1f} ops/sec)")

    assert len(results) == N
    assert all(r is not None for r in results)
