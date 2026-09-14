"""
test_stale_snapshot_concurrency.py - Stale-Snapshot Concurrency and Conflict Tests

Tests against the PUBLIC compatibility API:
  - verification_service.load_db() / save_db()
  - accounts_store.load_users_db() / save_users_db()
  - accounts_store.load_sessions_db() / save_sessions_db()
  - auth_service.load_credentials_db() / save_credentials_db()

Verifies:
  1. Independent snapshots on different records: both updates survive (no lost update).
  2. Same-record concurrent modifications: deterministic last-write-wins, no record corruption,
     no field loss.
  3. Concurrent thread execution with independent snapshots across 20 iterations.
  4. Concurrent user, session, and credential modifications with independent snapshots.
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
import auth_service
import postgres_store
import verification_service


@pytest.fixture(autouse=True)
def ensure_postgres():
    os.environ["ONEBHOOMI_PERSISTENCE_BACKEND"] = "postgres"
    postgres_store.init_db()


def test_stale_snapshot_different_records_verification_db():
    """
    OBJECTIVE 1A:
    Thread A: snapshot_a = load_db()
    Thread B: snapshot_b = load_db()
    snapshot_a[id1]["status"] = "APPROVED"
    snapshot_b[id2]["status"] = "REJECTED"
    save_db(snapshot_a)
    save_db(snapshot_b)
    Verify: id1.status == APPROVED and id2.status == REJECTED.
    Both updates MUST survive.
    """
    id1 = f"test_stale_diff_1_{uuid.uuid4().hex[:8]}"
    id2 = f"test_stale_diff_2_{uuid.uuid4().hex[:8]}"

    # Seed two initial records
    rec1 = {
        "verification_id": id1,
        "status": "READY_FOR_APPROVAL",
        "is_land_document": True,
        "file_hash": "hash_111",
        "document_payload": {"district": "Rangareddy", "survey_number": "101"},
        "field_provenance": {"district": "ocr"},
        "raw_ocr": {"page1": "Raw OCR 1"},
        "created_at": "2026-09-01T10:00:00Z",
    }
    rec2 = {
        "verification_id": id2,
        "status": "READY_FOR_APPROVAL",
        "is_land_document": True,
        "file_hash": "hash_222",
        "document_payload": {"district": "Medchal", "survey_number": "202"},
        "field_provenance": {"district": "ocr"},
        "raw_ocr": {"page1": "Raw OCR 2"},
        "created_at": "2026-09-01T10:00:00Z",
    }
    verification_service.save_record(rec1)
    verification_service.save_record(rec2)

    # 1. Independent stale snapshots
    snapshot_a = verification_service.load_db()
    snapshot_b = verification_service.load_db()

    assert id1 in snapshot_a
    assert id2 in snapshot_a
    assert id1 in snapshot_b
    assert id2 in snapshot_b

    # 2. Modify distinct records in independent snapshots
    snapshot_a[id1]["status"] = "APPROVED"
    snapshot_b[id2]["status"] = "REJECTED"

    # 3. Save snapshot A, then save stale snapshot B
    verification_service.save_db(snapshot_a)
    verification_service.save_db(snapshot_b)

    # 4. Final state check: both updates must survive!
    final_db = verification_service.load_db()
    assert final_db[id1]["status"] == "APPROVED", f"Expected APPROVED for {id1}, got {final_db[id1]['status']}"
    assert final_db[id2]["status"] == "REJECTED", f"Expected REJECTED for {id2}, got {final_db[id2]['status']}"
    # Verify fields were preserved
    assert final_db[id1]["raw_ocr"] == {"page1": "Raw OCR 1"}
    assert final_db[id2]["raw_ocr"] == {"page1": "Raw OCR 2"}


def test_stale_snapshot_new_record_not_deleted():
    """
    Test that a record added concurrently after snapshot_b was loaded
    is NOT deleted when snapshot_b is saved.
    """
    id_old = f"test_stale_old_{uuid.uuid4().hex[:8]}"
    id_new = f"test_stale_new_{uuid.uuid4().hex[:8]}"

    rec_old = {
        "verification_id": id_old,
        "status": "PENDING",
        "document_payload": {"village": "A"},
    }
    verification_service.save_record(rec_old)

    # Snapshot B is taken BEFORE id_new is added
    snapshot_b = verification_service.load_db()
    assert id_old in snapshot_b
    assert id_new not in snapshot_b

    # Thread A adds a new record
    rec_new = {
        "verification_id": id_new,
        "status": "APPROVED",
        "document_payload": {"village": "B"},
    }
    verification_service.save_record(rec_new)

    # Thread B modifies id_old and saves its stale snapshot (which does NOT have id_new)
    snapshot_b[id_old]["status"] = "REVIEWED"
    verification_service.save_db(snapshot_b)

    # id_new must STILL exist in database!
    final_db = verification_service.load_db()
    assert id_new in final_db, "Concurrently added record was deleted by stale snapshot save!"
    assert final_db[id_old]["status"] == "REVIEWED"


def test_stale_snapshot_same_record_concurrency():
    """
    OBJECTIVE 1B:
    Both snapshots modify the SAME record.
    Verifies deterministic last-write-wins without corruption or lost fields.
    """
    rec_id = f"test_stale_same_{uuid.uuid4().hex[:8]}"
    initial_rec = {
        "verification_id": rec_id,
        "status": "PENDING",
        "document_payload": {"owner_name": "Original Owner", "extent": "5.0"},
        "field_provenance": {"owner_name": "ocr"},
        "raw_ocr": {"page1": "Raw Owner Text"},
    }
    verification_service.save_record(initial_rec)

    # Both take snapshots
    snap_a = verification_service.load_db()
    snap_b = verification_service.load_db()

    # Modify same record
    snap_a[rec_id]["status"] = "READY_FOR_APPROVAL"
    snap_a[rec_id]["document_payload"]["extent"] = "5.5"

    snap_b[rec_id]["status"] = "REJECTED"
    snap_b[rec_id]["document_payload"]["extent"] = "6.0"

    # Save A then save B -> B should win on this record (last-write-wins)
    verification_service.save_db(snap_a)
    verification_service.save_db(snap_b)

    final = verification_service.get_record(rec_id)
    assert final is not None
    assert final["status"] == "REJECTED"
    assert final["document_payload"]["extent"] == "6.0"
    # Verify uncorrupted and preserved fields
    assert final["raw_ocr"] == {"page1": "Raw Owner Text"}
    assert final["field_provenance"] == {"owner_name": "ocr"}


def test_stale_snapshot_users_db():
    """
    OBJECTIVE 1A with users:
    Independent snapshots modify distinct users -> both updates survive.
    """
    u1 = f"usr_stale_u1_{uuid.uuid4().hex[:8]}"
    u2 = f"usr_stale_u2_{uuid.uuid4().hex[:8]}"

    accounts_store.create_user(name="User One", role="user", user_id=u1)
    accounts_store.create_user(name="User Two", role="clerk", user_id=u2)

    # Independent snapshots
    snap_a = accounts_store.load_users_db()
    snap_b = accounts_store.load_users_db()

    snap_a[u1]["name"] = "User One Updated"
    snap_b[u2]["role"] = "officer"

    accounts_store.save_users_db(snap_a)
    accounts_store.save_users_db(snap_b)

    final_users = accounts_store.load_users_db()
    assert final_users[u1]["name"] == "User One Updated"
    assert final_users[u2]["role"] == "officer"


def test_stale_snapshot_sessions_db():
    """
    OBJECTIVE 1A with sessions:
    Independent snapshots modify distinct sessions -> both survive.
    """
    u = f"usr_sess_{uuid.uuid4().hex[:8]}"
    accounts_store.create_user(name="Sess User", role="user", user_id=u)

    tok1 = f"tok_stale_1_{uuid.uuid4().hex[:12]}"
    tok2 = f"tok_stale_2_{uuid.uuid4().hex[:12]}"

    accounts_store.create_session(user_id=u, custom_token=tok1)
    accounts_store.create_session(user_id=u, custom_token=tok2)

    snap_a = accounts_store.load_sessions_db()
    snap_b = accounts_store.load_sessions_db()

    new_exp_a = "2030-01-01T00:00:00Z"
    new_exp_b = "2031-01-01T00:00:00Z"
    snap_a[tok1]["expires_at"] = new_exp_a
    snap_b[tok2]["expires_at"] = new_exp_b

    accounts_store.save_sessions_db(snap_a)
    accounts_store.save_sessions_db(snap_b)

    final_sessions = accounts_store.load_sessions_db()
    assert final_sessions[tok1]["expires_at"] == new_exp_a
    assert final_sessions[tok2]["expires_at"] == new_exp_b


def test_stale_snapshot_credentials_db():
    """
    OBJECTIVE 1A with auth_credentials:
    Independent snapshots modify distinct credentials -> both survive.
    """
    u1 = f"usr_cred_1_{uuid.uuid4().hex[:8]}"
    u2 = f"usr_cred_2_{uuid.uuid4().hex[:8]}"
    accounts_store.create_user(name="Cred User 1", role="user", user_id=u1)
    accounts_store.create_user(name="Cred User 2", role="user", user_id=u2)

    k1 = f"email:cred1_{uuid.uuid4().hex[:6]}@test.com"
    k2 = f"email:cred2_{uuid.uuid4().hex[:6]}@test.com"

    auth_service.save_credentials_db({
        k1: {"user_id": u1, "password_hash": "hash_initial_1"},
        k2: {"user_id": u2, "password_hash": "hash_initial_2"},
    })

    snap_a = auth_service.load_credentials_db()
    snap_b = auth_service.load_credentials_db()

    snap_a[k1]["password_hash"] = "hash_updated_A"
    snap_b[k2]["password_hash"] = "hash_updated_B"

    auth_service.save_credentials_db(snap_a)
    auth_service.save_credentials_db(snap_b)

    final_creds = auth_service.load_credentials_db()
    assert final_creds[k1]["password_hash"] == "hash_updated_A"
    assert final_creds[k2]["password_hash"] == "hash_updated_B"


def test_twenty_iterations_concurrent_stale_snapshots():
    """
    OBJECTIVE 1D:
    Create tests with:
    * two threads
    * independent snapshots
    * different records
    * same record
    * at least 20 repeated iterations
    The tests must use the actual public persistence wrappers.
    Report: iterations, workers, conflicts, final-state correctness
    """
    ITERATIONS = 25
    WORKERS = 2
    conflicts_handled = 0

    for i in range(ITERATIONS):
        id_a = f"test_iter_{i}_a_{uuid.uuid4().hex[:6]}"
        id_b = f"test_iter_{i}_b_{uuid.uuid4().hex[:6]}"
        id_shared = f"test_iter_{i}_shared_{uuid.uuid4().hex[:6]}"

        # Seed initial records
        verification_service.save_record({
            "verification_id": id_a,
            "status": "PENDING",
            "document_payload": {"survey_number": f"100-{i}"},
        })
        verification_service.save_record({
            "verification_id": id_b,
            "status": "PENDING",
            "document_payload": {"survey_number": f"200-{i}"},
        })
        verification_service.save_record({
            "verification_id": id_shared,
            "status": "PENDING",
            "document_payload": {"survey_number": f"shared-{i}", "counter": 0},
        })

        barrier = threading.Barrier(2)

        def worker_a():
            snap = verification_service.load_db()
            barrier.wait()
            # Worker A modifies record A and shared record
            snap[id_a]["status"] = "APPROVED"
            snap[id_shared]["status"] = "APPROVED_BY_A"
            verification_service.save_db(snap)

        def worker_b():
            snap = verification_service.load_db()
            barrier.wait()
            # Worker B modifies record B and shared record
            snap[id_b]["status"] = "REJECTED"
            snap[id_shared]["status"] = "REJECTED_BY_B"
            verification_service.save_db(snap)

        t_a = threading.Thread(target=worker_a)
        t_b = threading.Thread(target=worker_b)
        t_a.start()
        t_b.start()
        t_a.join(timeout=5.0)
        t_b.join(timeout=5.0)

        assert not t_a.is_alive(), "Worker A timed out"
        assert not t_b.is_alive(), "Worker B timed out"

        # Verification of final state
        db = verification_service.load_db()
        assert db[id_a]["status"] == "APPROVED", f"Iteration {i}: Record A update was lost!"
        assert db[id_b]["status"] == "REJECTED", f"Iteration {i}: Record B update was lost!"

        # Shared record must be valid and reflect either A or B without corruption
        assert db[id_shared]["status"] in ("APPROVED_BY_A", "REJECTED_BY_B")
        conflicts_handled += 1

    print(f"\n--- Stale Snapshot Multi-Threaded Results ---")
    print(f"iterations: {ITERATIONS}")
    print(f"workers: {WORKERS}")
    print(f"conflicts: {conflicts_handled}")
    print(f"final-state correctness: 100% PASS")
