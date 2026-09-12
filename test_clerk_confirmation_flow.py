#!/usr/bin/env python3
"""
test_clerk_confirmation_flow.py
Verifies that:
1. A document that passes all machine checks (status == READY_FOR_APPROVAL) has clerk_submitted == False by default.
2. It does NOT appear in the Officer's pending queue (render_officer_dashboard) while clerk_submitted == False.
3. It appears in the Clerk's 'Needs your review' queue (render_clerk_dashboard) with an enabled 'Submit to Officer' button.
4. When the clerk clicks 'Submit to Officer' (POST action=submit_to_officer):
   - clerk_submitted becomes True
   - status remains READY_FOR_APPROVAL (no check re-computation)
5. It NOW appears in the Officer's pending queue (render_officer_dashboard) and in the Clerk's 'Sent to officer' section.
6. Legacy database records migration: APPROVED/REJECTED/DUPLICATE default to clerk_submitted=True, while READY_FOR_APPROVAL defaults to False.
"""

import copy
import json
import unittest
import urllib.parse
import urllib.request
import uuid

import accounts_store
import dashboard_view
import verification_service
import web_app


class TestClerkConfirmationFlow(unittest.TestCase):

    def setUp(self):
        # Create a test clerk and test officer if not exist
        self.clerk_user = accounts_store.get_user_by_identity("email", "clerk_test_confirmation@telangana.gov.in")
        if not self.clerk_user:
            self.clerk_user = accounts_store.create_user(
                name="Clerk Test User",
                role="clerk",
                identities=[{"type": "email", "identifier": "clerk_test_confirmation@telangana.gov.in"}]
            )
        self.clerk_id = self.clerk_user["user_id"]

        self.officer_user = accounts_store.get_user_by_identity("email", "officer_test_confirmation@telangana.gov.in")
        if not self.officer_user:
            self.officer_user = accounts_store.create_user(
                name="Officer Test User",
                role="officer",
                identities=[{"type": "email", "identifier": "officer_test_confirmation@telangana.gov.in"}]
            )
        self.officer_id = self.officer_user["user_id"]

    def test_end_to_end_clerk_confirmation(self):
        # 1. Create a verification record with passing checks (READY_FOR_APPROVAL)
        unique_doc_num = f"CONFIRM-{uuid.uuid4().hex[:8].upper()}"
        payload = {
            "document_type": "Sale Deed",
            "document_number": unique_doc_num,
            "document_date": "2026-03-01",
            "execution_date": "2026-03-01",
            "property": {
                "survey_number": "142",
                "sub_survey_number": "A",
                "village": "Gachibowli",
                "mandal": "Serilingampally",
                "district": "Rangareddy",
                "area": "1200 Sq.Yards"
            },
            "stamp_information": {
                "stamp_number": "TS-8849201",
                "stamp_value": "45000",
                "sold_to": "K. Srinivas"
            },
            "parties": [
                {"name": "P. Ramana", "role": "Seller"},
                {"name": "K. Srinivas", "role": "Buyer"}
            ]
        }

        mock_result = {
            "canonical_payload": payload,
            "uploaded_by_user_id": self.clerk_id,
            "ocr_debug": {"raw_ocr": "SALE DEED Gachibowli Rangareddy 142/A TS-8849201"},
            "field_provenance": {}
        }

        rec = verification_service.create_verification_record(
            result=mock_result,
            file_hash=f"hash_{uuid.uuid4().hex[:12]}",
            uploaded_by_user_id=self.clerk_id
        )

        # Force record to READY_FOR_APPROVAL to simulate passing machine checks
        rec["status"] = "READY_FOR_APPROVAL"
        rec["document_payload"] = payload
        rec["duplicate_info"] = None
        rec["checks"] = []

        # Assert default value of clerk_submitted is False
        self.assertIn("clerk_submitted", rec)
        self.assertFalse(rec["clerk_submitted"])

        # Save record to DB
        verification_service.save_record(rec)
        vid = rec["verification_id"]

        # 2. Verify Officer Dashboard does NOT include this unsubmitted record
        officer_html = dashboard_view.render_officer_dashboard(
            user_name=self.officer_user["name"]
        ).decode("utf-8")

        # The record should NOT appear in the officer's approval queue
        self.assertNotIn(f"/record?verification_id={vid}&role=officer", officer_html)
        self.assertNotIn(unique_doc_num, officer_html)

        # 3. Verify Clerk Dashboard places it into "Needs your review" (not "Sent to officer")
        clerk_html = dashboard_view.render_clerk_dashboard(
            user_id=self.clerk_id,
            user_name=self.clerk_user["name"]
        ).decode("utf-8")

        # The document is visible to the clerk in review section
        self.assertIn(vid, clerk_html)
        self.assertIn(unique_doc_num, clerk_html)
        self.assertIn("Checks passed · Ready to submit", clerk_html)

        # 4. Verify Clerk Panel (_clerk_panel) shows enabled "Submit to Officer" button
        clerk_panel_html = web_app._clerk_panel(
            record=rec,
            message="",
            role="clerk"
        )
        self.assertIn('name="action" value="submit_to_officer"', clerk_panel_html)
        self.assertIn('name="action" value="correct"', clerk_panel_html)
        # Should NOT be disabled since status is READY_FOR_APPROVAL
        self.assertNotIn('value="submit_to_officer" class="btn btn-green" disabled', clerk_panel_html)

        # 5. Simulate Clerk submitting to Officer via multipart POST request to /extract
        session_obj = accounts_store.create_session(self.clerk_id)
        session_token = session_obj["session_token"]

        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"action\"\r\n\r\nsubmit_to_officer\r\n",
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"verification_id\"\r\n\r\n{vid}\r\n",
            f"--{boundary}--\r\n",
        ]
        post_data = "".join(parts).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:8001/extract",
            data=post_data,
            headers={
                "Cookie": f"session_token={session_token}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            resp_body = resp.read().decode("utf-8")
            self.assertEqual(resp.status, 200)
            self.assertIn("Document submitted to Officer approval queue successfully", resp_body)

        # 6. Verify record in DB after submission
        updated_rec = verification_service.get_record(vid)
        self.assertTrue(updated_rec.get("clerk_submitted"))
        # Status MUST still be READY_FOR_APPROVAL
        self.assertEqual(updated_rec.get("status"), "READY_FOR_APPROVAL")

        # 7. Verify Officer Dashboard NOW displays this record in the queue!
        officer_html_after = dashboard_view.render_officer_dashboard(
            user_name=self.officer_user["name"]
        ).decode("utf-8")
        self.assertIn(f"/record?verification_id={vid}&role=officer", officer_html_after)
        self.assertIn(unique_doc_num, officer_html_after)

        # 8. Verify Clerk Dashboard NOW places it into "Sent to officer"
        clerk_html_after = dashboard_view.render_clerk_dashboard(
            user_id=self.clerk_id,
            user_name=self.clerk_user["name"]
        ).decode("utf-8")
        self.assertIn("Awaiting officer review &amp; legal seal", clerk_html_after)

        # 9. Verify Clerk Panel now displays the locked banner for the clerk
        clerk_panel_locked = web_app._clerk_panel(
            record=updated_rec,
            message="",
            role="clerk"
        )
        self.assertIn("Record Sent to Officer (Pending Approval)", clerk_panel_locked)

    def test_legacy_record_migration(self):
        # Verify legacy migration rules:
        # APPROVED, REJECTED, DUPLICATE -> clerk_submitted = True
        # READY_FOR_APPROVAL -> clerk_submitted = False
        db = verification_service.load_db()

        test_records = {
            "test_app": {"verification_id": "test_app", "status": "APPROVED"},
            "test_rej": {"verification_id": "test_rej", "status": "REJECTED"},
            "test_dup": {"verification_id": "test_dup", "status": "DUPLICATE"},
            "test_ready": {"verification_id": "test_ready", "status": "READY_FOR_APPROVAL"},
            "test_ext": {"verification_id": "test_ext", "status": "EXTRACTED"},
        }

        # Save these records directly without clerk_submitted
        for k, r in test_records.items():
            verification_service.save_record(r)

        # Reload DB and check how migration handled them
        loaded = verification_service.load_db()
        self.assertTrue(loaded["test_app"]["clerk_submitted"])
        self.assertTrue(loaded["test_rej"]["clerk_submitted"])
        self.assertTrue(loaded["test_dup"]["clerk_submitted"])
        self.assertFalse(loaded["test_ready"]["clerk_submitted"])
        self.assertFalse(loaded["test_ext"]["clerk_submitted"])

        # Clean up test records
        db_curr = verification_service.load_db()
        for k in test_records:
            db_curr.pop(k, None)
        verification_service.save_db(db_curr)


if __name__ == "__main__":
    unittest.main()
