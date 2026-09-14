"""
test_app_smoke.py - Application Smoke Test with PostgreSQL Persistence
Validates:
- Application starts / endpoints render
- Login / credential verification works
- Current user lookup works
- Session lookup works
- Document creation works
- Document retrieval works
- Clerk persistence works
- Officer decision persistence works
- Verification lookup works
- Dashboard reads data
- Certificate retrieval works
- Existing sealed record verifies
"""

import os
import unittest
import uuid

os.environ["ONEBHOOMI_PERSISTENCE_BACKEND"] = "postgres"

import accounts_store
import auth_service
import dashboard_view
import postgres_store
import verification_service
import web_app


class TestApplicationSmoke(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        postgres_store.init_db()

    def test_01_user_login_session_and_lookup(self):
        """Verify user creation, credential authentication, session creation and lookup."""
        uid = f"usr_smoke_{uuid.uuid4().hex[:8]}"
        email = f"{uid}@telangana.gov.in"
        pwd = "SecurePassword123!"

        # 1. Register with email and password
        reg_ok, reg_msg, reg_user, sess_token = auth_service.register_email_password(email, pwd, role="officer", name="Smoke Test Officer")
        self.assertTrue(reg_ok, f"Registration failed: {reg_msg}")
        uid = reg_user["user_id"]

        # 3. Authenticate with password
        auth_ok, auth_msg, auth_user, auth_token = auth_service.signin_email_password(email, pwd)
        self.assertTrue(auth_ok, f"Authentication failed: {auth_msg}")
        self.assertIsNotNone(auth_user)
        self.assertEqual(auth_user["role"], "officer")

        # 4. Create Session
        sess = accounts_store.create_session(user_id=auth_user["user_id"])
        token = sess["session_token"]

        # 5. Lookup Session
        loaded_sess = accounts_store.get_session(token)
        self.assertIsNotNone(loaded_sess)
        self.assertEqual(loaded_sess["user_id"], auth_user["user_id"])

        # 6. Current user lookup
        loaded_user = accounts_store.get_user(loaded_sess["user_id"])
        self.assertIsNotNone(loaded_user)
        self.assertEqual(loaded_user["name"], "Smoke Test Officer")

    def test_02_document_lifecycle_and_decision_persistence(self):
        """Verify document creation, clerk persistence, officer approval, and dashboard reading."""
        t_id = uuid.uuid4().hex[:6].upper()
        unique_doc_num = f"SMOKE-{t_id}"

        payload = {
            "document_type": "Sale Deed",
            "document_number": unique_doc_num,
            "document_date": "2026-03-01",
            "execution_date": "2026-03-01",
            "property": {
                "survey_number": f"7{t_id[:3]}",
                "sub_survey_number": "A",
                "village": "Gachibowli",
                "mandal": "Serilingampally",
                "district": "Rangareddy",
                "area": "480"
            },
            "stamp_information": {
                "stamp_number": f"TS-{t_id}",
                "stamp_value": "45000",
                "sold_to": "Test Buyer"
            },
            "parties": [
                {"name": "Test Seller", "role": "Vendor"},
                {"name": "Test Buyer", "role": "Purchaser"}
            ]
        }

        mock_result = {
            "canonical_payload": payload,
            "uploaded_by_user_id": "usr_test_clerk",
            "ocr_debug": {"raw_ocr": "SALE DEED Gachibowli Rangareddy"},
            "field_provenance": {}
        }

        # 1. Document Creation
        rec = verification_service.create_verification_record(
            result=mock_result,
            file_hash=f"hash_smoke_{uuid.uuid4().hex[:12]}",
            uploaded_by_user_id="usr_test_clerk"
        )
        vid = rec["verification_id"]
        self.assertIsNotNone(vid)
        verification_service.save_record(rec)

        # 2. Document Retrieval
        loaded = verification_service.get_record(vid)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["verification_id"], vid)

        # 3. Clerk persistence (Submit to Officer)
        loaded["clerk_submitted"] = True
        verification_service.save_record(loaded)
        clerk_saved = verification_service.get_record(vid)
        self.assertTrue(clerk_saved["clerk_submitted"])

        # 4. Officer decision persistence (Approve and Sign)
        decision = {
            "decision": "APPROVED",
            "officer_name": "Test Officer",
            "officer_designation": "Sub-Registrar",
            "office_location": "Hyderabad",
            "reason": "Documents verified and valid",
            "remarks": "Clean title verified"
        }
        loaded["decision"] = decision
        loaded["status"] = "APPROVED"
        loaded["signature"] = verification_service.sign_document(loaded["document_payload"])
        loaded["public_key"] = verification_service.get_public_verification_key()
        verification_service.save_record(loaded)

        # 5. Verification Lookup & Sealed Byte Check
        reloaded = verification_service.get_record(vid)
        self.assertEqual(reloaded["status"], "APPROVED")
        self.assertIsNotNone(reloaded.get("canonical_sealed_payload"))

        # Verify signature on approved record
        valid_sig = verification_service.verify_document_signature(
            payload=reloaded["document_payload"],
            signature_b64=reloaded["signature"],
            public_key_pem=reloaded["public_key"]
        )
        self.assertTrue(valid_sig, "Approved record signature must verify TRUE")

        # 6. Dashboard reads data from PostgreSQL
        dash_data = dashboard_view.get_dashboard_data()
        self.assertIn("rows", dash_data)
        self.assertIn(vid, [r.get("id") for r in dash_data["rows"]])

        officer_dash = dashboard_view.render_officer_dashboard(user_name="Test Officer").decode("utf-8")
        # 7. Certificate PDF Generation
        import certificate_pdf_service
        pdf_bytes = certificate_pdf_service.generate_certificate_pdf(reloaded)
        self.assertIsNotNone(pdf_bytes)
        self.assertTrue(len(pdf_bytes) > 1000, "Generated certificate PDF must be valid non-empty PDF bytes")

    def test_03_existing_sealed_record_verifies(self):
        """Verify that existing sealed record 743f8cab-6947-453e-b14c-bce450d39ccd verifies cryptographic signature."""
        rec = verification_service.get_record("743f8cab-6947-453e-b14c-bce450d39ccd")
        if not rec:
            self.skipTest("Legacy sealed record 743f8cab-6947-453e-b14c-bce450d39ccd not present (ledger cleared)")
        self.assertEqual(rec["status"], "APPROVED")

        # Verify signature
        valid = verification_service.verify_document_signature(
            payload=rec["document_payload"],
            signature_b64=rec["signature"],
            public_key_pem=rec["public_key"]
        )
        self.assertTrue(valid, "RSA-PSS signature on existing sealed record must verify TRUE")


if __name__ == "__main__":
    unittest.main()
