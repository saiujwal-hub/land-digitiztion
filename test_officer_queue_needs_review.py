import unittest
import uuid
import datetime
import verification_service
import dashboard_view

class TestOfficerQueueNeedsReview(unittest.TestCase):
    def setUp(self):
        self.suffix = uuid.uuid4().hex[:6]

    def _create_record(self, doc_num, status="READY_FOR_APPROVAL", submitted=True):
        payload = {
            "document_type": "Sale Deed",
            "document_number": doc_num,
            "property": {
                "survey_number": "202",
                "village": "Gachibowli",
                "mandal": "Serilingampally",
                "district": "Rangareddy",
                "area": "800 Sq.Yards"
            },
            "stamp_information": {"stamp_number": f"ST-{doc_num}"},
            "parties": [{"name": "Seller X", "role": "Seller"}, {"name": "Buyer Y", "role": "Buyer"}]
        }
        rec = verification_service.create_verification_record(
            result={"canonical_payload": payload, "field_provenance": {}},
            file_hash=f"hash_{uuid.uuid4().hex[:12]}",
            uploaded_by_user_id=f"clerk_{self.suffix}"
        )
        rec["status"] = status
        rec["document_payload"] = payload
        rec["checks"] = [{"status": "WARNING", "severity": "warning", "name": "Boundary Note"}] if status == "NEEDS_REVIEW" else []
        rec["clerk_submitted"] = submitted
        if submitted:
            rec["submitted_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        verification_service.save_record(rec)
        return rec

    def test_officer_queue_includes_ready_and_needs_review(self):
        doc1_num = f"DOC-READY-{self.suffix}"
        doc2_num = f"DOC-WARN-{self.suffix}"
        doc3_num = f"DOC-UNSUB-{self.suffix}"

        doc_ready = self._create_record(doc1_num, status="READY_FOR_APPROVAL", submitted=True)
        doc_warn = self._create_record(doc2_num, status="NEEDS_REVIEW", submitted=True)
        doc_unsub = self._create_record(doc3_num, status="READY_FOR_APPROVAL", submitted=False)

        officer_html = dashboard_view.render_officer_dashboard(
            user_name="Officer Sharma"
        ).decode("utf-8")

        # 1. Clerk-submitted READY_FOR_APPROVAL document MUST appear
        self.assertIn(doc_ready["verification_id"], officer_html)
        self.assertIn(doc1_num, officer_html)

        # 2. Clerk-submitted NEEDS_REVIEW document MUST appear
        self.assertIn(doc_warn["verification_id"], officer_html)
        self.assertIn(doc2_num, officer_html)

        # 3. Unsubmitted document MUST NOT appear
        self.assertNotIn(doc_unsub["verification_id"], officer_html)
        self.assertNotIn(doc3_num, officer_html)

if __name__ == "__main__":
    unittest.main()
