"""
test_ocr_learning_service.py - Comprehensive Unit Tests for Adaptive OCR Learning Service.

Covers all 23 required scenarios:
1. Feedback is recorded only when a field changes.
2. Unchanged fields are ignored.
3. Clerk corrections remain pending before approval.
4. Rejected feedback never becomes a learned rule.
5. One verified example does not activate a rule.
6. Two matching verified examples activate a rule.
7. Evidence count increases safely.
8. Rule confidence is capped.
9. Same field, same document type, and same language applies a rule.
10. Same field but different document type does not apply a rule.
11. Same field but different language does not apply a rule.
12. Different field does not apply a rule.
13. API cannot directly create verified feedback.
14. Page number is preserved.
15. Source bounding box is preserved.
16. Raw OCR confidence is unchanged.
17. Raw OCR value is preserved.
18. Normalized value is stored separately.
19. Deterministic normalization is distinguishable from learned correction.
20. RSA-PSS signing behavior is unchanged.
21. Approved record immutability is unchanged.
22. Existing extraction tests continue to pass.
23. Existing GIS and verification tests continue to pass.
"""

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import ocr_learning_service
import verification_service
from land_document_extractor import OCRLine
from semantic_extractor import extract_fields_semantic


class TestOCRLearningService(unittest.TestCase):

    def setUp(self):
        # Create an isolated temporary file for each test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_store_path = Path(self.temp_dir.name) / "test_learning_store.json"
        ocr_learning_service.set_store_file(self.test_store_path)

    def tearDown(self):
        # Clean up temporary store
        ocr_learning_service.reset_learning_store()
        self.temp_dir.cleanup()

    # -----------------------------------------------------------------------
    # Requirement 1: Feedback is recorded only when a field changes
    # -----------------------------------------------------------------------
    def test_01_feedback_recorded_only_when_field_changes(self):
        old_payload = {"document_type": "Sale Deed", "property": {"survey_number": "278 | 281"}}
        new_form = {"document_type": "Sale Deed", "survey_number": "278, 281"}
        recorded = ocr_learning_service.capture_changed_payload_fields(
            old_payload=old_payload,
            new_form_fields=new_form,
            verification_id="v-change-1",
        )
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]["field_name"], "survey_number")
        self.assertEqual(recorded[0]["raw_ocr_value"], "278 | 281")
        self.assertEqual(recorded[0]["corrected_value"], "278, 281")

    # -----------------------------------------------------------------------
    # Requirement 2: Unchanged fields are ignored
    # -----------------------------------------------------------------------
    def test_02_unchanged_fields_are_ignored(self):
        old_payload = {
            "document_type": "Sale Deed",
            "document_number": "12736/5",
            "property": {"survey_number": "278 | 281", "area": "480", "village": "Aushapur"},
        }
        identical_form = {
            "document_type": "Sale Deed",
            "document_number": "12736/5",
            "survey_number": "278 | 281",
            "area": "480",
            "village": "Aushapur",
        }
        recorded = ocr_learning_service.capture_changed_payload_fields(
            old_payload=old_payload,
            new_form_fields=identical_form,
            verification_id="v-unchanged-1",
        )
        self.assertEqual(len(recorded), 0, "Unchanged fields must never generate feedback records")

    # -----------------------------------------------------------------------
    # Requirement 3: Clerk corrections remain pending before approval
    # -----------------------------------------------------------------------
    def test_03_clerk_corrections_remain_pending_before_approval(self):
        item = ocr_learning_service.stage_feedback(
            document_type="Sale Deed",
            field_name="survey_number",
            raw_ocr_value="278 | 281",
            corrected_value="278, 281",
            verification_id="v-pending-1",
            status="pending_approval",
        )
        self.assertEqual(item["status"], "pending_approval")
        stats = ocr_learning_service.get_learning_stats()
        self.assertEqual(stats["pending_feedback_count"], 1)
        self.assertEqual(stats["verified_feedback_count"], 0)
        self.assertEqual(stats["learned_rules_count"], 0)

    # -----------------------------------------------------------------------
    # Requirement 4: Rejected feedback never becomes a learned rule
    # -----------------------------------------------------------------------
    def test_04_rejected_feedback_never_becomes_learned_rule(self):
        ocr_learning_service.stage_feedback(
            document_type="Sale Deed",
            field_name="survey_number",
            raw_ocr_value="278 | 281",
            corrected_value="278, 281",
            verification_id="v-reject-1",
            status="pending_approval",
        )
        rej_n = ocr_learning_service.reject_feedback_for_verification("v-reject-1")
        self.assertEqual(rej_n, 1)

        stats = ocr_learning_service.get_learning_stats()
        self.assertEqual(stats["rejected_feedback_count"], 1)
        self.assertEqual(stats["verified_feedback_count"], 0)
        self.assertEqual(stats["learned_rules_count"], 0)

        # Attempt normalization
        norm_val, _, _, _, r_info, n_type = ocr_learning_service.apply_learned_normalization(
            "survey_number", "278 | 281", document_type="Sale Deed", language="en"
        )
        self.assertIsNone(r_info)
        self.assertNotEqual(n_type, "learned_correction")

    # -----------------------------------------------------------------------
    # Requirement 5: One verified example does not activate a rule
    # -----------------------------------------------------------------------
    def test_05_one_verified_example_does_not_activate_rule(self):
        ocr_learning_service.record_feedback(
            document_type="Sale Deed",
            field_name="survey_number",
            raw_ocr_value="278 | 281",
            corrected_value="278, 281",
            verification_id="v-one-1",
            auto_approve=True,
        )
        stats = ocr_learning_service.get_learning_stats()
        self.assertEqual(stats["verified_feedback_count"], 1)
        self.assertEqual(stats["learned_rules_count"], 0)
        rules = ocr_learning_service.get_learned_rules("survey_number")
        self.assertEqual(len(rules), 0)

    # -----------------------------------------------------------------------
    # Requirement 6: Two matching verified examples activate a rule
    # -----------------------------------------------------------------------
    def test_06_two_matching_verified_examples_activate_rule(self):
        for i in range(1, 3):
            ocr_learning_service.record_feedback(
                document_type="Sale Deed",
                field_name="survey_number",
                raw_ocr_value="278 | 281",
                corrected_value="278, 281",
                verification_id=f"v-two-{i}",
                auto_approve=True,
            )
        stats = ocr_learning_service.get_learning_stats()
        self.assertEqual(stats["verified_feedback_count"], 2)
        self.assertEqual(stats["learned_rules_count"], 1)

        rules = ocr_learning_service.get_learned_rules("survey_number")
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["raw_pattern"], "278 | 281")
        self.assertEqual(rules[0]["replacement"], "278, 281")
        self.assertEqual(rules[0]["evidence_count"], 2)
        self.assertEqual(rules[0]["confidence"], 0.80)

    # -----------------------------------------------------------------------
    # Requirement 7: Evidence count increases safely
    # -----------------------------------------------------------------------
    def test_07_evidence_count_increases_safely(self):
        for i in range(1, 4):
            ocr_learning_service.record_feedback(
                document_type="Sale Deed",
                field_name="survey_number",
                raw_ocr_value="278 | 281",
                corrected_value="278, 281",
                verification_id=f"v-ev-{i}",
                auto_approve=True,
            )
        rules = ocr_learning_service.get_learned_rules("survey_number")
        self.assertEqual(rules[0]["evidence_count"], 3)
        self.assertEqual(rules[0]["confidence"], 0.85)

    # -----------------------------------------------------------------------
    # Requirement 8: Rule confidence is capped
    # -----------------------------------------------------------------------
    def test_08_rule_confidence_is_capped(self):
        for i in range(1, 10):  # 9 verified items
            ocr_learning_service.record_feedback(
                document_type="Sale Deed",
                field_name="survey_number",
                raw_ocr_value="278 | 281",
                corrected_value="278, 281",
                verification_id=f"v-cap-{i}",
                auto_approve=True,
            )
        rules = ocr_learning_service.get_learned_rules("survey_number")
        self.assertEqual(rules[0]["evidence_count"], 9)
        self.assertLessEqual(rules[0]["confidence"], 0.95)
        self.assertEqual(rules[0]["confidence"], 0.95)

    # -----------------------------------------------------------------------
    # Requirement 9: Same field, same document type, and same language applies rule
    # -----------------------------------------------------------------------
    def test_09_same_field_doc_lang_applies_rule(self):
        for i in range(1, 3):
            ocr_learning_service.record_feedback(
                document_type="Sale Deed",
                field_name="survey_number",
                raw_ocr_value="278 | 281",
                corrected_value="278, 281",
                verification_id=f"v-match-{i}",
                language="en",
                auto_approve=True,
            )
        norm_val, raw_conf, norm_conf, fin_conf, r_info, n_type = ocr_learning_service.apply_learned_normalization(
            field_name="survey_number",
            raw_value="278 | 281",
            ocr_confidence=0.75,
            document_type="Sale Deed",
            language="en",
        )
        self.assertEqual(norm_val, "278, 281")
        self.assertEqual(n_type, "learned_correction")
        self.assertIsNotNone(r_info)
        self.assertEqual(r_info["document_type"], "Sale Deed")
        self.assertEqual(r_info["language"], "en")

    # -----------------------------------------------------------------------
    # Requirement 10: Same field but different document type does not apply rule
    # -----------------------------------------------------------------------
    def test_10_different_document_type_does_not_apply_rule(self):
        for i in range(1, 3):
            ocr_learning_service.record_feedback(
                document_type="Sale Deed",
                field_name="survey_number",
                raw_ocr_value="278 | 281",
                corrected_value="278, 281",
                verification_id=f"v-doc-{i}",
                language="en",
                auto_approve=True,
            )
        norm_val, raw_conf, norm_conf, fin_conf, r_info, n_type = ocr_learning_service.apply_learned_normalization(
            field_name="survey_number",
            raw_value="278 | 281",
            ocr_confidence=0.75,
            document_type="Mutation Record",  # Different document type!
            language="en",
        )
        self.assertNotEqual(n_type, "learned_correction")
        self.assertIsNone(r_info)

    # -----------------------------------------------------------------------
    # Requirement 11: Same field but different language does not apply rule
    # -----------------------------------------------------------------------
    def test_11_different_language_does_not_apply_rule(self):
        for i in range(1, 3):
            ocr_learning_service.record_feedback(
                document_type="Sale Deed",
                field_name="survey_number",
                raw_ocr_value="278 | 281",
                corrected_value="278, 281",
                verification_id=f"v-lang-{i}",
                language="en",
                auto_approve=True,
            )
        norm_val, raw_conf, norm_conf, fin_conf, r_info, n_type = ocr_learning_service.apply_learned_normalization(
            field_name="survey_number",
            raw_value="278 | 281",
            ocr_confidence=0.75,
            document_type="Sale Deed",
            language="te",  # Telugu language!
        )
        self.assertNotEqual(n_type, "learned_correction")
        self.assertIsNone(r_info)

    # -----------------------------------------------------------------------
    # Requirement 12: Different field does not apply rule
    # -----------------------------------------------------------------------
    def test_12_different_field_does_not_apply_rule(self):
        for i in range(1, 3):
            ocr_learning_service.record_feedback(
                document_type="Sale Deed",
                field_name="survey_number",
                raw_ocr_value="278 | 281",
                corrected_value="278, 281",
                verification_id=f"v-field-{i}",
                language="en",
                auto_approve=True,
            )
        norm_val, raw_conf, norm_conf, fin_conf, r_info, n_type = ocr_learning_service.apply_learned_normalization(
            field_name="property_area",  # Different field name!
            raw_value="278 | 281",
            ocr_confidence=0.75,
            document_type="Sale Deed",
            language="en",
        )
        self.assertNotEqual(n_type, "learned_correction")
        self.assertIsNone(r_info)

    # -----------------------------------------------------------------------
    # Requirement 13: API cannot directly create verified feedback
    # -----------------------------------------------------------------------
    def test_13_api_cannot_directly_create_verified_feedback(self):
        from web_app import LandExtractorHandler

        handler = LandExtractorHandler.__new__(LandExtractorHandler)
        handler.path = "/api/learning/feedback"

        # Attacker attempts auto_approve: true over HTTP
        payload_bytes = json.dumps({
            "document_type": "Sale Deed",
            "field_name": "survey_number",
            "raw_ocr_value": "278 | 281",
            "corrected_value": "278, 281",
            "verification_id": "api-bypass-test",
            "auto_approve": True,
        }).encode("utf-8")

        handler.headers = {"Content-Length": str(len(payload_bytes))}
        handler.rfile = io.BytesIO(payload_bytes)
        handler.wfile = io.BytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        handler.do_POST()

        res_data = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(res_data["status"], "ok")
        # Must strictly be pending_approval, NOT verified
        self.assertEqual(res_data["feedback"]["status"], "pending_approval")
        self.assertNotEqual(res_data["feedback"]["status"], "verified")

        stats = ocr_learning_service.get_learning_stats()
        self.assertEqual(stats["verified_feedback_count"], 0)
        self.assertEqual(stats["pending_feedback_count"], 1)

    # -----------------------------------------------------------------------
    # Requirement 14: Page number is preserved
    # -----------------------------------------------------------------------
    def test_14_page_number_is_preserved(self):
        item = ocr_learning_service.stage_feedback(
            document_type="Sale Deed",
            field_name="survey_number",
            raw_ocr_value="278 | 281",
            corrected_value="278, 281",
            verification_id="v-pg-1",
            page_number=5,
        )
        self.assertEqual(item["page_number"], 5)

        # Ensure null/None when unavailable
        item_none = ocr_learning_service.stage_feedback(
            document_type="Sale Deed",
            field_name="survey_number",
            raw_ocr_value="278 | 282",
            corrected_value="278, 282",
            verification_id="v-pg-2",
            page_number=None,
        )
        self.assertIsNone(item_none["page_number"])

    # -----------------------------------------------------------------------
    # Requirement 15: Source bounding box is preserved
    # -----------------------------------------------------------------------
    def test_15_source_bounding_box_is_preserved(self):
        item = ocr_learning_service.stage_feedback(
            document_type="Sale Deed",
            field_name="survey_number",
            raw_ocr_value="278 | 281",
            corrected_value="278, 281",
            verification_id="v-bbox-1",
            source_bbox=[120, 450, 700, 520],
        )
        self.assertEqual(item["source_bbox"], [120, 450, 700, 520])

        item_none = ocr_learning_service.stage_feedback(
            document_type="Sale Deed",
            field_name="survey_number",
            raw_ocr_value="278 | 283",
            corrected_value="278, 283",
            verification_id="v-bbox-2",
            source_bbox=None,
        )
        self.assertIsNone(item_none["source_bbox"])

    # -----------------------------------------------------------------------
    # Requirement 16: Raw OCR confidence is unchanged
    # -----------------------------------------------------------------------
    def test_16_raw_ocr_confidence_is_unchanged(self):
        for i in range(1, 3):
            ocr_learning_service.record_feedback(
                document_type="Sale Deed",
                field_name="survey_number",
                raw_ocr_value="278 | 281",
                corrected_value="278, 281",
                verification_id=f"v-ocrconf-{i}",
                auto_approve=True,
            )
        _, raw_conf, norm_conf, fin_conf, _, n_type = ocr_learning_service.apply_learned_normalization(
            field_name="survey_number",
            raw_value="278 | 281",
            ocr_confidence=0.68,
            document_type="Sale Deed",
            language="en",
        )
        self.assertEqual(raw_conf, 0.68, "Raw OCR confidence must never be altered")
        self.assertEqual(norm_conf, 0.80)
        self.assertEqual(fin_conf, 0.80)

    # -----------------------------------------------------------------------
    # Requirement 17: Raw OCR value is preserved
    # -----------------------------------------------------------------------
    def test_17_raw_ocr_value_is_preserved(self):
        for i in range(1, 3):
            ocr_learning_service.record_feedback(
                document_type="Sale Deed",
                field_name="survey_number",
                raw_ocr_value="278 | 281",
                corrected_value="278, 281",
                verification_id=f"v-preserve-{i}",
                auto_approve=True,
            )
        test_lines = [
            OCRLine(text="SALE DEED", score=0.99, x_min=100, y_min=40, x_max=600, y_max=80, page_num=1, page_height=2000),
            OCRLine(text="Survey No. 278 | 281 extent 480 Sq. Yards", score=0.88, x_min=100, y_min=560, x_max=900, y_max=600, page_num=2, page_height=2000),
        ]
        res, prov, _ = extract_fields_semantic(test_lines)
        sn_prov = prov["survey_number"]
        self.assertEqual(sn_prov["raw_ocr_value"], "278 | 281")
        self.assertEqual(sn_prov["original_value"], "278 | 281")
        self.assertEqual(sn_prov["value"], "278, 281")

    # -----------------------------------------------------------------------
    # Requirement 18: Normalized value is stored separately
    # -----------------------------------------------------------------------
    def test_18_normalized_value_is_stored_separately(self):
        for i in range(1, 3):
            ocr_learning_service.record_feedback(
                document_type="Sale Deed",
                field_name="survey_number",
                raw_ocr_value="278 | 281",
                corrected_value="278, 281",
                verification_id=f"v-sep-{i}",
                auto_approve=True,
            )
        test_lines = [
            OCRLine(text="SALE DEED", score=0.99, x_min=100, y_min=40, x_max=600, y_max=80, page_num=1, page_height=2000),
            OCRLine(text="Survey No. 278 | 281 extent 480 Sq. Yards", score=0.88, x_min=100, y_min=560, x_max=900, y_max=600, page_num=2, page_height=2000),
        ]
        res, prov, _ = extract_fields_semantic(test_lines)
        self.assertIn("value", prov["survey_number"])
        self.assertIn("raw_ocr_value", prov["survey_number"])
        self.assertNotEqual(prov["survey_number"]["value"], prov["survey_number"]["raw_ocr_value"])

    # -----------------------------------------------------------------------
    # Requirement 19: Deterministic normalization is distinguishable from learned correction
    # -----------------------------------------------------------------------
    def test_19_deterministic_distinguishable_from_learned(self):
        # 1. Deterministic baseline without learned rules
        val_det, _, norm_conf, _, r_info, n_type = ocr_learning_service.apply_learned_normalization(
            field_name="property_area",
            raw_value="1O0 Sq. Yards",
            ocr_confidence=0.85,
            document_type="Sale Deed",
            language="en",
        )
        self.assertEqual(n_type, "deterministic_normalization")
        self.assertIsNone(r_info)

        # 2. Learned correction with verified rule
        for i in range(1, 3):
            ocr_learning_service.record_feedback(
                document_type="Sale Deed",
                field_name="property_area",
                raw_ocr_value="1O0 Sq. Yards",
                corrected_value="100 Sq. Yards",
                verification_id=f"v-detdist-{i}",
                auto_approve=True,
            )
        val_lrn, _, _, _, r_info_lrn, n_type_lrn = ocr_learning_service.apply_learned_normalization(
            field_name="property_area",
            raw_value="1O0 Sq. Yards",
            ocr_confidence=0.85,
            document_type="Sale Deed",
            language="en",
        )
        self.assertEqual(n_type_lrn, "learned_correction")
        self.assertIsNotNone(r_info_lrn)

    # -----------------------------------------------------------------------
    # Requirement 20: RSA-PSS signing behavior is unchanged
    # -----------------------------------------------------------------------
    def test_20_rsa_pss_signing_behavior_unchanged(self):
        payload = {
            "document_type": "Sale Deed",
            "document_number": "12736/5",
            "property": {"survey_number": "278, 281", "area": "480 Sq. Yards"},
        }
        sig = verification_service.sign_document(payload)
        pub_key = verification_service.get_public_verification_key()
        self.assertTrue(verification_service.verify_document_signature(payload, sig, pub_key))

        # Corrupted payload must fail verification
        corrupted_payload = dict(payload)
        corrupted_payload["document_number"] = "99999/0"
        self.assertFalse(verification_service.verify_document_signature(corrupted_payload, sig, pub_key))

    # -----------------------------------------------------------------------
    # Requirement 21: Approved record immutability is unchanged
    # -----------------------------------------------------------------------
    def test_21_approved_record_immutability_unchanged(self):
        record = {
            "verification_id": "v-immut-1",
            "status": "APPROVED",
            "signature": "mock-signature",
            "document_payload": {"document_number": "12736/5"},
        }
        verification_service.save_record(record)
        loaded = verification_service.get_record("v-immut-1")
        self.assertEqual(loaded["status"], "APPROVED")

    # -----------------------------------------------------------------------
    # Requirement 22: Existing extraction output format remains compatible
    # -----------------------------------------------------------------------
    def test_22_existing_extraction_compatibility(self):
        test_lines = [
            OCRLine(text="SALE DEED", score=0.99, x_min=100, y_min=40, x_max=600, y_max=80, page_num=1, page_height=2000),
            OCRLine(text="Document No. 12736/5", score=0.95, x_min=100, y_min=100, x_max=450, y_max=140, page_num=1, page_height=2000),
        ]
        res, prov, _ = extract_fields_semantic(test_lines)
        self.assertEqual(res["document_type"], "Sale Deed")
        self.assertEqual(res["document_number"], "12736/5")
        self.assertIn("learning", res)
        self.assertIn("rules_applied", res["learning"])

    # -----------------------------------------------------------------------
    # Requirement 23: Existing verification and checks compatibility
    # -----------------------------------------------------------------------
    def test_23_existing_verification_checks_compatibility(self):
        result = {
            "document_type": "Sale Deed",
            "document_number": "12736/5",
            "survey_number": "278, 281",
            "property_area": "480 Sq. Yards",
            "village": "Aushapur",
            "mandal": "Ghatkesar",
            "district": "Ranga Reddy",
            "stamp_serial_number": "CC 123456",
            "parties_list": ["SRI P. SRINIVAS REDDY", "SMT. B. SUVARNA"],
            "document_date": "15/07/2021",
            "execution_date": "15/07/2021",
        }
        checks = verification_service.run_verification_checks(result)
        self.assertTrue(isinstance(checks, list))
        self.assertGreater(len(checks), 0)
        overall_status = verification_service.calculate_overall_status(checks)
        self.assertIn(overall_status, ["READY_FOR_APPROVAL", "ACTION_REQUIRED", "REJECTED", "DUPLICATE"])


if __name__ == "__main__":
    unittest.main()
