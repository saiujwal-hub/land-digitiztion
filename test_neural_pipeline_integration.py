"""
test_neural_pipeline_integration.py - Full Pipeline Integration Tests for Qwen Neural NLP Layer.

Verifies:
1. Realistic land-deed OCR text (English + Telugu mixed) reaches Neural NLP layer.
2. Qwen JSON response correctly parsed into 12 canonical fields.
3. Existing deterministic semantic extraction remains primary/authoritative.
4. Qwen output NEVER overwrites semantic extraction, verification records, or seals.
5. Discrepancies between semantic and neural extractions are detected and flagged as advisory conflicts.
6. Unsupported/missing fields resolve to None/null (evidence-grounded, zero hallucination).
7. Pipeline continues operating normally when Qwen/GPU is unavailable.
8. Verification record accurately stores advisory comparison without mutating core payload.
"""

import json
import unittest
from unittest.mock import patch, MagicMock

import neural_nlp_service
import verification_service


# Realistic land-deed sample with bilingual English + Telugu text
BILINGUAL_LAND_DEED_OCR = """
GOVERNMENT OF TELANGANA - REGISTRATION AND STAMPS DEPARTMENT
SALE DEED (విక్రయ పత్రము)
Document No: 4892/2024
Date of Registration: 18/09/2024

విక్రేత (Vendor / Executant):
Sri K. Venkata Ramana, S/o Narayana, R/o Hyderabad

కొనుగోలుదారు (Purchaser / Claimant):
Smt. P. Lakshmi Devi, W/o Srinivasa Rao, R/o Secunderabad

SCHEDULE OF PROPERTY:
All that piece and parcel of Agricultural / Residential Land situated at:
Survey No: 342
Sub-Division / Sub Survey No: 342/B
Extent of Area: 3.50 Acres
Village: Kothapalli (కొత్తపల్లి)
Mandal: Ghatkesar (ఘట్‌కేసర్)
District: Medchal-Malkajgiri (మేడ్చల్-మల్కాజ్‌గిరి)
State: Telangana

CONSIDERATION:
Total sale consideration amount paid: Rs. 45,00,000/- (Rupees Forty Five Lakhs only).
Market Value: Rs. 42,00,000/-
Stamp Duty Paid: Rs. 2,25,000/-
"""


class TestNeuralPipelineIntegration(unittest.TestCase):
    """Integration test suite for Qwen Neural NLP hybrid pipeline."""

    def test_01_canonical_schema_and_advisory_invariants(self):
        """Verify 12 canonical fields and strict advisory contracts."""
        self.assertEqual(len(neural_nlp_service.CANONICAL_SCHEMA_KEYS), 12)
        expected_fields = [
            "document_type", "document_number", "document_date",
            "vendor", "purchaser", "survey_number",
            "sub_survey_number", "property_area", "village",
            "mandal", "district", "consideration_amount"
        ]
        self.assertEqual(neural_nlp_service.CANONICAL_SCHEMA_KEYS, expected_fields)

    def test_02_bilingual_ocr_text_to_neural_layer_and_json_parsing(self):
        """Verify bilingual English + Telugu OCR text is processed and parsed into 12 canonical fields."""
        mock_model_output = json.dumps({
            "document_type": "Sale Deed",
            "document_number": "4892/2024",
            "document_date": "18/09/2024",
            "vendor": "K. Venkata Ramana",
            "purchaser": "P. Lakshmi Devi",
            "survey_number": "342",
            "sub_survey_number": "342/B",
            "property_area": "3.50 Acres",
            "village": "Kothapalli",
            "mandal": "Ghatkesar",
            "district": "Medchal-Malkajgiri",
            "consideration_amount": "Rs. 45,00,000/-",
        })

        parsed = neural_nlp_service.parse_model_json_response(mock_model_output)
        self.assertEqual(parsed["document_type"], "Sale Deed")
        self.assertEqual(parsed["document_number"], "4892/2024")
        self.assertEqual(parsed["survey_number"], "342")
        self.assertEqual(parsed["sub_survey_number"], "342/B")
        self.assertEqual(parsed["village"], "Kothapalli")
        self.assertEqual(parsed["mandal"], "Ghatkesar")
        self.assertEqual(parsed["district"], "Medchal-Malkajgiri")
        self.assertEqual(parsed["consideration_amount"], "Rs. 45,00,000/-")

        # Verify evidence grounding finds matching lines from bilingual OCR text
        vendor_evidence = neural_nlp_service.find_field_evidence(parsed["vendor"], BILINGUAL_LAND_DEED_OCR)
        self.assertIsNotNone(vendor_evidence)
        self.assertIn("Venkata Ramana", vendor_evidence)

    def test_03_authoritative_semantic_preservation_and_no_overwrite(self):
        """Verify semantic extraction remains untouched when advisory neural NLP runs."""
        authoritative_semantic = {
            "document_type": "Sale Deed",
            "document_number": "4892/2024",
            "document_date": "18/09/2024",
            "survey_number": "342",
            "sub_survey_number": "342/B",
            "property_area": "3.50 Acres",
            "village": "Kothapalli",
            "mandal": "Ghatkesar",
            "district": "Medchal-Malkajgiri",
            "parties_list": [
                {"role": "vendor", "name": "K. Venkata Ramana"},
                {"role": "purchaser", "name": "P. Lakshmi Devi"},
            ],
            "stamp_information": {
                "stamp_value": "45,00,000",
            },
        }

        # Simulated Qwen response
        qwen_extraction = {
            "document_type": "Sale Deed",
            "document_number": "4892/2024",
            "document_date": "18/09/2024",
            "vendor": "K. Venkata Ramana",
            "purchaser": "P. Lakshmi Devi",
            "survey_number": "342",
            "sub_survey_number": "342/B",
            "property_area": "3.50 Acres",
            "village": "Kothapalli",
            "mandal": "Ghatkesar",
            "district": "Medchal-Malkajgiri",
            "consideration_amount": "45,00,000",
        }

        with patch("neural_nlp_service.extract_legal_fields_with_qwen") as mock_extract:
            mock_extract.return_value = {
                "success": True,
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "backend": "remote_gpu",
                "gpu": True,
                "result": qwen_extraction,
                "evidence": {"survey_number": "Survey No: 342"},
            }

            advisory = neural_nlp_service.run_neural_nlp_advisory(
                raw_ocr_text=BILINGUAL_LAND_DEED_OCR,
                semantic_result=authoritative_semantic,
                language="te",
            )

        # 1. Advisory status must be AVAILABLE
        self.assertEqual(advisory["status"], "AVAILABLE")
        self.assertTrue(advisory["is_advisory"])
        # 2. No conflicts on matching data
        self.assertEqual(advisory["conflict_count"], 0)
        self.assertFalse(advisory["has_conflicts"])
        # 3. Authoritative dictionary must NOT be mutated
        self.assertEqual(authoritative_semantic["document_number"], "4892/2024")
        self.assertEqual(authoritative_semantic["survey_number"], "342")

    def test_04_conflict_detection_and_flagging(self):
        """Verify field-level discrepancies between semantic and neural extractions are flagged."""
        authoritative_semantic = {
            "document_type": "Sale Deed",
            "document_number": "4892/2024",
            "survey_number": "342",  # Authoritative survey number
            "village": "Kothapalli",
            "mandal": "Ghatkesar",
            "district": "Medchal-Malkajgiri",
            "property_area": "3.50 Acres",
        }

        # Suppose neural model predicted a conflicting survey number and district
        conflicting_qwen = {
            "document_type": "Sale Deed",
            "document_number": "4892/2024",
            "survey_number": "999-CONFLICT",  # Conflict!
            "village": "Kothapalli",
            "mandal": "Ghatkesar",
            "district": "Rangareddy",          # Conflict!
            "property_area": "3.50 Acres",
            "vendor": None,
            "purchaser": None,
            "sub_survey_number": None,
            "document_date": None,
            "consideration_amount": None,
        }

        with patch("neural_nlp_service.extract_legal_fields_with_qwen") as mock_extract:
            mock_extract.return_value = {
                "success": True,
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "backend": "remote_gpu",
                "gpu": True,
                "result": conflicting_qwen,
                "evidence": {},
            }

            advisory = neural_nlp_service.run_neural_nlp_advisory(
                raw_ocr_text=BILINGUAL_LAND_DEED_OCR,
                semantic_result=authoritative_semantic,
            )

        self.assertTrue(advisory["has_conflicts"])
        self.assertGreaterEqual(advisory["conflict_count"], 2)
        conflict_fields = [c["field"] for c in advisory["conflicts"]]
        self.assertIn("survey_number", conflict_fields)
        self.assertIn("district", conflict_fields)

        # Primary source of truth remains unchanged
        self.assertEqual(authoritative_semantic["survey_number"], "342")
        self.assertEqual(authoritative_semantic["district"], "Medchal-Malkajgiri")

    def test_05_missing_values_become_null_without_hallucination(self):
        """Verify fields not present in OCR text are returned as None (null) and not invented."""
        sparse_ocr = "SALE DEED. Document Number 555. Survey No: 10."
        raw_json_with_nulls = json.dumps({
            "document_type": "Sale Deed",
            "document_number": "555",
            "survey_number": "10",
            "vendor": "null",
            "purchaser": "None",
            "sub_survey_number": "",
            "property_area": "not specified",
            "village": None,
            "mandal": None,
            "district": None,
            "consideration_amount": None,
            "document_date": None,
        })

        parsed = neural_nlp_service.parse_model_json_response(raw_json_with_nulls)
        self.assertEqual(parsed["document_type"], "Sale Deed")
        self.assertEqual(parsed["document_number"], "555")
        self.assertEqual(parsed["survey_number"], "10")
        self.assertIsNone(parsed["vendor"])
        self.assertIsNone(parsed["purchaser"])
        self.assertIsNone(parsed["property_area"])
        self.assertIsNone(parsed["village"])
        self.assertIsNone(parsed["mandal"])
        self.assertIsNone(parsed["district"])
        self.assertIsNone(parsed["consideration_amount"])

    def test_06_graceful_fallback_when_gpu_or_remote_unavailable(self):
        """Verify pipeline does not crash when Qwen is unavailable."""
        semantic_data = {"document_type": "Gift Deed", "document_number": "101"}

        with patch("neural_nlp_service.extract_legal_fields_with_qwen") as mock_extract:
            mock_extract.return_value = {
                "success": False,
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "backend": "unavailable",
                "gpu": False,
                "error": "Connection timed out to Kaggle GPU worker",
                "result": neural_nlp_service.get_default_empty_schema(),
                "evidence": {},
            }

            advisory = neural_nlp_service.run_neural_nlp_advisory(
                raw_ocr_text="Some deed text",
                semantic_result=semantic_data,
            )

        self.assertEqual(advisory["status"], "UNAVAILABLE")
        self.assertFalse(advisory["has_conflicts"])
        self.assertEqual(advisory["conflict_count"], 0)
        self.assertTrue(advisory["is_advisory"])
        self.assertIn("Connection timed out", advisory["error"])

    def test_07_verification_record_stores_advisory_without_mutating_core_state(self):
        """Verify verification_service.create_verification_record preserves neural_nlp without side effects."""
        mock_result = {
            "document_type": "Sale Deed",
            "document_number": "4892/2024",
            "document_date": "18/09/2024",
            "parties": [{"role": "vendor", "name": "K. Venkata Ramana"}],
            "property": {
                "survey_number": "342",
                "area": "3.50 Acres",
                "village": "Kothapalli",
                "district": "Medchal-Malkajgiri",
            },
            "neural_nlp": {
                "status": "AVAILABLE",
                "is_advisory": True,
                "has_conflicts": False,
                "conflict_count": 0,
            },
        }

        record = verification_service.create_verification_record(mock_result, file_hash="dummy_hash")
        self.assertIn("neural_nlp", record)
        self.assertEqual(record["neural_nlp"]["status"], "AVAILABLE")
        # Ensure status calculation, checks, and payload are unaltered
        self.assertIn("checks", record)
        self.assertIn("document_payload", record)
        self.assertEqual(record["document_payload"]["document_number"], "4892/2024")

    def test_08_complete_12_field_evaluation_categories(self):
        """Verify all 12 canonical fields are evaluated and classified into the 5 standard categories."""
        semantic_data = {
            "document_type": "Sale Deed",      # Match -> AGREEMENT
            "document_number": "1001",         # Match -> AGREEMENT
            "village": "OldVillage",           # Differs -> DISCREPANCY_ADVISORY
            "mandal": "Ghatkesar",             # Qwen missing -> NEURAL_MISSING
            # vendor missing in semantic, present in Qwen -> SEMANTIC_MISSING
            # consideration_amount missing in both -> BOTH_MISSING
        }

        qwen_data = {
            "document_type": "Sale Deed",
            "document_number": "1001",
            "village": "NewVillage",
            "mandal": None,
            "vendor": "Sri Ramesh",
            "purchaser": None,
            "survey_number": None,
            "sub_survey_number": None,
            "property_area": None,
            "district": None,
            "document_date": None,
            "consideration_amount": None,
        }

        comparison = neural_nlp_service.compare_semantic_and_neural(semantic_data, qwen_data)

        self.assertEqual(comparison["total_fields_evaluated"], 12)
        self.assertEqual(len(comparison["field_evaluations"]), 12)

        statuses = {ev["field"]: ev["status"] for ev in comparison["field_evaluations"]}

        self.assertEqual(statuses["document_type"], "AGREEMENT")
        self.assertEqual(statuses["document_number"], "AGREEMENT")
        self.assertEqual(statuses["village"], "DISCREPANCY_ADVISORY")
        self.assertEqual(statuses["mandal"], "NEURAL_MISSING")
        self.assertEqual(statuses["vendor"], "SEMANTIC_MISSING")
        self.assertEqual(statuses["consideration_amount"], "BOTH_MISSING")

        # Crucial requirement: SEMANTIC_MISSING is NOT classified as a conflict
        self.assertEqual(comparison["conflict_count"], 1)
        self.assertEqual(len(comparison["conflicts"]), 1)
        self.assertEqual(comparison["conflicts"][0]["field"], "village")


if __name__ == "__main__":
    unittest.main()
