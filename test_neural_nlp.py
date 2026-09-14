"""
test_neural_nlp.py - Isolated Unit Tests for Neural NLP (Qwen2.5-7B-Instruct) Layer.

Tests:
1. Model configuration and schema definitions.
2. Missing fields return null (None) without hallucination.
3. Strict evidence-based prompt construction.
4. Advisory-only comparison matrix (read-only, no side effects).
5. Graceful fallback when local GPU or remote Kaggle service is unavailable.
6. Remote endpoint response handling.
7. Zero mutation of semantic extraction and verification state.
"""

import copy
import json
import unittest
from unittest.mock import patch, MagicMock

import neural_nlp_service


class TestNeuralNLPService(unittest.TestCase):

    def test_model_id_and_canonical_keys(self):
        """Verify model identifier and schema keys match specifications."""
        self.assertEqual(neural_nlp_service.MODEL_ID, "Qwen/Qwen2.5-7B-Instruct")
        expected_keys = [
            "document_type",
            "document_number",
            "document_date",
            "vendor",
            "purchaser",
            "survey_number",
            "sub_survey_number",
            "property_area",
            "village",
            "mandal",
            "district",
            "consideration_amount",
        ]
        self.assertEqual(neural_nlp_service.CANONICAL_SCHEMA_KEYS, expected_keys)

        schema = neural_nlp_service.get_default_empty_schema()
        self.assertEqual(len(schema), 12)
        for key in expected_keys:
            self.assertIn(key, schema)
            self.assertIsNone(schema[key])

    def test_parse_model_json_response_strict_schema(self):
        """Verify parsing converts missing, empty, or placeholder values to null (None)."""
        raw_llm_output = """```json
        {
            "document_type": "Sale Deed",
            "survey_number": "1413",
            "village": "Mangapet",
            "district": "Mulugu",
            "document_number": "None",
            "vendor": "",
            "purchaser": "   ",
            "property_area": null,
            "consideration_amount": "Rs. 5,00,000/-"
        }
        ```"""

        parsed = neural_nlp_service.parse_model_json_response(raw_llm_output)

        # Non-empty extracted fields
        self.assertEqual(parsed["document_type"], "Sale Deed")
        self.assertEqual(parsed["survey_number"], "1413")
        self.assertEqual(parsed["village"], "Mangapet")
        self.assertEqual(parsed["district"], "Mulugu")
        self.assertEqual(parsed["consideration_amount"], "Rs. 5,00,000/-")

        # Missing or empty fields must be strictly None (null)
        self.assertIsNone(parsed["document_number"])
        self.assertIsNone(parsed["vendor"])
        self.assertIsNone(parsed["purchaser"])
        self.assertIsNone(parsed["sub_survey_number"])
        self.assertIsNone(parsed["property_area"])
        self.assertIsNone(parsed["mandal"])
        self.assertIsNone(parsed["document_date"])

    def test_parse_invalid_or_malformed_json_fallback(self):
        """Ensure malformed or non-JSON model output does not crash and returns null schema."""
        malformed_output = "I cannot extract any information from the image because the text is blurry."
        parsed = neural_nlp_service.parse_model_json_response(malformed_output)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(len(parsed), 12)
        for val in parsed.values():
            self.assertIsNone(val)

    def test_evidence_prompt_constraints(self):
        """Verify prompt contains strict anti-hallucination instructions and language context."""
        ocr_text = "THIS SALE DEED IS EXECUTED AT WARANGAL ON 03-08-2019"
        prompt = neural_nlp_service.build_evidence_prompt(ocr_text, language="te")

        self.assertIn("Telugu", prompt)
        self.assertIn("NEVER infer, hallucinate, assume, or invent", prompt)
        self.assertIn("return null for that field", prompt)
        self.assertIn(ocr_text, prompt)

    def test_find_field_evidence(self):
        """Verify textual provenance finder locates supporting lines without hallucinating boxes."""
        ocr_text = "FIRST PAGE OF SALE DEED\nSY.NO. 1413 SITUATED IN MANGAPET\nCONSIDERATION RS. 500000"
        evidence = neural_nlp_service.find_field_evidence("1413", ocr_text)
        self.assertIsNotNone(evidence)
        self.assertIn("1413", evidence)

        missing_evidence = neural_nlp_service.find_field_evidence("NonExistentField", ocr_text)
        self.assertIsNone(missing_evidence)

    def test_compare_semantic_and_neural_advisory_only(self):
        """Verify comparison layer is purely advisory, read-only, and reports agreements/conflicts."""
        semantic_data = {
            "document_type": "Sale Deed",
            "document_number": "379230",
            "survey_number": "1413",
            "village": "Mangapet",
            "district": "Mulugu",
            "mandal": "Mangapet",
        }
        neural_data = {
            "document_type": "Sale Deed",          # Agreement
            "document_number": "379230/2019",      # Discrepancy/Conflict
            "survey_number": "1413",              # Agreement
            "village": "Mangapet",                # Agreement
            "district": "Warangal",               # Conflict
            "mandal": None,                       # Unmatched
        }

        sem_copy = copy.deepcopy(semantic_data)
        neu_copy = copy.deepcopy(neural_data)

        comparison = neural_nlp_service.compare_semantic_and_neural(semantic_data, neural_data)

        # Verify read-only: input dictionaries must be completely unmutated
        self.assertEqual(semantic_data, sem_copy)
        self.assertEqual(neural_data, neu_copy)

        # Verify advisory contract
        self.assertEqual(comparison["status"], "ADVISORY_ONLY")
        self.assertEqual(comparison["primary_source_of_truth"], "semantic_extractor.py (Rule-Based)")
        self.assertGreater(comparison["agreement_count"], 0)
        self.assertGreater(comparison["conflict_count"], 0)

        # Verify agreement detected for survey_number and village
        agree_fields = [a["field"] for a in comparison["agreements"]]
        self.assertIn("survey_number", agree_fields)
        self.assertIn("village", agree_fields)

        # Verify conflict detected for district
        conflict_fields = [c["field"] for c in comparison["conflicts"]]
        self.assertIn("district", conflict_fields)

    def test_graceful_fallback_when_local_gpu_unavailable(self):
        """Ensure execution on machine without CUDA falls back gracefully without throwing exceptions."""
        result = neural_nlp_service.extract_legal_fields_with_qwen(
            ocr_text="TEST DEED TEXT",
            language="en",
            remote_url=None,
        )
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("result", result)
        self.assertEqual(len(result["result"]), 12)
        # Should not crash even when CUDA is not present
        if not result["success"]:
            self.assertIn("error", result)

    def test_empty_ocr_text_handling(self):
        """Ensure empty OCR text returns success=True with empty schema and warning."""
        result = neural_nlp_service.extract_legal_fields_with_qwen("", language="en")
        self.assertTrue(result["success"])
        self.assertIn("warning", result)
        self.assertEqual(result["result"], neural_nlp_service.get_default_empty_schema())

    def test_remote_kaggle_gpu_endpoint_routing(self):
        """Verify remote URL routing sends POST request to /nlp and deserializes result."""
        mock_remote_response = {
            "success": True,
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "gpu": True,
            "result": {
                "document_type": "Sale Deed",
                "document_number": "18452/2019",
                "document_date": "03-08-2019",
                "vendor": "Guduru Malakondaiah",
                "purchaser": "Rangineni Venkateshwar Rao",
                "survey_number": "1413",
                "sub_survey_number": None,
                "property_area": "200 Sq.Yards",
                "village": "Mangapet",
                "mandal": "Mangapet",
                "district": "Mulugu",
                "consideration_amount": "Rs. 5,00,000/-"
            },
            "evidence": {
                "survey_number": "SY.NO. 1413",
                "village": "MANGAPET VILLAGE"
            }
        }

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_remote_response

            res = neural_nlp_service.extract_legal_fields_with_qwen(
                ocr_text="SAMPLE DEED TEXT",
                remote_url="https://mock-kaggle-tunnel.trycloudflare.com",
            )

            mock_post.assert_called_once()
            endpoint_called = mock_post.call_args[0][0]
            self.assertTrue(endpoint_called.endswith("/nlp"))
            self.assertTrue(res["success"])
            self.assertEqual(res["result"]["survey_number"], "1413")
            self.assertEqual(res["result"]["village"], "Mangapet")


if __name__ == "__main__":
    unittest.main()
