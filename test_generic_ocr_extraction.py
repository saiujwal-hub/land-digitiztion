"""
test_generic_ocr_extraction.py
Comprehensive regression tests verifying generic OCR extraction without sample-specific bias.

Verifies:
1. Document A vs Document B isolation (zero cross-contamination).
2. Missing-field document returns value=None, status="NOT_FOUND", confidence=0.0, needs_review=True, evidence=[].
3. Ambiguous/conflicting fields return status="CONFLICT", needs_review=True, penalized confidence.
4. Sample-specific value scanner confirms zero occurrences of sample values in generic outputs.
5. Script-based language detection uses Unicode ranges, not English keyword matching.
6. Handwriting structure strictly separates detection from recognition.
"""

import unittest
from land_document_extractor import OCRLine, detect_languages, extract_land_document_from_lines
from semantic_extractor import extract_fields_semantic
from run_telangana_ocr import locate_field_provenance, locate_party_provenance


class TestGenericOCRExtraction(unittest.TestCase):

    def setUp(self):
        self.sample_forbidden_values = [
            "278", "281", "282", "1023/1", "1023/2", "480", "12736",
            "Aushapur", "Srinidhi", "Suvarna", "P. Srinivas Reddy", "Srinidhi Enclave"
        ]

    def _assert_no_forbidden_values(self, obj, path=""):
        """Recursively scan an object for forbidden sample-specific strings."""
        if isinstance(obj, str):
            for forbidden in self.sample_forbidden_values:
                self.assertNotIn(
                    forbidden.lower(),
                    obj.lower(),
                    f"Forbidden sample value '{forbidden}' found in path '{path}': {obj}"
                )
        elif isinstance(obj, dict):
            for k, v in obj.items():
                self._assert_no_forbidden_values(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                self._assert_no_forbidden_values(item, f"{path}[{idx}]")

    def test_document_a_vs_document_b_isolation(self):
        """Test two completely distinct documents to verify zero cross-contamination."""
        # Document A
        doc_a_lines = [
            OCRLine(text="SALE DEED", score=0.98, x_min=300, y_min=100, x_max=600, y_max=130, page_num=1, page_height=2000),
            OCRLine(text="Document No. 5678 of 2021", score=0.95, x_min=100, y_min=200, x_max=500, y_max=230, page_num=1, page_height=2000),
            OCRLine(text="SRI RAJESH KUMAR S/O SRI MOHAN KUMAR", score=0.92, x_min=100, y_min=400, x_max=700, y_max=430, page_num=1, page_height=2000),
            OCRLine(text="HEREINAFTER CALLED THE VENDOR", score=0.90, x_min=100, y_min=450, x_max=600, y_max=480, page_num=1, page_height=2000),
            OCRLine(text="IN FAVOUR OF", score=0.88, x_min=100, y_min=500, x_max=300, y_max=520, page_num=1, page_height=2000),
            OCRLine(text="SMT. PRIYA SHARMA W/O SRI ANKIT SHARMA", score=0.94, x_min=100, y_min=540, x_max=750, y_max=570, page_num=1, page_height=2000),
            OCRLine(text="HEREINAFTER CALLED THE PURCHASER", score=0.89, x_min=100, y_min=590, x_max=650, y_max=620, page_num=1, page_height=2000),
            OCRLine(text="Plot No. 45 in Survey No. 412/415 measuring an extent of 350 Sq. Yards", score=0.93, x_min=100, y_min=700, x_max=900, y_max=740, page_num=2, page_height=2000),
            OCRLine(text="Situated at Narsingi Village, Gandipet Mandal, Ranga Reddy District, Telangana", score=0.91, x_min=100, y_min=760, x_max=950, y_max=800, page_num=2, page_height=2000),
        ]

        # Document B
        doc_b_lines = [
            OCRLine(text="GIFT SETTLEMENT DEED", score=0.97, x_min=300, y_min=100, x_max=600, y_max=130, page_num=1, page_height=2000),
            OCRLine(text="Document No. 9912 of 2018", score=0.93, x_min=100, y_min=200, x_max=500, y_max=230, page_num=1, page_height=2000),
            OCRLine(text="SRI VENKAT RAMAN S/O SRI KRISHNA MURTHY", score=0.91, x_min=100, y_min=400, x_max=700, y_max=430, page_num=1, page_height=2000),
            OCRLine(text="HEREINAFTER CALLED THE DONOR", score=0.89, x_min=100, y_min=450, x_max=600, y_max=480, page_num=1, page_height=2000),
            OCRLine(text="IN FAVOUR OF", score=0.87, x_min=100, y_min=500, x_max=300, y_max=520, page_num=1, page_height=2000),
            OCRLine(text="SRI ADITYA RAMAN S/O SRI VENKAT RAMAN", score=0.95, x_min=100, y_min=540, x_max=750, y_max=570, page_num=1, page_height=2000),
            OCRLine(text="HEREINAFTER CALLED THE PURCHASER", score=0.88, x_min=100, y_min=590, x_max=650, y_max=620, page_num=1, page_height=2000),
            OCRLine(text="Plot No. 12 in Survey No. 789/790 measuring an extent of 600 Sq. Yards", score=0.94, x_min=100, y_min=700, x_max=900, y_max=740, page_num=2, page_height=2000),
            OCRLine(text="Situated at Miyapur Village, Serilingampally Mandal, Medchal District, Telangana", score=0.92, x_min=100, y_min=760, x_max=950, y_max=800, page_num=2, page_height=2000),
        ]

        res_a, prov_a, _ = extract_fields_semantic(doc_a_lines)
        res_b, prov_b, _ = extract_fields_semantic(doc_b_lines)

        # Assert Doc A extraction
        self.assertIn("412/415", res_a["survey_number"])
        self.assertEqual(res_a["property_area"], 350)
        self.assertEqual(res_a["village"], "Narsingi")
        self.assertEqual(res_a["mandal"], "Gandipet")

        # Assert Doc B extraction
        self.assertIn("789/790", res_b["survey_number"])
        self.assertEqual(res_b["property_area"], 600)
        self.assertEqual(res_b["village"], "Miyapur")
        self.assertEqual(res_b["mandal"], "Serilingampally")

        # Assert ZERO cross-contamination
        self.assertNotIn("412/415", str(res_b))
        self.assertNotIn("350", str(res_b))
        self.assertNotIn("Narsingi", str(res_b))
        self.assertNotIn("Gandipet", str(res_b))
        self.assertNotIn("Rajesh", str(res_b))

        self.assertNotIn("789/790", str(res_a))
        self.assertNotIn("600", str(res_a))
        self.assertNotIn("Miyapur", str(res_a))
        self.assertNotIn("Serilingampally", str(res_a))
        self.assertNotIn("Venkat", str(res_a))

        # Scan both against forbidden sample values
        self._assert_no_forbidden_values(res_a)
        self._assert_no_forbidden_values(res_b)
        self._assert_no_forbidden_values(prov_a)
        self._assert_no_forbidden_values(prov_b)

    def test_missing_fields_return_not_found(self):
        """A minimal document lacking fields must return value=None, status='NOT_FOUND', confidence=0.0."""
        sparse_lines = [
            OCRLine(text="MEMORANDUM OF UNDERSTANDING", score=0.95, x_min=200, y_min=100, x_max=600, y_max=130, page_num=1, page_height=2000),
            OCRLine(text="This agreement is entered into this day.", score=0.88, x_min=100, y_min=200, x_max=600, y_max=220, page_num=1, page_height=2000),
        ]

        result, provenance, _ = extract_fields_semantic(sparse_lines)

        # In semantic result dict
        self.assertIsNone(result["survey_number"])
        self.assertIsNone(result["property_area"])
        self.assertIsNone(result["village"])
        self.assertIsNone(result["mandal"])
        self.assertIsNone(result["district"])

        # In semantic provenance dict
        for fld in ["survey_number", "property_area", "village", "mandal", "district"]:
            entry = provenance[fld]
            self.assertIsNone(entry["value"])
            self.assertEqual(entry["status"], "NOT_FOUND")
            self.assertEqual(entry["confidence"], 0.0)
            self.assertTrue(entry["needs_review"])
            self.assertEqual(entry["evidence"], [])

        # In runner provenance schema (locate_field_provenance)
        runner_prov = locate_field_provenance(result["survey_number"], provenance.get("survey_number", {}), "survey_numbers", all_lines=sparse_lines)
        self.assertEqual(runner_prov["field_name"], "survey_numbers")
        self.assertIsNone(runner_prov["value"])
        self.assertEqual(runner_prov["status"], "NOT_FOUND")
        self.assertEqual(runner_prov["confidence"], 0.0)
        self.assertTrue(runner_prov["needs_review"])
        self.assertEqual(runner_prov["evidence"], [])
        self.assertIsNone(runner_prov["source_text"])
        self.assertIsNone(runner_prov["page_number"])

        # Scan for forbidden values
        self._assert_no_forbidden_values(result)
        self._assert_no_forbidden_values(provenance)
        self._assert_no_forbidden_values(runner_prov)

    def test_ambiguous_conflicting_fields(self):
        """Conflicting candidates should trigger status='CONFLICT', needs_review=True, and lowered confidence."""
        conflicting_lines = [
            OCRLine(text="Property measuring an extent of 200 Sq. Yards in first clause", score=0.90, x_min=100, y_min=300, x_max=800, y_max=330, page_num=1, page_height=2000),
            OCRLine(text="WHEREAS the Schedule mentions 500 Sq. Yards instead", score=0.88, x_min=100, y_min=400, x_max=800, y_max=430, page_num=2, page_height=2000),
        ]

        result, provenance, _ = extract_fields_semantic(conflicting_lines)
        area_info = provenance["property_area"]

        # Multiple conflicting areas were found
        self.assertEqual(area_info["status"], "CONFLICT")
        self.assertTrue(area_info["needs_review"])
        self.assertLess(area_info["confidence"], 0.70)
        self.assertGreater(len(area_info["evidence"]), 1)

    def test_script_based_language_detection(self):
        """Ensure language detection uses Unicode scripts and does not report Telugu for English text."""
        # Pure English with "Telangana" keyword
        english_with_telangana_keyword = "Government of Telangana Registration and Stamps Department Sale Deed Hyderabad"
        langs = detect_languages(english_with_telangana_keyword)
        self.assertIn("English", langs)
        self.assertNotIn("Telugu", langs)

        # Actual Telugu script (Unicode \u0C00-\u0C7F)
        telugu_text = "తెలంగాణ ప్రభుత్వం రిజిస్ట్రేషన్ శాఖ"
        langs_te = detect_languages(telugu_text)
        self.assertIn("Telugu", langs_te)

        # Actual Hindi / Devanagari script (\u0900-\u097F)
        hindi_text = "विक्रय विलेख भारत सरकार"
        langs_hi = detect_languages(hindi_text)
        self.assertIn("Hindi", langs_hi)

        # Garbage / punctuation only
        garbage_text = "--- === 12345 67890 !!! ??? ..."
        langs_unknown = detect_languages(garbage_text)
        self.assertEqual(langs_unknown, ["unknown"])

    def test_separated_handwriting_structure(self):
        """Handwriting structure must separate detection from recognition."""
        doc = extract_land_document_from_lines([], raw_text="Some plain printed text", image_path="")
        features = doc["document_features"]
        self.assertIn("handwriting", features)
        hw = features["handwriting"]
        self.assertIn("detected", hw)
        self.assertIn("recognized", hw)
        self.assertIn("needs_review", hw)
        self.assertIsInstance(hw["detected"], bool)
        self.assertFalse(hw["recognized"])  # strictly False until handwriting engine is implemented
        self.assertIsInstance(hw["needs_review"], bool)


if __name__ == "__main__":
    unittest.main()
