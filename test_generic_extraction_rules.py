"""
Comprehensive test suite verifying generic extraction rules:
1. No hardcoded owner/vendor/purchaser names (e.g. K. Rajesh Kumar, Ananya Sharma are never replaced with Srinidhi or Suvarna).
2. No hardcoded survey numbers (e.g. 504/A, 505/B are never replaced with 278, 281, 282).
3. No hardcoded area coercion (e.g. 325 sq. yards is preserved, 488 sq. yards is NOT coerced to 480).
4. Missing values return None with status: "NOT_FOUND".
5. Conflicting candidates return status: "CONFLICT" with needs_review: True and candidate list preserved.
"""

import unittest
from semantic_extractor import (
    extract_fields_semantic,
    clean_user_facing_schema,
    aggregate_survey_numbers,
    select_best,
    FieldCandidate,
)


class MockLine:
    def __init__(self, text, page_num=1, y_rel=0.5, score=0.95):
        self.text = text
        self.page_num = page_num
        self.y_rel = y_rel
        self.score = score


class TestGenericExtractionRules(unittest.TestCase):

    def test_different_parties_never_replaced(self):
        """Verify arbitrary parties are preserved and never fallback to Srinidhi Homes or Suvarna."""
        lines = [
            MockLine("THIS DEED OF SALE is made and executed by", page_num=1),
            MockLine("K. RAJESH KUMAR, S/O. K. VENKATESH", page_num=1),
            MockLine("(Hereinafter called the 'VENDOR') of the First Part.", page_num=1),
            MockLine("IN FAVOUR OF", page_num=1),
            MockLine("SMT. ANANYA SHARMA W/O. DR. VIKRAM SHARMA", page_num=1),
            MockLine("(Hereinafter called the 'PURCHASER') of the Second Part.", page_num=1),
        ]
        result, provenance, _ = extract_fields_semantic(lines)
        parties = result["parties_list"]
        self.assertEqual(len(parties), 2)
        
        vendor = next(p for p in parties if p["role"] == "Vendor")
        purchaser = next(p for p in parties if p["role"] == "Purchaser")

        self.assertIn("Rajesh Kumar", vendor["name"])
        self.assertNotIn("Srinidhi", vendor["name"])
        
        self.assertIn("Ananya Sharma", purchaser["name"])
        self.assertNotIn("Suvarna", purchaser["name"])

    def test_different_survey_numbers_never_replaced(self):
        """Verify arbitrary survey numbers (e.g. 504/A, 505/B) are never coerced to 278, 281, 282."""
        lines = [
            MockLine("SCHEDULE OF THE PROPERTY", page_num=5),
            MockLine("Land situated in Survey Nos. 504/A and 505/B of Kondapur Village", page_num=5),
        ]
        result, provenance, _ = extract_fields_semantic(lines)
        sn = result["survey_number"]
        self.assertIn("504/A", sn)
        self.assertIn("505/B", sn)
        self.assertNotIn("278", sn)
        self.assertNotIn("281", sn)
        self.assertNotIn("282", sn)

    def test_different_area_not_coerced(self):
        """Verify arbitrary area (325 sq. yards) is preserved, and 488 is NOT coerced to 480."""
        # 1. Arbitrary area 325 sq. yards
        lines_325 = [
            MockLine("SCHEDULE OF THE PROPERTY", page_num=5),
            MockLine("admeasuring an extent of 325 Sq.Yards or 271.7 Sq.Mtrs.", page_num=5),
        ]
        res_325, _, _ = extract_fields_semantic(lines_325)
        clean_325 = clean_user_facing_schema(res_325)
        self.assertEqual(clean_325["property_area"], 325)

        # 2. Area 488 sq. yards must stay 488, NOT coerced to 480
        lines_488 = [
            MockLine("SCHEDULE OF THE PROPERTY", page_num=5),
            MockLine("admeasuring an extent of 488 Sq.Yards or 408 Sq.Mtrs.", page_num=5),
        ]
        res_488, _, _ = extract_fields_semantic(lines_488)
        clean_488 = clean_user_facing_schema(res_488)
        self.assertEqual(clean_488["property_area"], 488)

    def test_missing_values_return_not_found(self):
        """Verify that when a field cannot be found, value is None and status is NOT_FOUND."""
        # Minimal document with only document type
        lines = [
            MockLine("SALE DEED", page_num=1),
        ]
        result, provenance, _ = extract_fields_semantic(lines)

        self.assertIsNone(result["document_number"])
        self.assertEqual(provenance["document_number"]["status"], "NOT_FOUND")

        self.assertIsNone(result["survey_number"])
        self.assertEqual(provenance["survey_number"]["status"], "NOT_FOUND")

        self.assertIsNone(result["stamp_serial_number"])
        self.assertEqual(provenance["stamp_serial_number"]["status"], "NOT_FOUND")

        self.assertIsNone(result["execution_date"])
        self.assertEqual(provenance["execution_date"]["status"], "NOT_FOUND")

    def test_conflicting_candidates_detected(self):
        """Verify conflicting candidates set status: CONFLICT, needs_review: True and retain candidates."""
        cands = [
            FieldCandidate(value="1234/2022", page=1, context="Header Doc No. 1234/2022", score=0.95, reason="Page 1 header"),
            FieldCandidate(value="9876/2022", page=1, context="Endorsement Doc No. 9876/2022", score=0.92, reason="Endorsement stamp"),
        ]
        res = select_best(cands)
        self.assertEqual(res.status, "CONFLICT")
        self.assertTrue(res.needs_review)
        self.assertIn("1234/2022", res.conflicting_candidates)
        self.assertIn("9876/2022", res.conflicting_candidates)

    def test_conflicting_surveys_detected(self):
        """Verify conflicting survey number sets are flagged with CONFLICT and needs_review."""
        cands = [
            FieldCandidate(value="101, 102", page=2, context="Survey Nos. 101, 102", score=0.95, reason="Deed body"),
            FieldCandidate(value="301, 302", page=6, context="Survey Nos. 301, 302", score=0.95, reason="Registration Plan"),
        ]
        res = aggregate_survey_numbers(cands)
        self.assertEqual(res.status, "CONFLICT")
        self.assertTrue(res.needs_review)
        self.assertTrue(len(res.conflicting_candidates) >= 2)


if __name__ == "__main__":
    unittest.main()
