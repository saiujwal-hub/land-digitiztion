#!/usr/bin/env python
"""
test_semantic_accuracy_and_conflicts.py
Unit tests verifying:
1. Two different vendor names (conflict reporting)
2. Corrupted vendor OCR (confidence downscaling, needs_review)
3. Conflicting areas such as 480 and 488 sq. yards
4. Multiple date types (stamp purchase date vs execution date vs registration date)
5. Conflicting survey numbers
6. Missing fields returning null with status NOT_FOUND
7. No sample-value leakage (generic documents with entirely different names/data)
"""

import unittest
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from land_document_extractor import OCRLine
from semantic_extractor import extract_fields_semantic, clean_user_facing_schema
from run_telangana_ocr import locate_field_provenance, locate_party_provenance


class TestSemanticAccuracyAndConflicts(unittest.TestCase):

    def test_two_different_vendor_names_detected_as_conflict(self):
        """Verify that when two fundamentally distinct vendor corporate entities are named, status is CONFLICT."""
        lines = [
            OCRLine(text="THIS SALE DEED MADE BY", score=0.98, x_min=10, y_min=10, x_max=300, y_max=30, page_num=1),
            OCRLine(text="M/s. APEX INFRASTRUCTURE PRIVATE LIMITED", score=0.96, x_min=10, y_min=40, x_max=500, y_max=70, page_num=1),
            OCRLine(text="(HEREINAFTER CALLED THE VENDOR)", score=0.98, x_min=10, y_min=80, x_max=400, y_max=100, page_num=1),
            OCRLine(text="IN FAVOUR OF", score=0.98, x_min=10, y_min=110, x_max=200, y_max=130, page_num=1),
            OCRLine(text="SRI RAMESH KUMAR S/O SURESH KUMAR", score=0.97, x_min=10, y_min=140, x_max=450, y_max=160, page_num=1),
            OCRLine(text="(HEREINAFTER CALLED THE PURCHASER)", score=0.98, x_min=10, y_min=170, x_max=400, y_max=190, page_num=1),
            # Plan page gives a contradictory vendor
            OCRLine(text="VENDOR: M/s VERTEX HOUSING PROJECTS PVT. LTD.", score=0.95, x_min=10, y_min=50, x_max=500, y_max=80, page_num=2),
            OCRLine(text="REPRESENTED BY MANAGING DIRECTOR: SRI K. VIKRAM", score=0.95, x_min=10, y_min=90, x_max=500, y_max=110, page_num=2),
        ]

        result, provenance, _ = extract_fields_semantic(lines)
        parties = result.get("parties_list", [])
        vendor = next((p for p in parties if p.get("role") == "Vendor"), None)
        self.assertIsNotNone(vendor)
        self.assertEqual(vendor["status"], "CONFLICT")
        self.assertTrue(vendor["needs_review"])
        self.assertGreaterEqual(len(vendor["conflicting_candidates"]), 2)
        all_text = " ".join(vendor["candidates"]) + " " + " ".join(vendor["conflicting_candidates"])
        self.assertIn("Apex", all_text)
        self.assertIn("Vertex", all_text)

    def test_corrupted_vendor_ocr_downscales_confidence_and_flags_review(self):
        """Verify noisy OCR vendor line results in calibrated confidence and needs_review=True."""
        lines = [
            OCRLine(text="THIS DEED OF SALE IS MADE BY", score=0.95, x_min=10, y_min=10, x_max=300, y_max=30, page_num=1),
            # Corrupted line: ROPPeBOnTOd bY Sto instead of Represented by Sri
            OCRLine(text="BHARAT BUILDERS PRIVATE LIMITED, ROPPeBOnTOd bY Sto RAJESH", score=0.82, x_min=10, y_min=40, x_max=600, y_max=70, page_num=1),
            OCRLine(text="(HEREINAFTER CALLED THE VENDOR)", score=0.95, x_min=10, y_min=80, x_max=350, y_max=100, page_num=1),
            OCRLine(text="IN FAVOUR OF", score=0.95, x_min=10, y_min=110, x_max=200, y_max=130, page_num=1),
            OCRLine(text="SMT. PRIYA SHARMA W/O ANIL SHARMA", score=0.95, x_min=10, y_min=140, x_max=450, y_max=160, page_num=1),
            OCRLine(text="(HEREINAFTER CALLED THE PURCHASER)", score=0.95, x_min=10, y_min=170, x_max=350, y_max=190, page_num=1),
        ]

        result, provenance, _ = extract_fields_semantic(lines)
        parties = result.get("parties_list", [])
        vendor = next((p for p in parties if p.get("role") == "Vendor"), None)
        self.assertIsNotNone(vendor)
        self.assertTrue(vendor["correction_applied"])
        self.assertTrue(vendor["needs_review"])
        self.assertLess(vendor["confidence"], 0.93)
        self.assertIn("Bharat Builders", vendor["name"])

    def test_conflicting_areas_480_and_488_yield_conflict_status(self):
        """Verify that when 488 sq. yds and 480 sq. yds both appear, area status is CONFLICT."""
        lines = [
            OCRLine(text="SALE DEED", score=0.98, x_min=10, y_min=10, x_max=200, y_max=30, page_num=1),
            # Page 1 deed recital mentions 480 Sq. Yards (401.4 Sq. Metres)
            OCRLine(text="SCHEDULE OF PROPERTY ADMEASURING AN EXTENT OF 480 SQ.YARDS OR 401.4 SQ.MTRS.", score=0.98, x_min=10, y_min=50, x_max=700, y_max=80, page_num=1),
            # Page 2 plan mentions PLOT AREA: 488.0 SQ. YDS.
            OCRLine(text="PLOT AREA : 488.0 SQ. YDS.", score=0.97, x_min=10, y_min=100, x_max=400, y_max=130, page_num=2),
        ]

        result, provenance, _ = extract_fields_semantic(lines)
        area_prov = provenance.get("property_area", {})
        self.assertEqual(area_prov.get("status"), "CONFLICT")
        self.assertTrue(area_prov.get("needs_review"))
        cands = area_prov.get("candidates", []) + area_prov.get("conflicting_candidates", [])
        cands_str = " ".join(str(x) for x in cands)
        self.assertIn("480", cands_str)
        self.assertIn("488", cands_str)

    def test_multiple_date_types_distinguished(self):
        """Verify stamp purchase date, execution date, and registration date have distinct semantics."""
        lines = [
            OCRLine(text="SALE DEED", score=0.98, x_min=10, y_min=10, x_max=200, y_max=30, page_num=1),
            # Stamp sheet date
            OCRLine(text="DATE: 09-10-2003", score=0.95, x_min=10, y_min=40, x_max=200, y_max=60, page_num=1),
            OCRLine(text="SERIAL NO: 12104", score=0.95, x_min=10, y_min=70, x_max=200, y_max=90, page_num=1),
            # Execution date in deed body
            OCRLine(text="THIS DEED OF SALE IS MADE AND EXECUTED ON THIS THE 15TH DAY OF OCTOBER 2003", score=0.98, x_min=10, y_min=100, x_max=800, y_max=130, page_num=1),
            # Link document reference - should NOT be treated as registration date!
            OCRLine(text="REGD. DOCT. NOS. 5121/2002 & 5941/2002 AT S.R.O. GHATKESAR", score=0.95, x_min=10, y_min=150, x_max=700, y_max=180, page_num=2),
        ]

        result, provenance, _ = extract_fields_semantic(lines)
        # Stamp purchase date
        self.assertEqual(result.get("stamp_purchase_date"), "09-10-2003")
        # Execution date
        self.assertEqual(result.get("execution_date"), "15-10-2003")
        # Registration date must NOT be 5121/2002 or 5941/2002
        self.assertIsNone(result.get("registration_date"))
        reg_prov = provenance.get("registration_date", {})
        self.assertEqual(reg_prov.get("status"), "NOT_FOUND")

    def test_conflicting_survey_numbers_reported(self):
        """Verify conflicting survey number sets are flagged with status CONFLICT."""
        lines = [
            OCRLine(text="SALE DEED", score=0.98, x_min=10, y_min=10, x_max=200, y_max=30, page_num=1),
            # Recital claims survey numbers 101, 102
            OCRLine(text="SCHEDULE OF PROPERTY IN SURVEY NO. 101, 102", score=0.95, x_min=10, y_min=50, x_max=500, y_max=80, page_num=1),
            # Plan claims survey numbers 305, 306
            OCRLine(text="REGISTRATION PLAN SHOWING PLOT IN SY. NO. 305, 306", score=0.95, x_min=10, y_min=50, x_max=500, y_max=80, page_num=2),
        ]

        result, provenance, _ = extract_fields_semantic(lines)
        sn_prov = provenance.get("survey_number", {})
        self.assertEqual(sn_prov.get("status"), "CONFLICT")
        self.assertTrue(sn_prov.get("needs_review"))
        self.assertGreaterEqual(len(sn_prov.get("conflicting_candidates", [])), 2)

    def test_missing_fields_return_null_with_status_not_found(self):
        """Verify absent fields return null value with status NOT_FOUND and needs_review=True."""
        lines = [
            OCRLine(text="SALE DEED", score=0.98, x_min=10, y_min=10, x_max=200, y_max=30, page_num=1),
            OCRLine(text="SURVEY NO. 450", score=0.95, x_min=10, y_min=50, x_max=200, y_max=70, page_num=1),
        ]

        result, provenance, _ = extract_fields_semantic(lines)
        user_schema = clean_user_facing_schema(result)

        for field in ["khasra_number", "khata_number", "patta_number", "registration_date"]:
            prov = provenance.get(field, {})
            self.assertIsNone(user_schema.get(field))
            self.assertEqual(prov.get("status"), "NOT_FOUND")
            self.assertEqual(prov.get("confidence"), 0.0)
            self.assertTrue(prov.get("needs_review"))

    def test_no_sample_value_leakage(self):
        """Verify that on a document with completely distinct values, NONE of the Telangana sample values appear."""
        lines = [
            OCRLine(text="SALE DEED", score=0.98, x_min=10, y_min=10, x_max=200, y_max=30, page_num=1),
            OCRLine(text="DOC NO 9999/2015", score=0.95, x_min=10, y_min=40, x_max=300, y_max=60, page_num=1),
            OCRLine(text="THIS DEED MADE BY M/s. KAVERI ESTATES PRIVATE LIMITED", score=0.96, x_min=10, y_min=70, x_max=600, y_max=90, page_num=1),
            OCRLine(text="(HEREINAFTER CALLED THE VENDOR)", score=0.98, x_min=10, y_min=100, x_max=350, y_max=120, page_num=1),
            OCRLine(text="IN FAVOUR OF SRI MANOJ KUMAR S/O DEEPAK KUMAR", score=0.96, x_min=10, y_min=130, x_max=550, y_max=150, page_num=1),
            OCRLine(text="(HEREINAFTER CALLED THE PURCHASER)", score=0.98, x_min=10, y_min=160, x_max=350, y_max=180, page_num=1),
            OCRLine(text="SURVEY NO. 777", score=0.95, x_min=10, y_min=190, x_max=200, y_max=210, page_num=1),
            OCRLine(text="EXTENT OF 350 SQ. YARDS", score=0.96, x_min=10, y_min=220, x_max=300, y_max=240, page_num=1),
            OCRLine(text="SITUATED AT MEDCHAL VILLAGE", score=0.95, x_min=10, y_min=250, x_max=350, y_max=270, page_num=1),
        ]

        result, provenance, _ = extract_fields_semantic(lines)
        user_schema = clean_user_facing_schema(result)

        forbidden = [
            "12736/3", "12736/5", "12736/2003",
            "278, 281, 282", "281, 282",
            "480", "488",
            "Aushapur", "Ghatkesar",
            "Srinidhi Homes", "Suvarna", "Srinivas Reddy"
        ]
        out_str = str(user_schema) + str(provenance)
        for val in forbidden:
            self.assertNotIn(val, out_str, f"Leaked hardcoded sample value '{val}' into generic extraction!")

    def test_provenance_schema_compliance(self):
        """Verify every field in provenance contains all required keys from Requirement 6."""
        lines = [
            OCRLine(text="SALE DEED", score=0.98, x_min=10, y_min=10, x_max=200, y_max=30, page_num=1),
            OCRLine(text="SURVEY NO. 550", score=0.95, x_min=10, y_min=50, x_max=200, y_max=70, page_num=1),
        ]
        result, provenance, _ = extract_fields_semantic(lines)
        user_schema = clean_user_facing_schema(result)

        prov_entry = provenance.get("survey_number", {})
        located = locate_field_provenance(user_schema.get("survey_number"), prov_entry, "survey_number", all_lines=lines)

        required_keys = [
            "value", "status", "confidence", "needs_review",
            "candidates", "page_number", "source_text",
            "original_ocr_value", "correction_applied"
        ]
        for key in required_keys:
            self.assertIn(key, located, f"Missing required provenance key: {key}")

    def test_multiple_stamp_dates_recorded_with_sheet_evidence(self):
        """Verify dates on stamp sheets are stored as stamp_purchase_date/stamp_sheet_dates, not document_date or conflicts."""
        lines = [
            OCRLine(text="SALE DEED", score=0.98, x_min=10, y_min=10, x_max=200, y_max=30, page_num=1),
            OCRLine(text="ONE HUNDRED RUPEES Dt 09-10-2003", score=0.95, x_min=10, y_min=50, x_max=400, y_max=80, page_num=1),
            OCRLine(text="ONE HUNDRED RUPEES DATE: 04-10-2003", score=0.95, x_min=10, y_min=50, x_max=400, y_max=80, page_num=2),
        ]
        result, provenance, _ = extract_fields_semantic(lines)
        user_schema = clean_user_facing_schema(result)

        # Primary stamp date is preserved
        self.assertEqual(user_schema.get("stamp_purchase_date"), "09-10-2003")
        # Multiple stamp dates are in stamp_sheet_dates with page evidence
        sheet_dates = user_schema.get("stamp_sheet_dates", [])
        self.assertGreaterEqual(len(sheet_dates), 2)
        pages_with_dates = [sd["page"] for sd in sheet_dates]
        self.assertIn(1, pages_with_dates)
        self.assertIn(2, pages_with_dates)

        # Not labeled document_date!
        self.assertIsNone(user_schema.get("document_date"))
        doc_date_prov = provenance.get("document_date", {})
        self.assertEqual(doc_date_prov.get("status"), "NOT_FOUND")

        # Stamp purchase date is NOT a conflict
        stamp_prov = provenance.get("stamp_purchase_date", {})
        self.assertEqual(stamp_prov.get("status"), "EXTRACTED")
        self.assertEqual(len(stamp_prov.get("conflicting_candidates", [])), 0)

    def test_city_survey_number_vs_revenue_survey_number(self):
        """Verify city survey number (C.S./C·S.) is preserved separately and never merged with revenue survey number."""
        lines = [
            OCRLine(text="SALE DEED", score=0.98, x_min=10, y_min=10, x_max=200, y_max=30, page_num=1),
            # City survey endorsement in top margin
            OCRLine(text="C·S.12719", score=0.88, x_min=10, y_min=40, x_max=200, y_max=70, page_num=1),
            # Revenue survey in recital
            OCRLine(text="SCHEDULE OF LAND BEARING REVENUE SURVEY NO. 281, 282", score=0.97, x_min=10, y_min=100, x_max=700, y_max=130, page_num=1),
        ]
        result, provenance, _ = extract_fields_semantic(lines)
        user_schema = clean_user_facing_schema(result)

        # Separate city survey number preserved
        self.assertEqual(user_schema.get("city_survey_number"), "12719")
        cs_prov = provenance.get("city_survey_number", {})
        self.assertEqual(cs_prov.get("status"), "EXTRACTED")
        self.assertTrue(cs_prov.get("needs_review"))
        self.assertLessEqual(cs_prov.get("confidence"), 0.80)

        # Revenue survey number isolated
        self.assertEqual(user_schema.get("survey_number"), "281, 282")
        self.assertNotIn("12719", user_schema.get("survey_number"))

    def test_layout_name_versus_locality_address(self):
        """Verify property layout name does not conflict with party address locality (HMT Nagar vs Srinidhi Enclave-II)."""
        lines = [
            OCRLine(text="SALE DEED", score=0.98, x_min=10, y_min=10, x_max=200, y_max=30, page_num=1),
            # Party address locality
            OCRLine(text="PURCHASED BY : P. SRINIVAS REDDY", score=0.95, x_min=10, y_min=40, x_max=400, y_max=60, page_num=1),
            OCRLine(text="P/O. 9-121/5 HMT NAGAR P.R.DIST", score=0.92, x_min=10, y_min=70, x_max=400, y_max=90, page_num=1),
            # Property description layout
            OCRLine(text="PLOT NOS. 1023/1 & 1023/2 OF SRINIDHI ENCLAVE-II SITUATED AT AUSHAPUR", score=0.96, x_min=10, y_min=120, x_max=800, y_max=150, page_num=2),
        ]
        result, provenance, _ = extract_fields_semantic(lines)
        user_schema = clean_user_facing_schema(result)

        # Layout name is Srinidhi Enclave-II and NOT in conflict with HMT Nagar
        self.assertEqual(user_schema.get("layout_name"), "Srinidhi Enclave-II")
        layout_prov = provenance.get("layout_name", {})
        self.assertEqual(layout_prov.get("status"), "EXTRACTED")
        self.assertEqual(len(layout_prov.get("conflicting_candidates", [])), 0)

        # Locality / address holds HMT Nagar separately
        self.assertEqual(user_schema.get("locality_or_address"), "Hmt Nagar")

    def test_multiple_stamp_serial_numbers_recorded_without_conflict(self):
        """Verify differing stamp serials across sheets are stored in stamp_serial_numbers[] and not marked as conflict."""
        lines = [
            OCRLine(text="SALE DEED", score=0.98, x_min=10, y_min=10, x_max=200, y_max=30, page_num=1),
            OCRLine(text="ONE HUNDRED RUPEES SERIAL NO. 12,104", score=0.95, x_min=10, y_min=40, x_max=400, y_max=70, page_num=1),
            OCRLine(text="ONE HUNDRED RUPEES SERIAL NO. 10,610", score=0.95, x_min=10, y_min=40, x_max=400, y_max=70, page_num=2),
            OCRLine(text="ONE HUNDRED RUPEES SERIAL NO. 10,510", score=0.95, x_min=10, y_min=40, x_max=400, y_max=70, page_num=3),
        ]
        result, provenance, _ = extract_fields_semantic(lines)
        user_schema = clean_user_facing_schema(result)

        # Primary serial is sheet 1
        self.assertEqual(user_schema.get("stamp_serial_number"), "12,104")
        stamp_prov = provenance.get("stamp_serial_number", {})
        self.assertEqual(stamp_prov.get("status"), "EXTRACTED")
        self.assertEqual(len(stamp_prov.get("conflicting_candidates", [])), 0)

        # stamp_serial_numbers contains all three sheets
        serials_list = user_schema.get("stamp_serial_numbers", [])
        self.assertEqual(len(serials_list), 3)
        pages = [s["page"] for s in serials_list]
        self.assertEqual(pages, [1, 2, 3])

    def test_genuine_same_field_date_conflict(self):
        """Verify that when multiple explicit document dates contradict, status is CONFLICT."""
        lines = [
            OCRLine(text="SALE DEED", score=0.98, x_min=10, y_min=10, x_max=200, y_max=30, page_num=1),
            OCRLine(text="DOCUMENT DATE: 10-10-2023", score=0.96, x_min=10, y_min=50, x_max=300, y_max=80, page_num=1),
            OCRLine(text="DATE OF DEED: 15-10-2023", score=0.96, x_min=10, y_min=100, x_max=300, y_max=130, page_num=2),
        ]
        result, provenance, _ = extract_fields_semantic(lines)
        dd_prov = provenance.get("document_date", {})
        self.assertEqual(dd_prov.get("status"), "CONFLICT")
        self.assertTrue(dd_prov.get("needs_review"))
        self.assertGreaterEqual(len(dd_prov.get("conflicting_candidates", [])), 2)

    def test_genuine_same_field_area_conflict(self):
        """Verify that area 480 vs 488 for the same property context is flagged as a genuine CONFLICT."""
        lines = [
            OCRLine(text="SALE DEED", score=0.98, x_min=10, y_min=10, x_max=200, y_max=30, page_num=1),
            OCRLine(text="LAND EXTENT ADMEASURING 480 SQ. YARDS", score=0.97, x_min=10, y_min=50, x_max=500, y_max=80, page_num=2),
            OCRLine(text="REGISTRATION PLAN SHOWING PLOT AREA: 488.0 SQ. YDS.", score=0.97, x_min=10, y_min=50, x_max=500, y_max=80, page_num=6),
        ]
        result, provenance, _ = extract_fields_semantic(lines)
        area_prov = provenance.get("property_area", {})
        self.assertEqual(area_prov.get("status"), "CONFLICT")
        self.assertTrue(area_prov.get("needs_review"))
        all_cands = " ".join(str(x) for x in (area_prov.get("candidates", []) + area_prov.get("conflicting_candidates", [])))
        self.assertIn("480", all_cands)
        self.assertIn("488", all_cands)


if __name__ == "__main__":
    unittest.main()

