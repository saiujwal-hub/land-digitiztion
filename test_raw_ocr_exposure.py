import json
import unittest
import copy
from land_document_extractor import (
    OCRLine,
    build_raw_ocr_payload,
    serialize_ocr_line,
    extract_land_document_from_lines,
)
import verification_service
import ocr_learning_service


import tempfile
import os
from pathlib import Path


class TestRawOCRExposure(unittest.TestCase):
    def setUp(self):
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".json")
        os.close(self.temp_db_fd)
        with open(self.temp_db_path, "w", encoding="utf-8") as f:
            f.write("{}")
        self.orig_db_path = verification_service.DB_PATH
        verification_service.DB_PATH = Path(self.temp_db_path)

        # Sample 6-page multi-page lines representing realistic raw OCR
        self.raw_pages_data = [
            {
                "page_number": 1,
                "raw_text": "GOVERNMENT OF TELANGANA\nREGISTRATION AND STAMPS DEPARTMENT\nSTAMP DUTY RS. 100",
                "lines": [
                    {
                        "line_number": 1,
                        "text": "GOVERNMENT OF TELANGANA",
                        "confidence": 0.965,
                        "bbox": [50, 60, 480, 85],
                        "page_number": 1,
                        "language": "English",
                        "script": "Latin",
                    },
                    {
                        "line_number": 2,
                        "text": "REGISTRATION AND STAMPS DEPARTMENT",
                        "confidence": 0.942,
                        "bbox": [50, 90, 620, 115],
                        "page_number": 1,
                        "language": "English",
                        "script": "Latin",
                    },
                    {
                        "line_number": 3,
                        "text": "STAMP DUTY RS. 100",
                        "confidence": 0.915,
                        "bbox": [50, 120, 310, 142],
                        "page_number": 1,
                        "language": "English",
                        "script": "Latin",
                    },
                ],
                "preprocessing": {"selected_variant": "original_grayscale", "operations": ["grayscale", "deskew"]},
            },
            {
                "page_number": 2,
                "raw_text": "SCHEDULE OF PROPERTY\nSurvey No. 278, 281 in Nanakramguda Village\nTotal Area 400 Sq.Yards",
                "lines": [
                    {
                        "line_number": 1,
                        "text": "SCHEDULE OF PROPERTY",
                        "confidence": 0.958,
                        "bbox": [60, 80, 450, 105],
                        "page_number": 2,
                        "language": "English",
                        "script": "Latin",
                    },
                    {
                        "line_number": 2,
                        "text": "Survey No. 278, 281 in Nanakramguda Village",
                        "confidence": 0.923,
                        "bbox": [60, 115, 680, 140],
                        "page_number": 2,
                        "language": "English",
                        "script": "Latin",
                    },
                    {
                        "line_number": 3,
                        "text": "Total Area 400 Sq.Yards",
                        "confidence": 0.895,
                        "bbox": [60, 150, 420, 175],
                        "page_number": 2,
                        "language": "English",
                        "script": "Latin",
                    },
                ],
                "preprocessing": {"selected_variant": "clahe_enhanced", "operations": ["clahe"]},
            },
            {
                "page_number": 3,
                "raw_text": "SALE DEED EXECUTED BY Sri M. Raghavender\nIn favour of Smt. B. Suvarna",
                "lines": [
                    {
                        "line_number": 1,
                        "text": "SALE DEED EXECUTED BY Sri M. Raghavender",
                        "confidence": 0.935,
                        "bbox": [70, 90, 650, 118],
                        "page_number": 3,
                        "language": "English",
                        "script": "Latin",
                    },
                    {
                        "line_number": 2,
                        "text": "In favour of Smt. B. Suvarna",
                        "confidence": 0.912,
                        "bbox": [70, 125, 480, 150],
                        "page_number": 3,
                        "language": "English",
                        "script": "Latin",
                    },
                ],
                "preprocessing": {"selected_variant": "adaptive_threshold", "operations": ["adaptive_binarize"]},
            },
            {
                "page_number": 4,
                "raw_text": "BOUNDARIES OF THE SCHEDULE PROPERTY\nNorth: Road 30 Feet Wide\nSouth: Plot No. 45",
                "lines": [
                    {
                        "line_number": 1,
                        "text": "BOUNDARIES OF THE SCHEDULE PROPERTY",
                        "confidence": 0.941,
                        "bbox": [55, 75, 590, 100],
                        "page_number": 4,
                        "language": "English",
                        "script": "Latin",
                    },
                    {
                        "line_number": 2,
                        "text": "North: Road 30 Feet Wide",
                        "confidence": 0.918,
                        "bbox": [55, 110, 430, 135],
                        "page_number": 4,
                        "language": "English",
                        "script": "Latin",
                    },
                    {
                        "line_number": 3,
                        "text": "South: Plot No. 45",
                        "confidence": 0.905,
                        "bbox": [55, 145, 320, 170],
                        "page_number": 4,
                        "language": "English",
                        "script": "Latin",
                    },
                ],
                "preprocessing": {"selected_variant": "original_grayscale", "operations": ["grayscale"]},
            },
            {
                "page_number": 5,
                "raw_text": "EXECUTION AND WITNESS CLAUSE\nSigned on this 22nd day of August 2024",
                "lines": [
                    {
                        "line_number": 1,
                        "text": "EXECUTION AND WITNESS CLAUSE",
                        "confidence": 0.932,
                        "bbox": [80, 100, 520, 125],
                        "page_number": 5,
                        "language": "English",
                        "script": "Latin",
                    },
                    {
                        "line_number": 2,
                        "text": "Signed on this 22nd day of August 2024",
                        "confidence": 0.884,
                        "bbox": [80, 135, 590, 160],
                        "page_number": 5,
                        "language": "English",
                        "script": "Latin",
                    },
                ],
                "preprocessing": {"selected_variant": "original_grayscale", "operations": ["grayscale"]},
            },
            {
                "page_number": 6,
                "raw_text": "REGISTRATION PLAN OF PLOT IN SY NO 278, 281\nLayout approved by Gram Panchayat Nanakramguda",
                "lines": [
                    {
                        "line_number": 1,
                        "text": "REGISTRATION PLAN OF PLOT IN SY NO 278, 281",
                        "confidence": 0.915,
                        "bbox": [40, 50, 680, 80],
                        "page_number": 6,
                        "language": "English",
                        "script": "Latin",
                    },
                    {
                        "line_number": 2,
                        "text": "Layout approved by Gram Panchayat Nanakramguda",
                        "confidence": 0.892,
                        "bbox": [40, 90, 710, 118],
                        "page_number": 6,
                        "language": "English",
                        "script": "Latin",
                    },
                ],
                "preprocessing": {"selected_variant": "contrast_boost", "operations": ["clahe", "unsharp_mask"]},
            },
        ]

    def tearDown(self):
        verification_service.DB_PATH = self.orig_db_path
        try:
            if os.path.exists(self.temp_db_path):
                os.remove(self.temp_db_path)
        except Exception:
            pass

    def test_build_raw_ocr_payload_preserves_all_pages(self):
        """Test 1 & 2: Raw OCR is built and contains all 6 pages."""
        raw_ocr = build_raw_ocr_payload(
            self.raw_pages_data,
            backend="remote_gpu",
            model="PaddleOCR (Multilingual PP-OCRv6 GPU)",
            gpu_hardware="Tesla T4",
        )
        self.assertEqual(raw_ocr["backend"], "remote_gpu")
        self.assertEqual(raw_ocr["model"], "PaddleOCR (Multilingual PP-OCRv6 GPU)")
        self.assertEqual(raw_ocr["gpu_hardware"], "Tesla T4")
        self.assertEqual(raw_ocr["total_pages"], 6)
        self.assertEqual(len(raw_ocr["pages"]), 6)

        page_nums = [p["page_number"] for p in raw_ocr["pages"]]
        self.assertEqual(page_nums, [1, 2, 3, 4, 5, 6])

    def test_raw_ocr_lines_exact_fidelity(self):
        """Test 3, 4, 5, 6: Exact text, confidence, bounding boxes, page numbers preserved."""
        raw_ocr = build_raw_ocr_payload(
            self.raw_pages_data,
            backend="remote_gpu",
            model="PaddleOCR",
        )
        p2 = raw_ocr["pages"][1]
        self.assertEqual(p2["page_number"], 2)
        self.assertIn("Survey No. 278, 281 in Nanakramguda Village", p2["raw_text"])

        l2 = p2["lines"][1]
        self.assertEqual(l2["text"], "Survey No. 278, 281 in Nanakramguda Village")
        self.assertEqual(l2["confidence"], 0.923)
        self.assertEqual(l2["bbox"], [60, 115, 680, 140])
        self.assertEqual(l2["page_number"], 2)
        self.assertEqual(l2["language"], "English")
        self.assertEqual(l2["script"], "Latin")

    def test_raw_ocr_immutability_through_lifecycle(self):
        """Test 7, 8, 9: Raw OCR remains 100% unchanged through normalization, learning, clerk correction, approval, rejection."""
        raw_ocr = build_raw_ocr_payload(
            self.raw_pages_data,
            backend="remote_gpu",
            model="PaddleOCR (Multilingual PP-OCRv6 GPU)",
            gpu_hardware="Tesla T4",
        )
        initial_raw_ocr_json = json.dumps(raw_ocr, sort_keys=True)

        # 1. Semantic normalization step
        ocr_lines = []
        for p in self.raw_pages_data:
            for l in p["lines"]:
                ocr_lines.append(OCRLine(
                    text=l["text"],
                    score=l["confidence"],
                    x_min=l["bbox"][0],
                    y_min=l["bbox"][1],
                    x_max=l["bbox"][2],
                    y_max=l["bbox"][3],
                    page_num=l["page_number"],
                    language=l["language"],
                    script=l["script"],
                ))
        full_text = "\n\n".join(p["raw_text"] for p in self.raw_pages_data)
        extraction_result = extract_land_document_from_lines(
            ocr_lines, full_text, image_path="fake.pdf", raw_ocr=raw_ocr
        )

        # Ensure raw_ocr inside result is identical
        self.assertEqual(json.dumps(extraction_result["raw_ocr"], sort_keys=True), initial_raw_ocr_json)

        # 2. Verification record creation
        record = verification_service.create_verification_record(extraction_result)
        self.assertIn("raw_ocr", record)
        self.assertEqual(json.dumps(record["raw_ocr"], sort_keys=True), initial_raw_ocr_json)

        # 3. Clerk manual corrections (modify document_payload, verify raw_ocr is NOT changed)
        old_payload = copy.deepcopy(record["document_payload"])
        record["document_payload"]["survey_number"] = "278, 281, 282"
        record["document_payload"]["document_number"] = "4512/2024"
        form_fields = {
            "survey_number": "278, 281, 282",
            "document_number": "4512/2024",
        }
        ocr_learning_service.capture_changed_payload_fields(
            old_payload=old_payload,
            new_form_fields=form_fields,
            verification_id=record["verification_id"],
            field_provenance=record.get("field_provenance", {}),
        )
        verification_service.save_record(record)

        loaded_rec = verification_service.get_record(record["verification_id"])
        self.assertEqual(loaded_rec["document_payload"]["survey_number"], "278, 281, 282")
        self.assertEqual(json.dumps(loaded_rec["raw_ocr"], sort_keys=True), initial_raw_ocr_json)

        # 4. Officer approval & sealing
        loaded_rec["status"] = "APPROVED"
        loaded_rec["approved_at"] = "2024-08-22T10:00:00Z"
        loaded_rec["signature"] = "sample_rsa_sig_base64"
        verification_service.save_record(loaded_rec)

        approved_rec = verification_service.get_record(record["verification_id"])
        self.assertEqual(approved_rec["status"], "APPROVED")
        self.assertEqual(json.dumps(approved_rec["raw_ocr"], sort_keys=True), initial_raw_ocr_json)

        # 5. Revoke / Rejection
        approved_rec["status"] = "REJECTED"
        approved_rec["rejection_reason"] = "Audit discrepancy test"
        verification_service.save_record(approved_rec)

        rejected_rec = verification_service.get_record(record["verification_id"])
        self.assertEqual(rejected_rec["status"], "REJECTED")
        self.assertEqual(json.dumps(rejected_rec["raw_ocr"], sort_keys=True), initial_raw_ocr_json)

    def test_local_cpu_backend_metadata_never_hardcoded(self):
        """Test 13: Local CPU backend is labeled local_cpu, not remote_gpu."""
        lines = [OCRLine(text="Land Registry Record", score=0.95, x_min=10, y_min=10, x_max=200, y_max=30, page_num=1)]
        cpu_result = extract_land_document_from_lines(
            lines, "Land Registry Record", "fake.png", timings={"ocr_backend": "local_cpu", "model_name": "PaddleOCR (PP-OCRv6 CPU)"}
        )
        raw_ocr = cpu_result.get("raw_ocr")
        self.assertIsNotNone(raw_ocr)
        self.assertEqual(raw_ocr["backend"], "local_cpu")
        self.assertEqual(raw_ocr["model"], "PaddleOCR (PP-OCRv6 CPU)")


if __name__ == "__main__":
    unittest.main()
