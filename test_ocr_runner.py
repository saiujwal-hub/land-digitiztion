"""
Lightweight test suite for OCR runner correctness (run_telangana_ocr.py).
Mocks PaddleOCR to ensure fast, deterministic execution without GPU/CPU heavy model downloads.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pypdfium2 as pdfium

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from land_document_extractor import OCRLine
import run_telangana_ocr
from run_telangana_ocr import (
    validate_ocr_results,
    parse_args,
    process_document,
    main as runner_main,
)


class TestOCRRunnerCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small 2-page test PDF in a temporary directory
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.pdf_path = Path(cls.temp_dir.name) / "test_two_page.pdf"
        cls.out_json = Path(cls.temp_dir.name) / "test_out.json"

        doc = pdfium.PdfDocument.new()
        doc.new_page(200, 200)
        doc.new_page(200, 200)
        doc.save(str(cls.pdf_path))
        doc.close()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.temp_dir.cleanup()
        except Exception:
            pass

    def setUp(self):
        if self.out_json.exists():
            self.out_json.unlink()

    def test_parse_args_cache_behavior(self):
        """Verify cache flag semantics."""
        # 1. No flag -> use_cache is False (process PDF again by default)
        opts = parse_args([])
        self.assertFalse(opts["use_cache"])

        # 2. --from-cache -> use_cache is True
        opts = parse_args(["--from-cache"])
        self.assertTrue(opts["use_cache"])

        # 3. --reextract -> use_cache is False (must ignore cache)
        opts = parse_args(["--reextract"])
        self.assertFalse(opts["use_cache"])

        # 4. Both flags -> --reextract takes precedence, use_cache is False
        opts = parse_args(["--from-cache", "--reextract"])
        self.assertFalse(opts["use_cache"])

    def test_validation_function(self):
        """Test validation function edge cases."""
        # Perfect run
        is_valid, errors = validate_ocr_results(
            total_pages=2,
            pages_ocr=[{"page_number": 1}, {"page_number": 2}],
            failed_pages=[]
        )
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        # Mismatch in total pages
        is_valid, errors = validate_ocr_results(
            total_pages=2,
            pages_ocr=[{"page_number": 1}],
            failed_pages=[]
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("total_ocr_pages" in e for e in errors))

        # Duplicate page numbers
        is_valid, errors = validate_ocr_results(
            total_pages=2,
            pages_ocr=[{"page_number": 1}, {"page_number": 1}],
            failed_pages=[]
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("Duplicate" in e for e in errors))

        # Failed pages present
        is_valid, errors = validate_ocr_results(
            total_pages=2,
            pages_ocr=[{"page_number": 1}],
            failed_pages=[{"page_number": 2, "error": "Crash"}]
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("Processing failed for pages" in e for e in errors))

    @patch("run_telangana_ocr.run_paddle_ocr_page_image")
    def test_two_page_pdf_fresh_run_and_cache_cycle(self, mock_ocr):
        """
        Verify:
        - Fresh run calls OCR exactly twice (for 2 pages).
        - Page numbers processed are [1, 2].
        - Metadata fields (cache_used, processed_page_numbers, total_pages, total_ocr_pages) are accurate.
        - --from-cache does NOT call OCR.
        - --reextract calls OCR again.
        """
        def dummy_ocr(img, page_num):
            line = OCRLine(
                text=f"Sample deed text on page {page_num}",
                score=0.98,
                x_min=10,
                y_min=10,
                x_max=100,
                y_max=30,
                page_num=page_num
            )
            return [line], f"Sample deed text on page {page_num}", {"ocr_ms": 5.0}

        mock_ocr.side_effect = dummy_ocr

        # -------------------------------------------------------------
        # 1. Fresh Run (no cache)
        # -------------------------------------------------------------
        res_fresh = process_document(
            pdf_path=self.pdf_path,
            out_json=self.out_json,
            use_cache=False
        )

        # Verify OCR called exactly twice
        self.assertEqual(mock_ocr.call_count, 2, "OCR should be called once per page (total 2)")

        # Verify page numbers called are [1, 2]
        called_pages = [
            call.kwargs.get("page_num") if "page_num" in call.kwargs else call.args[1]
            for call in mock_ocr.call_args_list
        ]
        self.assertEqual(called_pages, [1, 2], "OCR should be called for page 1 then page 2")

        # Verify metadata
        meta = res_fresh["metadata"]
        self.assertFalse(meta["cache_used"])
        self.assertEqual(meta["total_pages"], 2)
        self.assertEqual(meta["total_ocr_pages"], 2)
        self.assertEqual(meta["processed_page_numbers"], [1, 2])
        self.assertTrue(meta["complete_processing"])
        self.assertEqual(meta["failed_pages"], [])

        # Verify file saved on disk
        self.assertTrue(self.out_json.exists())

        # -------------------------------------------------------------
        # 2. Run with --from-cache (use_cache=True)
        # -------------------------------------------------------------
        mock_ocr.reset_mock()

        res_cache = runner_main([
            "--from-cache",
            "--pdf", str(self.pdf_path),
            "--out", str(self.out_json)
        ])

        # OCR must NOT be called when loading from cache
        self.assertEqual(mock_ocr.call_count, 0, "OCR should NOT be called when --from-cache is used")

        meta_cache = res_cache["metadata"]
        self.assertTrue(meta_cache["cache_used"])
        self.assertEqual(meta_cache["total_pages"], 2)
        self.assertEqual(meta_cache["total_ocr_pages"], 2)
        self.assertEqual(meta_cache["processed_page_numbers"], [1, 2])
        self.assertTrue(meta_cache["complete_processing"])

        # -------------------------------------------------------------
        # 3. Run with --reextract (use_cache=False)
        # -------------------------------------------------------------
        mock_ocr.reset_mock()

        res_reextract = runner_main([
            "--reextract",
            "--pdf", str(self.pdf_path),
            "--out", str(self.out_json)
        ])

        # OCR must be called exactly twice on reextract
        self.assertEqual(mock_ocr.call_count, 2, "OCR should be called again when --reextract is passed")

        meta_reextract = res_reextract["metadata"]
        self.assertFalse(meta_reextract["cache_used"])
        self.assertEqual(meta_reextract["total_pages"], 2)
        self.assertEqual(meta_reextract["total_ocr_pages"], 2)
        self.assertEqual(meta_reextract["processed_page_numbers"], [1, 2])
        self.assertTrue(meta_reextract["complete_processing"])

    @patch("run_telangana_ocr.run_paddle_ocr_page_image")
    def test_page_failure_records_failed_page_and_does_not_claim_complete(self, mock_ocr):
        """
        Verify requirement 4:
        If any page fails, record the failed page and do not claim complete processing.
        """
        def fail_on_page_2(img, page_num):
            if page_num == 2:
                raise RuntimeError("Simulated OCR failure on page 2")
            return [
                OCRLine(text="Page 1 text", score=0.99, x_min=0, y_min=0, x_max=10, y_max=10, page_num=1)
            ], "Page 1 text", {}

        mock_ocr.side_effect = fail_on_page_2

        res = process_document(
            pdf_path=self.pdf_path,
            out_json=self.out_json,
            use_cache=False
        )

        meta = res["metadata"]
        self.assertFalse(meta["complete_processing"], "Should not claim complete processing on partial failure")
        self.assertEqual(meta["total_pages"], 2)
        self.assertEqual(meta["total_ocr_pages"], 1)
        self.assertEqual(meta["processed_page_numbers"], [1])
        self.assertEqual(len(meta["failed_pages"]), 1)
        self.assertEqual(meta["failed_pages"][0]["page_number"], 2)
        self.assertIn("Simulated OCR failure on page 2", meta["failed_pages"][0]["error"])


if __name__ == "__main__":
    unittest.main()
