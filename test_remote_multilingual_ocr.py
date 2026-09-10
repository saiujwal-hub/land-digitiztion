#!/usr/bin/env python
"""
test_remote_multilingual_ocr.py
Unit tests for the remote Kaggle/GPU OCR server and client integration.

Requirements tested:
1. 'en' is forwarded correctly (client -> server -> response).
2. 'te' is forwarded correctly and uses Telugu model, not English.
3. Unavailable language returns structured failure with needs_review=True and model_status="UNAVAILABLE".
4. No language silently changes to English.
5. Model cache in kaggle_gpu_server prevents reloading on every page.
6. run_telangana_ocr.py forwards --ocr-url and --lang to remote client.
"""

import sys
import io
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from land_document_extractor import OCRLine, run_remote_ocr_page_image
import kaggle_gpu_server
from kaggle_gpu_server import app, _GPU_OCR_MODELS, get_gpu_ocr_model
import run_telangana_ocr


class TestRemoteMultilingualOCR(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        # Sample test image
        img = np.full((100, 300, 3), 255, dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", img)
        self.img_bytes = enc.tobytes()

    def test_en_forwarded_and_processed_correctly_on_server(self):
        """Verify server accepts lang='en', page_number=1, and returns standard metadata."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [{
            "rec_texts": ["GOVERNMENT OF TELANGANA"],
            "rec_scores": [0.985],
            "rec_polys": [[[10, 10], [200, 10], [200, 40], [10, 40]]],
        }]

        with patch("kaggle_gpu_server.get_gpu_ocr_model", return_value=(mock_model, None)) as mock_getter:
            data = {
                "image": (io.BytesIO(self.img_bytes), "page_1.jpg"),
                "page_number": "1",
                "lang": "en",
            }
            resp = self.client.post("/ocr", data=data, content_type="multipart/form-data")
            self.assertEqual(resp.status_code, 200)
            res_json = resp.get_json()

            # Verify model getter was called with 'en'
            mock_getter.assert_called_with("en")

            # Verify response schema
            self.assertEqual(res_json["page_number"], 1)
            self.assertEqual(res_json["requested_language"], "en")
            self.assertEqual(res_json["ocr_model_language"], "en")
            self.assertEqual(res_json["model_status"], "AVAILABLE")
            self.assertFalse(res_json["needs_review"])
            self.assertEqual(len(res_json["lines"]), 1)
            self.assertEqual(res_json["lines"][0]["text"], "GOVERNMENT OF TELANGANA")
            self.assertEqual(res_json["lines"][0]["language"], "en")

    def test_te_forwarded_and_uses_telugu_model(self):
        """Verify server accepts lang='te', page_number=2, and loads/uses Telugu model (not English)."""
        mock_te_model = MagicMock()
        mock_te_model.predict.return_value = [{
            "rec_texts": ["తెలంగాణ ప్రభుత్వం"],
            "rec_scores": [0.965],
            "rec_polys": [[[10, 10], [250, 10], [250, 40], [10, 40]]],
        }]

        with patch("kaggle_gpu_server.get_gpu_ocr_model", return_value=(mock_te_model, None)) as mock_getter:
            data = {
                "image": (io.BytesIO(self.img_bytes), "page_2.jpg"),
                "page_number": "2",
                "lang": "te",
            }
            resp = self.client.post("/ocr", data=data, content_type="multipart/form-data")
            self.assertEqual(resp.status_code, 200)
            res_json = resp.get_json()

            # Verify model getter was explicitly called with 'te'
            mock_getter.assert_called_with("te")

            # Verify returned model language is 'te' and NOT 'en'
            self.assertEqual(res_json["requested_language"], "te")
            self.assertEqual(res_json["ocr_model_language"], "te")
            self.assertNotEqual(res_json["ocr_model_language"], "en")
            self.assertEqual(res_json["model_status"], "AVAILABLE")
            self.assertFalse(res_json["needs_review"])
            self.assertEqual(res_json["page_number"], 2)

    def test_unavailable_language_returns_structured_failure(self):
        """Verify unavailable language returns structured error with needs_review=True and model_status=UNAVAILABLE."""
        with patch("kaggle_gpu_server.get_gpu_ocr_model", return_value=(None, "Model weights missing")):
            data = {
                "image": (io.BytesIO(self.img_bytes), "page_1.jpg"),
                "page_number": "1",
                "lang": "ur",
            }
            resp = self.client.post("/ocr", data=data, content_type="multipart/form-data")
            self.assertEqual(resp.status_code, 400)
            res_json = resp.get_json()

            self.assertEqual(res_json["requested_language"], "ur")
            self.assertIsNone(res_json["ocr_model_language"])
            self.assertEqual(res_json["model_status"], "UNAVAILABLE")
            self.assertTrue(res_json["needs_review"])
            self.assertTrue(len(res_json["warnings"]) > 0)
            self.assertIn("Model weights missing", res_json["warnings"][0])

    def test_no_language_silently_changes_to_english(self):
        """When an Indic language is requested, the server must NEVER silently substitute English."""
        # When Telugu model fails to load:
        with patch("kaggle_gpu_server.get_gpu_ocr_model", return_value=(None, "PaddleOCR te weights not found")):
            data = {
                "image": (io.BytesIO(self.img_bytes), "page_1.jpg"),
                "page_number": "1",
                "lang": "te",
            }
            resp = self.client.post("/ocr", data=data, content_type="multipart/form-data")
            res_json = resp.get_json()

            # Must NOT claim English success
            self.assertNotEqual(res_json.get("ocr_model_language"), "en")
            self.assertEqual(res_json["model_status"], "UNAVAILABLE")
            self.assertTrue(res_json["needs_review"])

    def test_model_cache_prevents_reloading_on_every_page(self):
        """Verify model cache reuses already loaded model instance without re-instantiating PaddleOCR."""
        # Clear cache for test isolation
        with kaggle_gpu_server._GPU_OCR_LOCK:
            _GPU_OCR_MODELS.clear()

        with patch("kaggle_gpu_server.PaddleOCR") as mock_paddle_cls:
            dummy_instance = MagicMock()
            mock_paddle_cls.return_value = dummy_instance

            # Request 1: should instantiate
            m1, err1 = get_gpu_ocr_model("en")
            self.assertIsNone(err1)
            self.assertEqual(mock_paddle_cls.call_count, 1)

            # Request 2: should hit cache
            m2, err2 = get_gpu_ocr_model("en")
            self.assertIsNone(err2)
            self.assertEqual(mock_paddle_cls.call_count, 1)
            self.assertIs(m1, m2)

            # Request 3: should hit cache
            m3, err3 = get_gpu_ocr_model("en")
            self.assertIsNone(err3)
            self.assertEqual(mock_paddle_cls.call_count, 1)
            self.assertIs(m1, m3)

        # Clean up
        with kaggle_gpu_server._GPU_OCR_LOCK:
            _GPU_OCR_MODELS.clear()

    @patch("requests.post")
    def test_client_run_remote_ocr_page_image_forwards_parameters(self, mock_post):
        """Verify run_remote_ocr_page_image client correctly passes files, data (page_number, lang)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "page_number": 3,
            "requested_language": "te",
            "ocr_model_language": "te",
            "model_status": "AVAILABLE",
            "lines": [
                {
                    "text": "విక్రయ దస్తావేజు",
                    "confidence": 0.95,
                    "bbox": [10, 20, 150, 50],
                    "language": "te",
                }
            ],
            "needs_review": False,
            "warnings": [],
            "ocr_time_ms": 12.5,
        }
        mock_post.return_value = mock_resp

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        lines, raw_text, timings = run_remote_ocr_page_image(
            img, page_num=3, lang="te", ocr_url="http://remote-gpu:5000"
        )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        self.assertEqual(call_kwargs["data"]["page_number"], "3")
        self.assertEqual(call_kwargs["data"]["lang"], "te")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].text, "విక్రయ దస్తావేజు")
        self.assertEqual(lines[0].language, "Telugu")
        self.assertEqual(lines[0].script, "Telugu")
        self.assertFalse(timings["needs_review"])

    def test_run_telangana_ocr_parse_args_remote_and_lang(self):
        """Verify run_telangana_ocr.py parse_args parses --ocr-url and --lang."""
        opts = run_telangana_ocr.parse_args([
            "--ocr-url", "http://127.0.0.1:5000",
            "--lang", "te"
        ])
        self.assertEqual(opts["ocr_url"], "http://127.0.0.1:5000")
        self.assertEqual(opts["lang"], "te")

    @patch("run_telangana_ocr.run_remote_ocr_page_image")
    @patch("run_telangana_ocr.pdfium.PdfDocument")
    def test_remote_page_failure_marks_document_incomplete(self, mock_pdf_doc, mock_remote_ocr):
        """
        Requirement 7: Test failure behavior:
        - simulate one remote page failure
        - verify the document is marked incomplete
        - verify fallback_used remains false unless explicitly enabled
        - verify the failed page is recorded
        """
        import tempfile
        # Mock 2-page PDF
        mock_pdf = MagicMock()
        mock_pdf.__len__.return_value = 2
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        mock_pil = MagicMock()
        mock_pil.to_pil.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_page1.render.return_value = mock_pil
        mock_page2.render.return_value = mock_pil
        mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2]
        mock_pdf_doc.return_value = mock_pdf

        # Page 1 succeeds, Page 2 fails remotely
        line1 = OCRLine(text="Page 1 remote line", score=0.95, x_min=10, y_min=20, x_max=100, y_max=40, page_num=1)
        
        def fake_remote(image, page_num=1, lang="en", ocr_url="", timeout=60):
            if page_num == 1:
                return [line1], "Page 1 remote line", {
                    "ocr_language_model_status": "AVAILABLE",
                    "model_status": "AVAILABLE",
                    "needs_review": False,
                    "warnings": [],
                }
            else:
                return [], "", {
                    "ocr_language_model_status": "UNAVAILABLE",
                    "model_status": "UNAVAILABLE",
                    "needs_review": True,
                    "warnings": ["Remote GPU inference error on page 2"],
                }

        mock_remote_ocr.side_effect = fake_remote

        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_pdf = Path(tmp_dir) / "fake.pdf"
            fake_pdf.write_bytes(b"%PDF-1.4 dummy")
            out_file = Path(tmp_dir) / "test_remote_failure.json"
            result = run_telangana_ocr.process_document(
                pdf_path=fake_pdf,
                out_json=out_file,
                use_cache=False,
                lang="en",
                ocr_url="http://remote-gpu:5000",
            )

            meta = result["metadata"]
            self.assertEqual(meta["execution_mode"], "remote_gpu")
            self.assertEqual(meta["requested_language"], "en")
            # Verify document is marked incomplete
            self.assertFalse(meta["complete_processing"])
            # Verify fallback_used remains false (no silent fallback to local CPU)
            self.assertFalse(meta["fallback_used"])
            # Verify failed page is recorded
            self.assertEqual(len(meta["failed_pages"]), 1)
            self.assertEqual(meta["failed_pages"][0]["page_number"], 2)
            # Verify remote_failures recorded
            self.assertEqual(len(meta["remote_failures"]), 1)
            self.assertEqual(meta["remote_failures"][0]["page_number"], 2)
            # Verify pages_sent and pages_received
            self.assertEqual(meta["pages_sent"], [1, 2])
            self.assertEqual(meta["pages_received"], [1])
            self.assertEqual(meta["total_ocr_pages"], 1)
            self.assertEqual(meta["total_pages"], 2)


if __name__ == "__main__":
    unittest.main()
