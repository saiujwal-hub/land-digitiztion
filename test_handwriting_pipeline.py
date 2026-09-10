"""
test_handwriting_pipeline.py

Unit tests for handwriting region detection, safeguard enforcement, and pluggable recognition.
Ensures:
1. Detection method is labeled 'heuristic' (or 'model' only when trained model runs).
2. Low-confidence lines are NOT called handwriting merely because OCR is low; stored as 'possible_handwriting'.
3. Handwriting is separated from printed text, signatures, stamps/seals, and damaged regions.
4. Printed PaddleOCR is NEVER used as handwriting recognition.
5. Missing HTR models return MODEL_UNAVAILABLE with warning.
6. Structured fields are never overwritten with handwriting output.
7. Required metadata schema is strictly maintained.
"""

import unittest
import numpy as np
from land_document_extractor import (
    OCRLine,
    detect_handwriting_regions,
    detect_handwriting,
    HandwritingRecognizer,
    get_handwriting_recognizer,
    build_handwriting_metadata,
)


class TestHandwritingPipeline(unittest.TestCase):

    def test_detection_method_labeling_heuristic(self):
        """Rule 1: Region detected via coordinates/rules must be labeled 'heuristic'."""
        lines = [
            OCRLine("C.S. 12719", 0.85, 50, 40, 200, 80, page_num=1),
        ]
        regions = detect_handwriting_regions(lines, image_height=1000, image_width=800, page_num=1)
        self.assertEqual(len(regions), 1)
        reg = regions[0]
        self.assertEqual(reg["detection_method"], "heuristic")
        self.assertEqual(reg["region_type"], "possible_handwriting")
        self.assertTrue(reg["needs_review"])

    def test_low_confidence_printed_not_called_handwriting_merely_due_to_score(self):
        """Rule 2: Low-confidence printed line is 'possible_handwriting' and needs_review=True."""
        lines = [
            OCRLine("Short noisy line", 0.50, 100, 500, 300, 530, page_num=2),
        ]
        regions = detect_handwriting_regions(lines, image_height=1000, image_width=800, page_num=2)
        self.assertEqual(len(regions), 1)
        reg = regions[0]
        self.assertEqual(reg["region_type"], "possible_handwriting")
        self.assertEqual(reg["detection_method"], "heuristic")
        self.assertTrue(reg["needs_review"])

    def test_separate_handwriting_from_stamps_signatures_and_boilerplate(self):
        """Rule 3: Separate handwriting from stamps/seals, signatures, and damaged printed text."""
        lines = [
            # Stamp / seal line
            OCRLine("SUB REGISTRAR OFFICE STAMP SEAL", 0.90, 100, 100, 500, 140, page_num=1),
            # Bottom signature block
            OCRLine("SIGNATURE OF EXECUTANT", 0.85, 100, 920, 400, 960, page_num=1),
            # Printed legal boilerplate with low confidence (damaged/blurred printed text)
            OCRLine("HEREINAFTER CALLED THE VENDOR OF THE ONE PART SITUATED AT HYDERABAD", 0.45, 50, 400, 750, 430, page_num=1),
        ]
        regions = detect_handwriting_regions(lines, image_height=1000, image_width=800, page_num=1)
        # None of these should be classified as handwriting
        self.assertEqual(len(regions), 0)

    def test_model_unavailable_when_no_htr_installed(self):
        """Rule 4 & 5: When no dedicated HTR model is installed, returns MODEL_UNAVAILABLE."""
        recognizer = HandwritingRecognizer(backend="auto")
        # In current environment, no TrOCR/HTR weights exist
        self.assertEqual(recognizer.model_status, "UNAVAILABLE")

        sample_region = {
            "page_number": 1,
            "region_type": "possible_handwriting",
            "detection_method": "heuristic",
            "bounding_box": [50, 40, 200, 80],
            "detection_confidence": 0.65,
            "recognition_status": "NOT_RUN",
            "needs_review": True,
        }
        result = recognizer.recognize_region(sample_region)

        self.assertEqual(result["recognition_status"], "MODEL_UNAVAILABLE")
        self.assertIsNone(result["recognized_text"])
        self.assertEqual(result["recognition_confidence"], 0.0)
        self.assertTrue(result["needs_review"])
        self.assertIn("unavailable", result["warning"].lower())

    def test_printed_paddleocr_never_used_as_handwriting_recognition(self):
        """Rule 4: Recognizer must not copy printed OCR text into recognized_text."""
        recognizer = HandwritingRecognizer(backend="auto")
        region = {
            "page_number": 1,
            "bounding_box": [10, 10, 100, 50],
            "detection_method": "heuristic",
            "text": "Some printed text",
        }
        result = recognizer.recognize_region(region)
        self.assertIsNone(result["recognized_text"])
        self.assertNotEqual(result.get("recognized_text"), "Some printed text")

    def test_metadata_schema_requirements(self):
        """Rule 7: Every region must contain full required attributes."""
        recognizer = HandwritingRecognizer(backend="auto")
        region = {
            "page_number": 1,
            "bounding_box": [10, 10, 100, 50],
            "detection_method": "heuristic",
            "detection_confidence": 0.60,
        }
        processed = recognizer.recognize_region(region)
        expected_keys = [
            "page_number",
            "bounding_box",
            "detection_method",
            "detection_confidence",
            "recognition_status",
            "recognized_text",
            "recognition_confidence",
            "source_model",
            "needs_review",
            "warning",
        ]
        for key in expected_keys:
            self.assertIn(key, processed, f"Missing key: {key}")

        metadata = build_handwriting_metadata([processed])
        self.assertIn("detected", metadata)
        self.assertIn("regions", metadata)
        self.assertIn("recognized_region_count", metadata)
        self.assertIn("unrecognized_region_count", metadata)
        self.assertIn("manual_review_required", metadata)
        self.assertEqual(metadata["unrecognized_region_count"], 1)
        self.assertEqual(metadata["recognized_region_count"], 0)

    def test_segment_region_line_crops_preserves_bounding_box(self):
        """Requirement 7: Test line crops creation while preserving original region bbox."""
        from land_document_extractor import segment_region_line_crops
        # Single line box
        single_bbox = [100, 50, 400, 90]
        crops = segment_region_line_crops(single_bbox, image_height=1000, image_width=800)
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0], single_bbox)

        # Multi-line box (height 200 > 75 * 1.5)
        multi_bbox = [100, 50, 400, 250]
        multi_crops = segment_region_line_crops(multi_bbox, image_height=1000, image_width=800, max_line_height=75)
        self.assertGreater(len(multi_crops), 1)
        self.assertEqual(multi_crops[0][1], 50)
        self.assertEqual(multi_crops[-1][3], 250)

    def test_remote_htr_mock_success(self):
        """Requirement 12: Successful mocked HTR response with RECOGNIZED and needs_review=True."""
        from unittest.mock import patch, MagicMock
        recognizer = HandwritingRecognizer(backend="remote", ocr_url="http://fake-kaggle:5000")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "page_number": 1,
            "region_id": "page1_region1",
            "recognition_status": "RECOGNIZED",
            "recognized_text": "C.S. 12719",
            "confidence": 0.9123,
            "language": "English",
            "script": "Latin",
            "source_model": "microsoft/trocr-base-handwritten",
            "needs_review": True,
            "warning": None,
        }

        dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
        region = {
            "page_number": 1,
            "region_id": "page1_region1",
            "bounding_box": [10, 10, 100, 50],
            "detection_method": "heuristic",
            "detection_confidence": 0.8,
        }

        with patch("requests.post", return_value=mock_resp):
            processed = recognizer.recognize_region(region, image=dummy_img)

        self.assertEqual(processed["recognition_status"], "RECOGNIZED")
        self.assertEqual(processed["recognized_text"], "C.S. 12719")
        self.assertEqual(processed["source_model"], "microsoft/trocr-base-handwritten")
        self.assertEqual(processed["recognition_confidence"], 0.9123)
        self.assertTrue(processed["needs_review"])
        self.assertIn("line_crop_boxes", processed)

    def test_remote_htr_unavailable_response(self):
        """Requirement 12: Remote endpoint returns MODEL_UNAVAILABLE with warning."""
        from unittest.mock import patch, MagicMock
        recognizer = HandwritingRecognizer(backend="remote", ocr_url="http://fake-kaggle:5000")

        mock_resp = MagicMock()
        mock_resp.status_code = 404

        dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
        region = {
            "page_number": 1,
            "bounding_box": [10, 10, 100, 50],
            "detection_method": "heuristic",
        }

        with patch("requests.post", return_value=mock_resp):
            processed = recognizer.recognize_region(region, image=dummy_img)

        self.assertEqual(processed["recognition_status"], "MODEL_UNAVAILABLE")
        self.assertIsNone(processed["recognized_text"])
        self.assertTrue(processed["needs_review"])
        self.assertIn("unavailable", processed["warning"].lower())

    def test_remote_htr_inference_failure(self):
        """Requirement 12: Remote server returns INFERENCE_FAILED on corrupt crop."""
        from unittest.mock import patch, MagicMock
        recognizer = HandwritingRecognizer(backend="remote", ocr_url="http://fake-kaggle:5000")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "page_number": 1,
            "region_id": "page1_reg1",
            "recognition_status": "INFERENCE_FAILED",
            "recognized_text": None,
            "confidence": 0.0,
            "needs_review": True,
            "warning": "CUDA out of memory",
        }

        dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
        region = {
            "page_number": 1,
            "bounding_box": [10, 10, 100, 50],
            "detection_method": "heuristic",
        }

        with patch("requests.post", return_value=mock_resp):
            processed = recognizer.recognize_region(region, image=dummy_img)

        self.assertEqual(processed["recognition_status"], "INFERENCE_FAILED")
        self.assertIsNone(processed["recognized_text"])
        self.assertTrue(processed["needs_review"])
        self.assertEqual(processed["warning"], "CUDA out of memory")


if __name__ == "__main__":
    unittest.main()
