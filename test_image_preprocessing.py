"""
test_image_preprocessing.py - Comprehensive Unit Tests for Adaptive OCR Image Preprocessing.

Smart India Hackathon (SIH) Prototype: OneBhoomi Land Registry.

Verifies:
  1. Grayscale conversion returns valid image dimensions.
  2. CLAHE/contrast enhancement returns a valid image.
  3. Denoising returns a valid image.
  4. Adaptive thresholding returns a valid image.
  5. Upscaling changes dimensions predictably.
  6. Deskew does not rotate a clean page unnecessarily.
  7. Deskew returns a measurable angle for a synthetic skewed image.
  8. Quality analysis returns real numeric metrics.
  9. Variant generation is deterministic.
  10. Variant selection is deterministic.
  11. Original input image is not modified.
  12. Preprocessing metadata contains applied operations.
  13. OCR coordinates are mapped correctly after upscaling.
  14. Page number is preserved.
  15. Registration-plan-like thin lines are not destroyed by the safest variant.
  16. Existing OCR schemas remain compatible.
  17. Existing semantic extraction tests still pass.
  18. Existing learning tests still pass.
  19. Existing GIS and verification tests still pass.
"""

import copy
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import cv2
import numpy as np

import gis_service
import image_preprocessing
import ocr_learning_service
import verification_service
from land_document_extractor import OCRLine, OCRWord
from semantic_extractor import extract_fields_semantic


class TestImagePreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a synthetic clean 400x300 BGR document image with text lines
        self.sample_h = 400
        self.sample_w = 300
        self.clean_img = np.full((self.sample_h, self.sample_w, 3), 245, dtype=np.uint8)

        # Draw clean horizontal text-like lines
        for y in range(40, 360, 30):
            cv2.line(self.clean_img, (30, y), (270, y), (40, 40, 40), 2)

    # 1. Grayscale conversion returns valid image dimensions
    def test_01_grayscale_conversion_dimensions(self):
        gray = cv2.cvtColor(self.clean_img, cv2.COLOR_BGR2GRAY)
        self.assertEqual(gray.shape, (self.sample_h, self.sample_w))
        self.assertEqual(gray.ndim, 2)

    # 2. CLAHE/contrast enhancement returns a valid image
    def test_02_contrast_enhancement(self):
        enhanced = image_preprocessing.enhance_contrast(self.clean_img, clip_limit=2.0)
        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape, self.clean_img.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    # 3. Denoising returns a valid image
    def test_03_denoising(self):
        noisy = self.clean_img.copy()
        noise = np.random.randint(-15, 15, noisy.shape, dtype=np.int16)
        noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        denoised = image_preprocessing.remove_noise(noisy)
        self.assertIsNotNone(denoised)
        self.assertEqual(denoised.shape, self.clean_img.shape)

    # 4. Adaptive thresholding returns a valid image
    def test_04_adaptive_thresholding(self):
        thresh = image_preprocessing.adaptive_binarize(self.clean_img)
        self.assertIsNotNone(thresh)
        self.assertEqual(thresh.shape[:2], self.clean_img.shape[:2])
        # Binarized output should contain only 0 and 255
        unique_vals = set(np.unique(thresh))
        self.assertTrue(unique_vals.issubset({0, 255}))

    # 5. Upscaling changes dimensions predictably
    def test_05_upscaling_predictable(self):
        scale = 1.5
        upscaled = image_preprocessing.upscale_image(self.clean_img, scale=scale)
        expected_w = int(self.sample_w * scale)
        expected_h = int(self.sample_h * scale)
        self.assertEqual(upscaled.shape[1], expected_w)
        self.assertEqual(upscaled.shape[0], expected_h)

    # 6. Deskew does not rotate a clean page unnecessarily
    def test_06_deskew_clean_page(self):
        rotated, angle = image_preprocessing.correct_skew(self.clean_img)
        self.assertLess(abs(angle), 0.5)
        # Check that image was not altered
        self.assertEqual(rotated.shape, self.clean_img.shape)

    # 7. Deskew returns a measurable angle for a synthetic skewed image
    def test_07_deskew_synthetic_skewed_image(self):
        # Create image with pronounced 3.5-degree skew
        h, w = 600, 500
        skewed_target = np.full((h, w, 3), 250, dtype=np.uint8)
        for y in range(60, 540, 25):
            cv2.line(skewed_target, (40, y), (460, y), (20, 20, 20), 3)

        true_angle = 3.5
        rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), true_angle, 1.0)
        skewed_img = cv2.warpAffine(skewed_target, rot_mat, (w, h), borderValue=(250, 250, 250))

        detected_angle = image_preprocessing.estimate_skew_angle(skewed_img)
        self.assertGreater(abs(detected_angle), 0.5)
        self.assertLess(abs(abs(detected_angle) - abs(true_angle)), 2.5)

    # 8. Quality analysis returns real numeric metrics
    def test_08_quality_analysis_numeric_metrics(self):
        quality = image_preprocessing.analyze_image_quality(self.clean_img)
        self.assertIn("brightness", quality)
        self.assertIn("contrast", quality)
        self.assertIn("blur_score", quality)
        self.assertIn("noise_score", quality)
        self.assertIn("dark_pixel_ratio", quality)
        self.assertIn("white_background_ratio", quality)
        self.assertIn("quality_score", quality)
        self.assertIn("quality_flags", quality)
        self.assertIsInstance(quality["quality_score"], float)

    # 9. Variant generation is deterministic
    def test_09_variant_generation_deterministic(self):
        variants_1 = image_preprocessing.build_preprocessing_variants(self.clean_img)
        variants_2 = image_preprocessing.build_preprocessing_variants(self.clean_img)
        names_1 = [v["name"] for v in variants_1]
        names_2 = [v["name"] for v in variants_2]
        self.assertEqual(names_1, names_2)

    # 10. Variant selection is deterministic
    def test_10_variant_selection_deterministic(self):
        variants = image_preprocessing.build_preprocessing_variants(self.clean_img)
        best_1 = image_preprocessing.select_best_preprocessed_variant(variants)
        best_2 = image_preprocessing.select_best_preprocessed_variant(variants)
        self.assertEqual(best_1["selected_variant"], best_2["selected_variant"])
        self.assertEqual(best_1["selection_reason"], best_2["selection_reason"])

    # 11. Original input image is not modified
    def test_11_original_image_unmodified(self):
        copy_before = self.clean_img.copy()
        _ = image_preprocessing.preprocess_for_ocr(self.clean_img, page_number=1)
        np.testing.assert_array_equal(self.clean_img, copy_before)

    # 12. Preprocessing metadata contains applied operations
    def test_12_preprocessing_metadata_structure(self):
        _, meta = image_preprocessing.preprocess_for_ocr(self.clean_img, page_number=1)
        self.assertIn("page_number", meta)
        self.assertIn("selected_variant", meta)
        self.assertIn("operations", meta)
        self.assertIn("scale", meta)
        self.assertIn("quality_before", meta)
        self.assertIn("quality_after", meta)
        self.assertIsInstance(meta["operations"], list)

    # 13. OCR coordinates are mapped correctly after upscaling
    def test_13_ocr_coordinates_mapped_correctly(self):
        scale = 1.5
        upscaled_box = (150, 300, 300, 450)
        orig_box = image_preprocessing.map_coordinates_to_original(upscaled_box, scale=scale)
        self.assertEqual(orig_box, [100, 200, 200, 300])

    # 14. Page number is preserved
    def test_14_page_number_preserved(self):
        for pg in [1, 2, 6]:
            _, meta = image_preprocessing.preprocess_for_ocr(self.clean_img, page_number=pg)
            self.assertEqual(meta["page_number"], pg)

    # 15. Registration-plan-like thin lines are not destroyed by safest variant
    def test_15_registration_plan_thin_lines_preserved(self):
        # Create registration plan test image with fine 1-pixel boundary line
        plan_img = np.full((300, 300, 3), 255, dtype=np.uint8)
        cv2.line(plan_img, (50, 50), (250, 50), (30, 30, 30), 1)

        variants = image_preprocessing.build_preprocessing_variants(plan_img, page_type="registration_plan")
        selection = image_preprocessing.select_best_preprocessed_variant(variants, page_type="registration_plan")

        # Must not select aggressive adaptive thresholding for registration plans
        self.assertNotIn("threshold", selection["selected_variant"].lower())
        best_img = selection["variant_obj"]["image"]

        # Verify line pixels are intact in selected image
        gray_best = cv2.cvtColor(best_img, cv2.COLOR_BGR2GRAY) if best_img.ndim == 3 else best_img
        line_intensity = np.mean(gray_best[50, 60:240])
        self.assertLess(line_intensity, 200, "Thin boundary line must be preserved and visible")

    # 16. Existing OCR schemas remain compatible
    def test_16_existing_ocr_schemas_compatible(self):
        pts = [[10, 20], [50, 20], [50, 40], [10, 40]]
        word = OCRWord(text="DEED", score=0.98, points=pts)
        self.assertEqual(word.x_min, 10)
        self.assertEqual(word.x_max, 50)
        self.assertEqual(word.y_min, 20)
        self.assertEqual(word.y_max, 40)
        line = OCRLine(text="SALE DEED", score=0.98, x_min=10, y_min=20, x_max=100, y_max=40, page_num=1)
        self.assertEqual(line.text, "SALE DEED")
        self.assertEqual(line.page_num, 1)

    # 17. Existing semantic extraction tests still pass
    def test_17_existing_semantic_extraction_compatible(self):
        test_lines = [
            OCRLine(text="SALE DEED", score=0.99, x_min=100, y_min=40, x_max=600, y_max=80, page_num=1, page_height=2000),
            OCRLine(text="Document No. 12736/5", score=0.95, x_min=100, y_min=100, x_max=450, y_max=140, page_num=1, page_height=2000),
            OCRLine(text="Survey No. 278, 281 extent 480 Sq. Yards", score=0.88, x_min=100, y_min=560, x_max=900, y_max=600, page_num=2, page_height=2000),
        ]
        res, prov, _ = extract_fields_semantic(test_lines)
        self.assertEqual(res["document_type"], "Sale Deed")
        self.assertEqual(res["document_number"], "12736/5")
        self.assertEqual(prov["survey_number"]["value"], "278, 281")

    # 18. Existing learning tests still pass
    def test_18_existing_learning_tests_compatible(self):
        stats = ocr_learning_service.get_learning_stats()
        self.assertIn("verified_feedback_count", stats)
        self.assertIn("learned_rules_count", stats)

    # 19. Existing GIS and verification tests still pass
    def test_19_existing_gis_and_verification_compatible(self):
        # Verification check
        result = {
            "document_type": "Sale Deed",
            "document_number": "12736/5",
            "survey_number": "278, 281",
            "property_area": "480 Sq. Yards",
            "village": "Aushapur",
            "mandal": "Ghatkesar",
            "district": "Ranga Reddy",
        }
        checks = verification_service.run_verification_checks(result)
        self.assertTrue(len(checks) > 0)

        # GIS check
        sample_ocr = {"state": "Karnataka", "property": {"district": "Mandya"}}
        gis_report = gis_service.verify_gis_location(sample_ocr)
        self.assertEqual(gis_report["status"], "resolved")

    # 20. Remote GPU OCR receives preprocessed image and preserves metadata
    @patch("requests.post")
    def test_20_remote_gpu_ocr_receives_preprocessed_image_and_metadata(self, mock_post):
        from land_document_extractor import run_remote_ocr_page_image
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "page_number": 2,
            "lines": [
                {"text": "SURVEY NO 278", "confidence": 0.96, "bbox": [50, 60, 200, 90]}
            ],
            "ocr_time_ms": 15.0,
            "gpu_name": "NVIDIA A100",
        }
        mock_post.return_value = mock_resp

        lines, raw_text, timings = run_remote_ocr_page_image(
            self.clean_img, page_num=2, ocr_url="http://remote-gpu:5000", page_type="property_schedule"
        )

        mock_post.assert_called_once()
        self.assertIn("preprocessing", timings)
        self.assertEqual(timings["preprocessing"]["page_number"], 2)
        self.assertEqual(timings["preprocessing"]["page_type"], "property_schedule")
        self.assertTrue(len(timings["preprocessing"]["operations"]) > 0)
        self.assertIn("image", mock_post.call_args.kwargs["files"])

    # 21. Remote GPU OCR coordinate restoration when scaling is applied
    @patch("image_preprocessing.preprocess_for_ocr")
    @patch("requests.post")
    def test_21_remote_gpu_ocr_coordinate_restoration(self, mock_post, mock_preprocess):
        from land_document_extractor import run_remote_ocr_page_image
        # Simulate preprocessor applying 1.5x upscaling
        mock_meta = {
            "page_number": 1,
            "scale": 1.5,
            "operations": ["grayscale", "clahe", "upscale"],
            "selected_variant": "clahe_denoised_upscaled",
        }
        mock_preprocess.return_value = (self.clean_img, mock_meta)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "lines": [
                {"text": "TELANGANA STAMP", "confidence": 0.98, "bbox": [150, 300, 300, 450]}
            ]
        }
        mock_post.return_value = mock_resp

        lines, _, _ = run_remote_ocr_page_image(
            self.clean_img, page_num=1, ocr_url="http://remote-gpu:5000"
        )
        self.assertEqual(len(lines), 1)
        # Coordinates must be mapped back to original scale (divided by 1.5)
        self.assertEqual(lines[0].x_min, 100)
        self.assertEqual(lines[0].y_min, 200)
        self.assertEqual(lines[0].x_max, 200)
        self.assertEqual(lines[0].y_max, 300)

    # 22. Dashboard preprocessing panel telemetry and baseline labeling
    def test_22_dashboard_panel_telemetry_and_baseline_labeling(self):
        from dashboard_view import _render_preprocessing_panel
        # Test baseline fallback when no runtime telemetry is available
        with patch("image_preprocessing.get_latest_runtime_preprocessing", return_value=None):
            baseline_markup = _render_preprocessing_panel({})
            self.assertIn("REFERENCE PROFILE (TELANGANA DEED BASELINE)", baseline_markup)
            self.assertIn("[Baseline] Document Structure", baseline_markup)
            self.assertIn("[Baseline] Page 1: Stamp Paper", baseline_markup)

        # Test live runtime telemetry rendering
        sample_runtime = {
            "page_number": 2,
            "page_type": "property_schedule",
            "selected_variant": "clahe_denoised",
            "scale": 1.0,
            "operations": ["grayscale", "clahe", "denoise"],
            "deskew_applied": True,
            "skew_angle_degrees": 1.85,
            "selection_reason": "best contrast balance",
            "quality_before": {"quality_score": 0.65, "contrast": 32.0, "blur_score": 80.0},
            "quality_after": {"quality_score": 0.85, "contrast": 56.0, "blur_score": 120.0},
        }
        live_markup = _render_preprocessing_panel(sample_runtime)
        self.assertIn("LIVE RUNTIME METRICS", live_markup)
        self.assertIn("Page 2", live_markup)
        self.assertIn("clahe_denoised", live_markup)
        self.assertIn("65% &rarr; 85%", live_markup)


if __name__ == "__main__":
    unittest.main()
