#!/usr/bin/env python
"""
test_live_multilingual_models.py
Smoke-test workflow for live multilingual PaddleOCR models.

Requirements satisfied:
1. Accepts --lang (en, te, hi, kn, ta, mr, ur).
2. Accepts --image PATH.
3. Loads requested PaddleOCR model without hardcoding model names.
4. Reports exact status:
   - locally available
   - successfully loaded
   - unavailable
   - failed during inference
5. Runs OCR on supplied image and prints:
   detected text, language, confidence, and bounding boxes.
6. NEVER silently falls back to English.
7. Includes --download-models option (safe download in network-enabled environments).
8. If unavailable, exits with non-zero status (code 1), displays exact missing models,
   and instructions for running in Kaggle / networked environments.
9. Includes synthetic metadata test runner distinguishing metadata tests from real OCR inference.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import cv2

# Paddle OpenMP / MKLDNN settings
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from land_document_extractor import (
    get_paddle_ocr_model,
    resolve_paddle_model_names,
    is_model_locally_available,
    detect_script_and_language,
    build_words_from_paddle,
    group_words_into_lines,
    ModelUnavailableError,
    SUPPORTED_LANGUAGE_CODES,
)


def create_synthetic_test_image(text: str, width: int = 800, height: int = 200) -> np.ndarray:
    """
    Generate an in-memory BGR image with text rendered for smoke testing.
    Uses PIL to support Indic fonts when available, or OpenCV fallback.
    """
    from PIL import Image, ImageDraw, ImageFont

    img_pil = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_pil)

    # Try common Indic/system fonts
    font = None
    candidate_fonts = [
        "arial.ttf",
        "gautami.ttf",     # Telugu (Windows)
        "nirmala.ttf",     # Windows Indic Universal
        "mangal.ttf",      # Devanagari
        "tunga.ttf",       # Kannada
        "latha.ttf",       # Tamil
    ]
    for fn in candidate_fonts:
        try:
            font = ImageFont.truetype(fn, 28)
            break
        except Exception:
            continue

    if font is None:
        font = ImageFont.load_default()

    draw.text((30, 80), text, fill=(0, 0, 0), font=font)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def run_synthetic_metadata_tests() -> bool:
    """
    Run synthetic metadata tests that test Unicode script classification and schema formatting.
    DOES NOT claim OCR accuracy.
    """
    print("\n" + "=" * 70)
    print("RUNNING SYNTHETIC METADATA TESTS")
    print("Notice: These tests verify Unicode script detection and metadata contracts.")
    print("They DO NOT execute OCR model inference and DO NOT claim OCR accuracy.")
    print("=" * 70)

    test_cases = [
        ("English Latin", "SALE DEED OF AGRICULTURAL PROPERTY", "Latin", "English"),
        ("Telugu Script", "తెలంగాణ ప్రభుత్వం రిజిస్ట్రేషన్ మరియు స్టాంపుల శాఖ", "Telugu", "Telugu"),
        ("Devanagari Unknown", "विक्रय विलेख भारत सरकार", "Devanagari", "Hindi/Marathi unknown"),
        ("Kannada Script", "ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಕಂದಾಯ ಇಲಾಖೆ", "Kannada", "Kannada"),
        ("Tamil Script", "தமிழ்நாடு அரசு நில ஆவணங்கள்", "Tamil", "Tamil"),
        ("Urdu / Arabic", "اراضی ریکارڈ دستاویز حکومت", "Arabic", "Urdu"),
        ("Telangana Keyword in English", "Government of Telangana Hyderabad District", "Latin", "English"),
    ]

    all_passed = True
    for label, text, exp_script, exp_lang in test_cases:
        script, lang = detect_script_and_language(text)
        passed = (script == exp_script and lang == exp_lang)
        status = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f"  {status} [METADATA-ONLY TEST] {label:30} -> Script: {script} (exp: {exp_script}), Lang: {lang} (exp: {exp_lang})")

    print("=" * 70)
    if all_passed:
        print("[METADATA TESTS PASSED] All 7 Unicode metadata test cases passed.\n")
    else:
        print("[METADATA TESTS FAILED] One or more metadata test cases failed.\n")
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test workflow for live multilingual PaddleOCR models."
    )
    parser.add_argument(
        "--lang",
        choices=["en", "te", "hi", "kn", "ta", "mr", "ur"],
        default="en",
        help="Language to test (en, te, hi, kn, ta, mr, ur)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to the test image file (optional, synthetic sample created if omitted)",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        default=False,
        help="Allow downloading official PaddleOCR models if not locally available (requires internet)",
    )
    parser.add_argument(
        "--test-synthetic-metadata",
        action="store_true",
        default=False,
        help="Run synthetic Unicode metadata tests (distinguishes metadata tests from real OCR inference)",
    )
    args = parser.parse_args()

    if args.test_synthetic_metadata:
        success = run_synthetic_metadata_tests()
        if not args.image and args.lang == "en" and "--lang" not in sys.argv:
            return 0 if success else 1

    lang = args.lang.lower()
    lang_info = SUPPORTED_LANGUAGE_CODES.get(lang, {})
    lang_name = lang_info.get("name", lang)
    backend_lang = lang_info.get("backend_lang", lang)
    det_name, rec_name = resolve_paddle_model_names(lang)

    print("\n" + "=" * 70)
    print(f"PADDLEOCR MULTILINGUAL MODEL SMOKE TEST: '{lang.upper()}' ({lang_name})")
    print("=" * 70)
    print(f"Target Language:          {lang} ({lang_name})")
    print(f"PaddleOCR Backend Code:   {backend_lang}")
    print(f"Resolved Detection Model:  {det_name}")
    print(f"Resolved Recognition Model:{rec_name}")

    # Step 1: Check Local Availability
    is_local = is_model_locally_available(lang)
    if is_local:
        print(f"Local Availability:       YES (Pretrained weights exist on disk)")
    else:
        print(f"Local Availability:       NO (Weights missing from ~/.paddlex/official_models/)")

    # Step 2: Handle Model Loading / Availability Contract
    if not is_local and not args.download_models:
        print("\n" + "!" * 70)
        print(f"[STATUS] MODEL UNAVAILABLE: '{lang}'")
        print("!" * 70)
        print(f"Missing Language:          {lang} ({lang_name})")
        print(f"Missing Detection Weights: {det_name}")
        print(f"Missing Recognition Model: {rec_name}")
        print(f"Searched Local Path:       ~/.paddlex/official_models/")
        print()
        print("[POLICY] The model is configured in the pipeline, but its weights are NOT")
        print("downloaded on this machine. Under the multilingual safety contract:")
        print("  - The pipeline WILL NOT silently fall back to English.")
        print("  - This language IS NOT marked as successfully tested.")
        print("  - OCR accuracy CANNOT and WILL NOT be claimed without real model execution.")
        print()
        print("HOW TO RUN IN KAGGLE OR NETWORK-ENABLED ENVIRONMENTS:")
        print("  1. In your Kaggle Notebook / Linux VM, ensure internet access is enabled:")
        print("       Settings -> Internet: ON")
        print(f"  2. Run this smoke-test with the --download-models flag:")
        print(f"       python test_live_multilingual_models.py --lang {lang} --download-models --image path/to/image.png")
        print("  3. Or download weights directly into ~/.paddlex/official_models/:")
        print(f"       python -c \"from paddleocr import PaddleOCR; PaddleOCR(lang='{backend_lang}')\"")
        print("  4. Once downloaded, re-run standard offline verification:")
        print(f"       python test_live_multilingual_models.py --lang {lang} --image path/to/image.png")
        print("=" * 70 + "\n")
        return 1

    # Step 3: Load Model
    print(f"Loading PaddleOCR Model (allow_download={args.download_models})...")
    try:
        t_load_0 = time.perf_counter()
        model = get_paddle_ocr_model(lang, allow_download=args.download_models)
        load_ms = (time.perf_counter() - t_load_0) * 1000
        print(f"Model Loading Status:     SUCCESSFULLY LOADED ({load_ms:.1f} ms)")
    except ModelUnavailableError as e:
        print(f"\n[STATUS] UNAVAILABLE: {e}")
        return 1
    except Exception as e:
        print(f"\n[STATUS] LOAD FAILED: Unexpected error loading model for '{lang}': {e}")
        return 1

    # Step 4: Prepare Image
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[ERROR] Image file not found: {img_path}")
            return 1
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[ERROR] cv2.imread failed to load image: {img_path}")
            return 1
        img_desc = f"File: {img_path} ({image.shape[1]}x{image.shape[0]})"
    else:
        sample_texts = {
            "en": "GOVERNMENT OF INDIA SALE DEED 2024",
            "te": "తెలంగాణ ప్రభుత్వం రిజిస్ట్రేషన్ శాఖ",
            "hi": "भारत सरकार भूमि अभिलेख विक्रय विलेख",
            "mr": "महाराष्ट्र शासन नोंदणी व मुद्रांक विभाग",
            "kn": "ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಕಂದಾಯ ಇಲಾಖೆ",
            "ta": "தமிழ்நாடு அரசு நில ஆவணங்கள்",
            "ur": "حکومت ہند اراضی ریکارڈ دستاویز",
        }
        sample_txt = sample_texts.get(lang, "LAND RECORD DOCUMENT")
        image = create_synthetic_test_image(sample_txt)
        img_desc = f"In-Memory Synthetic Image with '{sample_txt}' ({image.shape[1]}x{image.shape[0]})"

    print(f"Input Image:              {img_desc}")

    # Step 5: Run Direct OCR Inference (NEVER fall back to English)
    print("\nRunning Direct OCR Inference on Image (Real Model Execution)...")
    try:
        t_inf_0 = time.perf_counter()
        predict_res = model.predict(image)[0]
        inf_ms = (time.perf_counter() - t_inf_0) * 1000

        words = build_words_from_paddle(predict_res)
        lines = group_words_into_lines(words)
    except Exception as e:
        print(f"\n[STATUS] FAILED DURING INFERENCE: {e}")
        return 1

    # Step 6: Print Inference Results
    print("\n" + "=" * 70)
    print(f"[STATUS] REAL OCR INFERENCE SUCCESS: '{lang}'")
    print(f"Inference Time:           {inf_ms:.2f} ms")
    print(f"Text Lines Detected:      {len(lines)}")
    print("=" * 70)

    if not lines:
        print("  (No text lines recognized in image. Inference completed without errors.)")
    else:
        for idx, line in enumerate(lines, start=1):
            detected_script, detected_lang = detect_script_and_language(line.text, context_lang=lang)
            print(
                f"  Line {idx:02d}: \"{line.text}\"\n"
                f"          Language:   {detected_lang}\n"
                f"          Script:     {detected_script}\n"
                f"          Confidence: {line.score:.4f}\n"
                f"          BBox:       [{line.x_min}, {line.y_min}, {line.x_max}, {line.y_max}]"
            )

    print("=" * 70)
    print(f"Language '{lang}' successfully executed real OCR inference.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
