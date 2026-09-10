#!/usr/bin/env python
"""
test_kaggle_smoke.py
Real Kaggle / Remote GPU OCR Smoke-Test.

Tests:
1. English image with --lang en
2. Telugu image with --lang te
3. Reports model loading and OCR success separately.
4. Does not claim accuracy from a clean synthetic image.
"""

import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import sys
import io
import time
from pathlib import Path
import numpy as np
import cv2

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import kaggle_gpu_server
from kaggle_gpu_server import app, get_gpu_ocr_model, SUPPORTED_LANGUAGES


def generate_test_image(text: str, width: int = 800, height: int = 300) -> bytes:
    """Generate in-memory test image as JPEG bytes."""
    from PIL import Image, ImageDraw, ImageFont
    pil_img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(pil_img)
    font = None
    for font_name in ["gautami.ttf", "nirmala.ttf", "arial.ttf"]:
        try:
            font = ImageFont.truetype(font_name, 26)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    draw.text((25, 55), text, fill=(0, 0, 0), font=font)
    _, enc = cv2.imencode(".jpg", cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
    return enc.tobytes()


def run_kaggle_smoke_test():
    print("\n" + "=" * 75)
    print("KAGGLE GPU OCR BACKEND - MULTILINGUAL SMOKE TEST")
    print("=" * 75)

    client = app.test_client()

    # --------------------------------------------------------------------------
    # 1. Health / Status Check
    # --------------------------------------------------------------------------
    status_resp = client.get("/status")
    status_data = status_resp.get_json()
    print(f"Server Status:            {status_data.get('status')}")
    print(f"Reported Device:          {status_data.get('device')}")
    print(f"Supported Languages:      {status_data.get('supported_languages')}")
    print(f"Initially Cached Models:  {status_data.get('loaded_models')}")

    # --------------------------------------------------------------------------
    # 2. English Smoke Test (--lang en)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("TEST 1: English Model & OCR (--lang en)")
    print("-" * 75)

    t0 = time.perf_counter()
    en_model, en_load_err = get_gpu_ocr_model("en")
    en_load_time = (time.perf_counter() - t0) * 1000

    if en_model is not None:
        print(f"• Model Loading Status:   SUCCESS ({en_load_time:.1f} ms)")
    else:
        print(f"• Model Loading Status:   FAILED ({en_load_err})")

    en_img_bytes = generate_test_image("GOVERNMENT OF TELANGANA SALE DEED")
    en_req_data = {
        "image": (io.BytesIO(en_img_bytes), "page_1.jpg"),
        "page_number": "1",
        "lang": "en",
    }
    t_inf_0 = time.perf_counter()
    en_resp = client.post("/ocr", data=en_req_data, content_type="multipart/form-data")
    en_inf_time = (time.perf_counter() - t_inf_0) * 1000

    if en_resp.status_code == 200:
        en_json = en_resp.get_json()
        print(f"• Endpoint Status Code:   200 OK")
        print(f"• OCR Execution Status:   SUCCESS (Roundtrip: {en_inf_time:.1f} ms, Server OCR: {en_json.get('ocr_time_ms', 0):.1f} ms)")
        print(f"• Requested Language:     {en_json.get('requested_language')}")
        print(f"• OCR Model Language:     {en_json.get('ocr_model_language')}")
        print(f"• Model Status:           {en_json.get('model_status')}")
        print(f"• Needs Review:           {en_json.get('needs_review')}")
        print(f"• Lines Detected:         {len(en_json.get('lines', []))}")
        for idx, l in enumerate(en_json.get("lines", []), 1):
            print(f"    Line {idx}: \"{l['text']}\" (Conf: {l['confidence']}, BBox: {l['bbox']})")
    else:
        print(f"• OCR Execution Status:   FAILED (HTTP {en_resp.status_code}): {en_resp.data.decode('utf-8')}")

    # --------------------------------------------------------------------------
    # 3. Telugu Smoke Test (--lang te)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("TEST 2: Telugu Model & OCR (--lang te)")
    print("-" * 75)

    t0 = time.perf_counter()
    te_model, te_load_err = get_gpu_ocr_model("te")
    te_load_time = (time.perf_counter() - t0) * 1000

    if te_model is not None:
        print(f"• Model Loading Status:   SUCCESS ({te_load_time:.1f} ms)")
        te_img_bytes = generate_test_image("తెలంగాణ ప్రభుత్వం రిజిస్ట్రేషన్ శాఖ")
        te_req_data = {
            "image": (io.BytesIO(te_img_bytes), "page_1.jpg"),
            "page_number": "1",
            "lang": "te",
        }
        te_resp = client.post("/ocr", data=te_req_data, content_type="multipart/form-data")
        if te_resp.status_code == 200:
            te_json = te_resp.get_json()
            print(f"• OCR Execution Status:   SUCCESS (Server OCR: {te_json.get('ocr_time_ms', 0):.1f} ms)")
            print(f"• Requested Language:     {te_json.get('requested_language')}")
            print(f"• OCR Model Language:     {te_json.get('ocr_model_language')}")
            print(f"• Lines Detected:         {len(te_json.get('lines', []))}")
            for idx, l in enumerate(te_json.get("lines", []), 1):
                print(f"    Line {idx}: \"{l['text']}\" (Conf: {l['confidence']}, BBox: {l['bbox']})")
        else:
            print(f"• OCR Execution Status:   FAILED (HTTP {te_resp.status_code}): {te_resp.data.decode('utf-8')}")
    else:
        print(f"• Model Loading Status:   UNAVAILABLE LOCALLY")
        print(f"  Reason:                 {te_load_err}")
        print("• Verification of Fallback Policy:")
        te_img_bytes = generate_test_image("తెలంగాణ ప్రభుత్వం")
        te_req_data = {
            "image": (io.BytesIO(te_img_bytes), "page_1.jpg"),
            "page_number": "1",
            "lang": "te",
        }
        te_resp = client.post("/ocr", data=te_req_data, content_type="multipart/form-data")
        te_json = te_resp.get_json()
        print(f"  - HTTP Status:          {te_resp.status_code} (Expected 400)")
        print(f"  - Model Status:         {te_json.get('model_status')} (Expected UNAVAILABLE)")
        print(f"  - Needs Review:         {te_json.get('needs_review')} (Expected True)")
        print(f"  - Silently Used English?{te_json.get('ocr_model_language') == 'en'} (Expected False)")
        print(f"  - Structured Warnings:  {te_json.get('warnings')}")

    # --------------------------------------------------------------------------
    # 4. Critical Accuracy Disclaimer
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("ACCURACY DISCLAIMER (Per Requirements):")
    print("  Notice: These tests verify endpoint contract, model routing, caching,")
    print("  and absence of silent fallback. Clean synthetic images DO NOT establish")
    print("  or claim OCR recognition accuracy on noisy historical land deeds.")
    print("  Field accuracy must only be claimed after evaluation on real scanned deeds.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_kaggle_smoke_test()
