"""
test_handwriting_recognition.py

CLI tool for testing handwriting recognition against an image or crop.
Adheres to strict safeguards:
1. Clearly labels detection method ("heuristic" vs "model").
2. Does not call low-confidence printed lines handwriting merely because OCR is low.
3. Separates handwriting from printed text, signatures, stamps/seals, and damaged regions.
4. Never uses printed PaddleOCR as handwriting recognition.
5. If no dedicated HTR model is installed, returns MODEL_UNAVAILABLE with warning.
6. Does not install large dependencies automatically.
7. Does not claim handwriting recognition is implemented unless a real HTR model runs.
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np

from land_document_extractor import (
    OCRLine,
    detect_handwriting_regions,
    get_handwriting_recognizer,
    build_handwriting_metadata,
)


def run_handwriting_test(
    image_path: str,
    backend: str = "auto",
    model_path: str | None = None,
    remote: bool = False,
    ocr_url: str | None = None,
) -> dict:
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}", file=sys.stderr)
        sys.exit(1)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image with OpenCV from {image_path}", file=sys.stderr)
        sys.exit(1)

    h, w = img.shape[:2]

    # Run PaddleOCR only for printed text layout reference / detection coordinates if available locally
    lines = []
    if not remote:
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang="en")
            results = ocr.ocr(image_path, cls=True)
            if results and results[0]:
                for item in results[0]:
                    coords = item[0]
                    text_conf = item[1]
                    xs = [pt[0] for pt in coords]
                    ys = [pt[1] for pt in coords]
                    lines.append(
                        OCRLine(
                            text=text_conf[0],
                            score=float(text_conf[1]),
                            x_min=int(min(xs)),
                            y_min=int(min(ys)),
                            x_max=int(max(xs)),
                            y_max=int(max(ys)),
                            page_num=1,
                        )
                    )
        except Exception:
            lines = []

    # If it's a tight crop of handwriting, treat the entire crop as a candidate region
    if not lines and (h < 500 or w < 800):
        regions = [{
            "page_number": 1,
            "region_type": "possible_handwriting",
            "detection_method": "heuristic",
            "bounding_box": [0, 0, w, h],
            "detection_confidence": 0.70,
            "recognition_status": "NOT_RUN",
            "recognized_text": None,
            "recognition_confidence": 0.0,
            "source_model": None,
            "needs_review": True,
            "warning": None,
        }]
    else:
        regions = detect_handwriting_regions(lines, image_height=h, image_width=w, page_num=1, image=img)

    # Check remote connectivity if remote requested
    request_reached_kaggle = False
    remote_server_status = None
    resolved_backend = "remote" if remote else backend
    effective_ocr_url = ocr_url if (remote or ocr_url) else None

    if effective_ocr_url:
        import requests
        try:
            st = requests.get(f"{effective_ocr_url.rstrip('/')}/status", timeout=6)
            if st.status_code == 200:
                request_reached_kaggle = True
                remote_server_status = st.json()
        except Exception as ex:
            remote_server_status = {"error": str(ex)}

    # Initialize pluggable recognizer
    recognizer = get_handwriting_recognizer(backend=resolved_backend, model_path=model_path, ocr_url=effective_ocr_url)

    # Process regions through HTR recognizer
    processed_regions = []
    for reg in regions:
        processed_r = recognizer.recognize_region(reg, image=img)
        processed_regions.append(processed_r)

    metadata = build_handwriting_metadata(processed_regions)

    actual_inference_ran = any(
        r.get("recognition_status") == "RECOGNIZED" and r.get("recognized_text") is not None
        for r in processed_regions
    )

    report = {
        "image_path": image_path,
        "image_dimensions": {"width": w, "height": h},
        "handwriting_detected": metadata["detected"],
        "htr_backend_requested": resolved_backend,
        "remote_mode": remote,
        "ocr_url": effective_ocr_url,
        "request_reached_kaggle": request_reached_kaggle,
        "remote_server_status": remote_server_status,
        "htr_model_status": recognizer.model_status,
        "htr_source_model": recognizer.model_name,
        "actual_htr_inference_ran": actual_inference_ran,
        "total_regions_detected": len(processed_regions),
        "recognized_region_count": metadata["recognized_region_count"],
        "unrecognized_region_count": metadata["unrecognized_region_count"],
        "manual_review_required": metadata["manual_review_required"],
        "regions": processed_regions,
        "safeguards_summary": {
            "never_fall_back_to_printed_ocr": True,
            "detection_method_labeled": all(r.get("detection_method") in ("heuristic", "model") for r in processed_regions),
            "no_overwrite_structured_fields": True,
            "needs_review_always_true": all(r.get("needs_review") is True for r in processed_regions),
        }
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Test handwriting detection and recognition on an image/crop.")
    parser.add_argument("--image", required=True, help="Path to input image or document crop")
    parser.add_argument("--backend", default="auto", choices=["auto", "trocr", "paddle_htr", "remote"], help="HTR backend to test")
    parser.add_argument("--remote", action="store_true", help="Send handwriting crop to remote Kaggle GPU backend")
    parser.add_argument("--ocr-url", default=None, help="Remote Kaggle GPU server URL")
    parser.add_argument("--model-path", default=None, help="Optional path to local HTR model weights")
    parser.add_argument("--json", action="store_true", help="Output raw JSON report")

    args = parser.parse_args()

    # Load default ocr-url from colab_url.txt if --remote passed without --ocr-url
    ocr_url = args.ocr_url
    if (args.remote or args.backend == "remote") and not ocr_url:
        colab_file = os.path.join(os.path.dirname(__file__), "colab_url.txt")
        if os.path.exists(colab_file):
            with open(colab_file, "r") as f:
                ocr_url = f.read().strip()

    report = run_handwriting_test(
        args.image,
        backend=args.backend,
        model_path=args.model_path,
        remote=args.remote,
        ocr_url=ocr_url,
    )

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("=" * 60)
    print("HANDWRITING DETECTION & RECOGNITION SMOKE TEST")
    print("=" * 60)
    print(f"Image:                 {report['image_path']} ({report['image_dimensions']['width']}x{report['image_dimensions']['height']})")
    print(f"Remote Mode:           {report['remote_mode']}")
    print(f"Remote URL:            {report['ocr_url']}")
    print(f"Reached Kaggle Server: {report['request_reached_kaggle']}")
    print(f"Handwriting Detected:  {report['handwriting_detected']}")
    print(f"HTR Model Status:      {report['htr_model_status']}")
    print(f"HTR Source Model:      {report['htr_source_model']}")
    print(f"Actual HTR Inference:  {report['actual_htr_inference_ran']}")
    print(f"Manual Review Needed:  {report['manual_review_required']}")
    print(f"Regions Detected:      {report['total_regions_detected']}")
    print(f"Regions Recognized:    {report['recognized_region_count']}")
    print(f"Regions Unrecognized:  {report['unrecognized_region_count']}")
    print("-" * 60)

    for i, r in enumerate(report["regions"], 1):
        print(f"Region #{i}:")
        print(f"  Page:                 {r.get('page_number')}")
        print(f"  Bounding Box:         {r.get('bounding_box')}")
        print(f"  Line Crop Boxes:      {r.get('line_crop_boxes')}")
        print(f"  Detection Method:     {r.get('detection_method')}")
        print(f"  Detection Confidence: {r.get('detection_confidence')}")
        print(f"  Recognition Status:   {r.get('recognition_status')}")
        print(f"  Recognized Text:      {r.get('recognized_text')}")
        print(f"  Confidence:           {r.get('recognition_confidence')}")
        print(f"  Source Model:         {r.get('source_model')}")
        print(f"  Needs Review:         {r.get('needs_review')}")
        print(f"  Warning:              {r.get('warning')}")

    print("=" * 60)
    if not report["actual_htr_inference_ran"]:
        print("NOTE: No trained HTR model was active or inference did not return text.")
        print("Printed PaddleOCR was NOT used as a handwriting recognition substitute.")
        print("Region results remain REVIEW CANDIDATES only.")
    else:
        print("SUCCESS: Actual HTR inference ran on the handwritten crop.")
        print("Safeguard notice: Results remain review candidates and do NOT overwrite legal fields.")


if __name__ == "__main__":
    main()
