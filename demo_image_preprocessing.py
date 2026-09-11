"""
demo_image_preprocessing.py - Adaptive Image Preprocessing Demonstration for Scanned Land Documents.

Smart India Hackathon (SIH) Prototype: OneBhoomi Land Registry.

Demonstrates:
  1. Loading Page 2 from the reference Telangana Sale Deed PDF (or synthetic page if offline).
  2. Calculating real input quality metrics (contrast, blur, noise, brightness, skew).
  3. Generating safe preprocessing variants.
  4. Selecting the best variant deterministically.
  5. Displaying quality-before vs quality-after metrics.
  6. Saving inspection images safely under scratch/preprocessing_debug/.
  7. Clear disclosure: Single-document prototype demonstration.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

import image_preprocessing

PDF_PATH = r"C:\Users\meesa\Downloads\telangana_official_land_document.pdf"
DEBUG_OUTPUT_DIR = Path(__file__).resolve().parent / "scratch" / "preprocessing_debug"


def load_sample_page(page_idx: int = 2) -> np.ndarray:
    """Loads page image from reference PDF or generates realistic synthetic deed scan."""
    if os.path.exists(PDF_PATH):
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(PDF_PATH)
            if len(pdf) >= page_idx:
                pil_img = pdf[page_idx - 1].render(scale=2).to_pil()
                img_np = np.array(pil_img)
                bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if img_np.ndim == 3 else cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                return bgr
        except Exception as e:
            print(f"Notice: pdfium loading failed ({e}); falling back to synthetic scan.")

    # Synthetic realistic page (1600x2200 with text lines and slight skew)
    h, w = 2200, 1600
    img = np.full((h, w, 3), 235, dtype=np.uint8)

    # Add simulated typewritten lines
    for y in range(200, 2000, 60):
        cv2.putText(
            img,
            f"TELANGANA LAND REGISTRY SCHEDULE OF PROPERTY EXTENT 480 SQ YARDS SY NO 278, 281 LINE {y}",
            (120, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )

    # Add slight synthetic skew (1.8 degrees)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 1.8, 1.0)
    skewed = cv2.warpAffine(img, m, (w, h), borderValue=(240, 240, 240))
    return skewed


def run_demo():
    print("=" * 80)
    print("OneBhoomi: Adaptive Image Preprocessing for Scanned Land-Document OCR")
    print("Smart India Hackathon (SIH) Prototype Demonstration")
    print("=" * 80)
    print("""
[DEMONSTRATION DISCLOSURE]
* This demonstration evaluates scanned land-document pages from the reference
  Telangana Sale Deed.
* It illustrates quality-aware preprocessing, variant selection, and coordinate mapping.
* This is a single-document demonstration for SIH prototype evaluation.
* It does NOT represent a statistically validated accuracy improvement across all scanners.
* Original source documents and coordinate provenance are permanently preserved.
""")
    print("=" * 80)

    page_num = 2
    page_type = "property_schedule"

    print(f"\n[STEP 1] Loading document Page {page_num} ('{page_type}')...")
    orig_img = load_sample_page(page_idx=page_num)
    h, w = orig_img.shape[:2]
    print(f"  -> Page dimensions: {w} x {h} pixels")

    print("\n[STEP 2] Analyzing input scan quality...")
    quality_before = image_preprocessing.analyze_image_quality(orig_img)
    print(f"  -> Brightness            : {quality_before['brightness']:.1f} / 255")
    print(f"  -> Contrast              : {quality_before['contrast']:.1f}")
    print(f"  -> Blur Score (Laplacian): {quality_before['blur_score']:.1f}")
    print(f"  -> Noise Score           : {quality_before['noise_score']:.4f}")
    print(f"  -> Estimated Skew Angle  : {quality_before['estimated_skew_angle']:.2f}°")
    print(f"  -> Detected Flags        : {quality_before['quality_flags']}")
    print(f"  -> Original Quality Score: {quality_before['quality_score']:.2f}")

    print("\n[STEP 3] Generating safe preprocessing variants...")
    variants = image_preprocessing.build_preprocessing_variants(orig_img, page_type=page_type)
    for i, v in enumerate(variants, start=1):
        print(f"  [{i}] Variant: {v['name']:<24} | Score: {v['quality_score']:.2f} | Ops: {', '.join(v['operations'])}")

    print("\n[STEP 4] Deterministic best-variant selection...")
    selection = image_preprocessing.select_best_preprocessed_variant(variants, page_type=page_type)
    best_v = selection["variant_obj"]
    selected_img = best_v["image"]
    print(f"  -> Selected Variant      : {selection['selected_variant']}")
    print(f"  -> Selection Reason       : {selection['selection_reason']}")

    print("\n[STEP 5] Quality comparison (Before vs After)...")
    quality_after = image_preprocessing.analyze_image_quality(selected_img)
    print(f"  -> Contrast Before/After : {quality_before['contrast']:.1f}  -->  {quality_after['contrast']:.1f}")
    print(f"  -> Blur Score Before/After: {quality_before['blur_score']:.1f}  -->  {quality_after['blur_score']:.1f}")
    print(f"  -> Quality Score         : {quality_before['quality_score']:.2f}  -->  {quality_after['quality_score']:.2f}")

    print(f"\n[STEP 6] Saving debug inspection images to '{DEBUG_OUTPUT_DIR}'...")
    p1 = image_preprocessing.save_preprocessing_debug_image(orig_img, page_num, "01_original", DEBUG_OUTPUT_DIR)
    p2 = image_preprocessing.save_preprocessing_debug_image(selected_img, page_num, f"02_selected_{best_v['name']}", DEBUG_OUTPUT_DIR)
    print(f"  -> Saved: {p1}")
    print(f"  -> Saved: {p2}")

    # Coordinate mapping demonstration
    print("\n[STEP 7] Verifying coordinate mapping back to original page space...")
    scale_used = best_v.get("scale", 1.0)
    simulated_ocr_bbox = [int(150 * scale_used), int(200 * scale_used), int(600 * scale_used), int(240 * scale_used)]
    mapped_bbox = image_preprocessing.map_coordinates_to_original(simulated_ocr_bbox, scale=scale_used)
    print(f"  -> Preprocessed space bbox (scale {scale_used}x): {simulated_ocr_bbox}")
    print(f"  -> Restored original page bbox           : {mapped_bbox}")
    assert mapped_bbox == [150, 200, 600, 240], "Coordinate mapping failed!"
    print("  -> Coordinate restoration verified 100% accurate.")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"Page: {page_num}")
    print(f"Original quality score: {quality_before['quality_score']:.2f}")
    print(f"Selected variant: {best_v['name']}")
    print(f"Processed quality score: {quality_after['quality_score']:.2f}")
    print(f"Operations: {', '.join(best_v['operations'])}")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
