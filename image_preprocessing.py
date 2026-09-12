"""
image_preprocessing.py - Adaptive Image Preprocessing Pipeline for Scanned Land Documents.

Smart India Hackathon (SIH) Prototype: OneBhoomi Land Registry.

Provides a quality-aware, safe image-preprocessing pipeline before PaddleOCR runs:
  Original PDF/image
  → Page rendering
  → Quality analysis
  → Safe preprocessing variants
  → Best variant selection
  → PaddleOCR
  → Coordinate mapping to original page space
  → Structured extraction

Key Principles:
1. Non-destructive: Original page image is ALWAYS preserved for evidence and review.
2. Coordinate integrity: If an image is upscaled or transformed, bounding boxes
   are accurately mapped back to the original page coordinate space.
3. Safe processing: Avoids over-enhancing or destroying dark stamp papers,
   faint handwriting, signatures, or registration plan boundary lines.
4. Deterministic: Deterministic variant generation and quality scoring.
5. No heavy dependencies: Built entirely with OpenCV, NumPy, and Pillow.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ignored directory for local debugging and hackathon demonstrations
DEBUG_PREPROCESSING_DIR = Path(__file__).resolve().parent / "scratch" / "preprocessing_debug"


# ---------------------------------------------------------------------------
# 1. Quality Analysis
# ---------------------------------------------------------------------------

def analyze_image_quality(image: np.ndarray) -> Dict[str, Any]:
    """
    Computes real, calculated image-quality metrics for a scanned land document page.

    Calculates:
      - width, height
      - estimated_dpi (heuristic based on A4 dimensions)
      - brightness (mean grayscale intensity)
      - contrast (standard deviation of pixel intensity)
      - blur_score (Laplacian variance; higher is sharper)
      - noise_score (difference between image and median-filtered image)
      - dark_pixel_ratio (ratio of dark pixels < 60)
      - white_background_ratio (ratio of light background pixels > 200)
      - estimated_skew_angle (estimated skew angle in degrees)
      - is_rotated (heuristic check for landscape vs portrait)
      - is_faded (low contrast / high brightness indicator)
      - has_dark_stamp_region (density of dark pixels in upper stamp area)
      - quality_score (composite score bounded in [0.0, 1.0])
      - quality_flags (list of detected issue tags)
    """
    if image is None or image.size == 0:
        return {
            "width": 0,
            "height": 0,
            "quality_score": 0.0,
            "quality_flags": ["empty_image"],
        }

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    # Fundamental statistics
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Blur score via Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # Noise score: deviation from a 3x3 median-smoothed baseline
    median_filtered = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, median_filtered)
    noise_score = float(np.mean(diff) / 255.0)

    # Pixel distributions
    dark_pixel_ratio = float(np.mean(gray < 60))
    white_background_ratio = float(np.mean(gray > 200))

    # Estimated DPI (assuming standard legal/A4 height ~ 11.7 in or width ~ 8.3 in)
    est_dpi = round(float(max(w / 8.27, h / 11.69))) if (w > 200 and h > 200) else None

    # Skew angle estimation
    skew_angle = estimate_skew_angle(gray)

    # Heuristic checks
    is_rotated = bool(w > h * 1.25)
    is_faded = bool(contrast < 35.0 or (brightness > 225.0 and contrast < 45.0))

    # Stamp region detection (top 35% of page often contains government stamp paper)
    top_region_h = max(10, int(h * 0.35))
    top_region = gray[:top_region_h, :]
    top_dark_ratio = float(np.mean(top_region < 80))
    has_dark_stamp_region = bool(top_dark_ratio > 0.18)

    # Build quality flags
    quality_flags = []
    if abs(skew_angle) >= 0.5:
        quality_flags.append("slightly_skewed")
    if contrast < 40.0:
        quality_flags.append("low_contrast")
    if is_faded:
        quality_flags.append("faded_text")
    if blur_score < 60.0:
        quality_flags.append("blurry")
    if has_dark_stamp_region:
        quality_flags.append("dark_stamp_region")
    if noise_score > 0.04:
        quality_flags.append("noisy")
    if is_rotated:
        quality_flags.append("landscape_orientation")

    # Composite quality score (0.0 to 1.0)
    contrast_norm = min(1.0, max(0.0, contrast / 70.0))
    blur_norm = min(1.0, max(0.0, blur_score / 200.0))
    noise_norm = max(0.0, 1.0 - (noise_score * 6.0))
    skew_norm = max(0.0, 1.0 - (abs(skew_angle) / 10.0))

    composite_score = round(
        float(0.35 * contrast_norm + 0.35 * blur_norm + 0.15 * noise_norm + 0.15 * skew_norm),
        4,
    )

    return {
        "width": int(w),
        "height": int(h),
        "estimated_dpi": est_dpi,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2),
        "noise_score": round(noise_score, 4),
        "dark_pixel_ratio": round(dark_pixel_ratio, 4),
        "white_background_ratio": round(white_background_ratio, 4),
        "estimated_skew_angle": round(skew_angle, 2),
        "is_rotated": is_rotated,
        "is_faded": is_faded,
        "has_dark_stamp_region": has_dark_stamp_region,
        "quality_score": composite_score,
        "quality_flags": quality_flags,
    }


# ---------------------------------------------------------------------------
# 2. Skew Estimation and Correction
# ---------------------------------------------------------------------------

def estimate_skew_angle(image: np.ndarray) -> float:
    """
    Estimates the text skew angle using horizontal text-line morphology and Hough lines.
    Returns angle in degrees (negative = counterclockwise, positive = clockwise).
    Only returns angles within a realistic document skew range [-15.0, 15.0].
    """
    if image is None or image.size == 0:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    h, w = gray.shape[:2]

    # Downscale for fast and robust skew estimation if image is large
    scale_factor = 1.0
    if max(h, w) > 1200:
        scale_factor = 1200.0 / max(h, w)
        small = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    else:
        small = gray

    # Binarize inverted so text is white
    _, thresh = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to merge letters into continuous text lines
    kernel_w = max(5, int(small.shape[1] * 0.025))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 2))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Hough Lines on morphological text strips
    lines = cv2.HoughLinesP(
        dilated,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(small.shape[1] * 0.15),
        maxLineGap=20,
    )

    if lines is not None and len(lines) > 0:
        angles = []
        for line in lines:
            coords = line.flatten() if hasattr(line, "flatten") else line
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            deg = np.degrees(np.arctan2(dy, dx))
            # Keep only near-horizontal lines (|angle| <= 15 degrees)
            if abs(deg) <= 15.0:
                angles.append(deg)

        if len(angles) >= 3:
            median_angle = float(np.median(angles))
            return round(median_angle, 2)

    # Fallback to minimum area bounding box on text pixel clusters
    pts = np.column_stack(np.where(dilated > 0))
    if len(pts) > 200:
        rect = cv2.minAreaRect(pts)
        angle = rect[-1]
        if angle < -45.0:
            angle = -(90.0 + angle)
        elif angle > 45.0:
            angle = -(angle - 90.0)
        else:
            angle = -angle
        if abs(angle) <= 15.0:
            return round(float(angle), 2)

    return 0.0


def correct_skew(image: np.ndarray, angle: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Rotates the image to correct skew.
    Only rotates if skew is meaningful (>= 0.5 degrees and <= 15.0 degrees).
    """
    if image is None or image.size == 0:
        return image, 0.0

    if angle is None:
        angle = estimate_skew_angle(image)

    # Safe guard: do not rotate for tiny or extreme angles
    if abs(angle) < 0.5 or abs(angle) > 15.0:
        return image.copy(), 0.0

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rot_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


# ---------------------------------------------------------------------------
# 3. Orientation Detection & Safe Border Removal
# ---------------------------------------------------------------------------

def correct_orientation(image: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Detects obvious 90/180/270-degree rotation if confident.
    If orientation cannot be confidently confirmed, preserves original image.
    """
    if image is None or image.size == 0:
        return image, 0

    h, w = image.shape[:2]
    # Land document standard is portrait. If width > 1.35 * height, it is likely landscape.
    if w > h * 1.35:
        # Rotate 90 degrees clockwise as standard landscape-to-portrait orientation
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return rotated, 90

    return image.copy(), 0


def remove_borders(image: np.ndarray) -> np.ndarray:
    """
    Detects and cleans only obvious dark scan borders on the extreme image edges
    without cropping into page numbers, stamps, signatures, or margins.
    """
    if image is None or image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    h, w = gray.shape[:2]

    # Inspect outer 2% margin on each side for pure black scanner border strip
    max_strip_y = max(2, int(h * 0.02))
    max_strip_x = max(2, int(w * 0.02))

    top_crop = 0
    bottom_crop = h
    left_crop = 0
    right_crop = w

    # Top border strip
    if np.mean(gray[:max_strip_y, :]) < 30:
        top_crop = max_strip_y
    # Bottom border strip
    if np.mean(gray[-max_strip_y:, :]) < 30:
        bottom_crop = h - max_strip_y
    # Left border strip
    if np.mean(gray[:, :max_strip_x]) < 30:
        left_crop = max_strip_x
    # Right border strip
    if np.mean(gray[:, -max_strip_x:]) < 30:
        right_crop = w - max_strip_x

    # Safe crop: only apply if bounded
    if (bottom_crop > top_crop + 100) and (right_crop > left_crop + 100):
        return image[top_crop:bottom_crop, left_crop:right_crop].copy()

    return image.copy()


# ---------------------------------------------------------------------------
# 4. Core Preprocessing Operations
# ---------------------------------------------------------------------------

def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) safely.
    Uses conservative clip_limit=2.0 to avoid over-enhancing dark stamp backgrounds.
    """
    if image is None or image.size == 0:
        return image

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    if image.ndim == 3:
        # Work in LAB color space to equalize Luminance without distorting color channels
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(image)


def remove_noise(image: np.ndarray) -> np.ndarray:
    """
    Lightweight bilateral filtering or median blur that eliminates scanning grain
    while preserving thin strokes, numbers, and signatures.
    """
    if image is None or image.size == 0:
        return image

    if image.ndim == 3:
        return cv2.bilateralFilter(image, d=5, sigmaColor=35, sigmaSpace=35)
    else:
        return cv2.bilateralFilter(image, d=5, sigmaColor=35, sigmaSpace=35)


def adaptive_binarize(image: np.ndarray) -> np.ndarray:
    """
    Applies Adaptive Gaussian Thresholding to produce a high-contrast binarized variant.
    """
    if image is None or image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    binarized = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=8,
    )
    if image.ndim == 3:
        return cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
    return binarized


def sharpen_image(image: np.ndarray) -> np.ndarray:
    """
    Applies controlled unsharp masking to enhance character edges without halo noise.
    """
    if image is None or image.size == 0:
        return image

    # Unsharp mask formula: sharpened = image + (image - blurred) * factor
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(image, 1.4, blurred, -0.4, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def upscale_image(image: np.ndarray, scale: float = 1.5) -> np.ndarray:
    """
    Upscales low-resolution pages and small text regions using text-optimal INTER_CUBIC interpolation.
    """
    if image is None or image.size == 0 or scale <= 1.0:
        return image.copy()

    return cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# 5. Preprocessing Variants Generation & Safe Selection
# ---------------------------------------------------------------------------

def score_preprocessed_variant(
    image: np.ndarray,
    variant_name: str,
    page_type: Optional[str] = None,
) -> float:
    """
    Computes a deterministic quality score for a preprocessed image variant.
    Favors high contrast and sharpness while penalizing blown-out black or white pixels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    contrast = float(np.std(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Stroke density
    dark_ratio = float(np.mean(gray < 100))
    white_ratio = float(np.mean(gray > 220))

    score = 0.5

    # Contrast reward
    score += min(0.25, contrast / 160.0)
    # Sharpness reward
    score += min(0.25, blur / 400.0)

    # Penalize excessive dark regions (e.g. over-binarized stamp blocks)
    if dark_ratio > 0.40:
        score -= 0.20
    # Penalize blank or completely washed-out images
    if white_ratio > 0.98:
        score -= 0.30

    # Page-type specific safety preferences
    p_type = (page_type or "unknown").lower()
    if p_type in ("stamp_metadata", "registration_plan"):
        # Thresholding destroys stamp seals and CAD boundary lines
        if "threshold" in variant_name.lower():
            score -= 0.35
        if "clahe_denoised" in variant_name.lower():
            score += 0.10
    elif p_type in ("deed_text", "property_schedule"):
        if "clahe_denoised" in variant_name.lower():
            score += 0.08

    return round(float(np.clip(score, 0.1, 0.99)), 4)


def build_preprocessing_variants(
    image: np.ndarray,
    page_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generates a curated, deterministic set of safe preprocessing variants:
      1. original_grayscale: Safe grayscale baseline
      2. clahe_enhanced: Contrast enhanced
      3. clahe_denoised: Denoised + CLAHE
      4. sharpened_clahe: Sharpened + CLAHE
      5. adaptive_threshold: Adaptive thresholded
      6. clahe_denoised_upscaled: 1.5x upscaled (if resolution is moderate)
    """
    if image is None or image.size == 0:
        return []

    # Ensure baseline is grayscale converted to 3-channel for PaddleOCR compatibility
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape[:2]

    # Deskew first if angle is meaningful
    deskewed_gray, skew_angle = correct_skew(gray)
    deskewed_bgr = cv2.cvtColor(deskewed_gray, cv2.COLOR_GRAY2BGR)
    deskew_applied = bool(abs(skew_angle) >= 0.5)

    variants: List[Dict[str, Any]] = []

    # Variant 1: Original / Deskewed Grayscale
    v1_ops = ["grayscale"]
    if deskew_applied:
        v1_ops.append("deskew")
    v1 = {
        "name": "original_grayscale",
        "image": deskewed_bgr,
        "operations": v1_ops,
        "scale": 1.0,
        "deskew_angle": skew_angle,
        "quality_score": score_preprocessed_variant(deskewed_bgr, "original_grayscale", page_type),
    }
    variants.append(v1)

    # Variant 2: CLAHE Enhanced
    enhanced = enhance_contrast(deskewed_gray, clip_limit=2.0)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    v2_ops = ["grayscale", "clahe"]
    if deskew_applied:
        v2_ops.append("deskew")
    v2 = {
        "name": "clahe_enhanced",
        "image": enhanced_bgr,
        "operations": v2_ops,
        "scale": 1.0,
        "deskew_angle": skew_angle,
        "quality_score": score_preprocessed_variant(enhanced_bgr, "clahe_enhanced", page_type),
    }
    variants.append(v2)

    # Variant 3: Denoised + CLAHE Enhanced
    denoised = remove_noise(enhanced_bgr)
    v3_ops = ["grayscale", "clahe", "denoise"]
    if deskew_applied:
        v3_ops.append("deskew")
    v3 = {
        "name": "clahe_denoised",
        "image": denoised,
        "operations": v3_ops,
        "scale": 1.0,
        "deskew_angle": skew_angle,
        "quality_score": score_preprocessed_variant(denoised, "clahe_denoised", page_type),
    }
    variants.append(v3)

    # Variant 4: Sharpened + CLAHE
    sharpened = sharpen_image(enhanced_bgr)
    v4_ops = ["grayscale", "clahe", "sharpen"]
    if deskew_applied:
        v4_ops.append("deskew")
    v4 = {
        "name": "sharpened_clahe",
        "image": sharpened,
        "operations": v4_ops,
        "scale": 1.0,
        "deskew_angle": skew_angle,
        "quality_score": score_preprocessed_variant(sharpened, "sharpened_clahe", page_type),
    }
    variants.append(v4)

    # Variant 5: Adaptive Threshold
    thresh = adaptive_binarize(deskewed_gray)
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) if thresh.ndim == 2 else thresh
    v5_ops = ["grayscale", "adaptive_threshold"]
    if deskew_applied:
        v5_ops.append("deskew")
    v5 = {
        "name": "adaptive_threshold",
        "image": thresh_bgr,
        "operations": v5_ops,
        "scale": 1.0,
        "deskew_angle": skew_angle,
        "quality_score": score_preprocessed_variant(thresh_bgr, "adaptive_threshold", page_type),
    }
    variants.append(v5)

    # Variant 6: Upscaled (1.5x) Denoised CLAHE (for small/medium resolution scans)
    if max(h, w) <= 2400:
        upscaled = upscale_image(denoised, scale=1.5)
        v6_ops = ["grayscale", "clahe", "denoise", "upscale"]
        if deskew_applied:
            v6_ops.append("deskew")
        v6 = {
            "name": "clahe_denoised_upscaled",
            "image": upscaled,
            "operations": v6_ops,
            "scale": 1.5,
            "deskew_angle": skew_angle,
            "quality_score": score_preprocessed_variant(upscaled, "clahe_denoised_upscaled", page_type),
        }
        variants.append(v6)

    return variants


def select_best_preprocessed_variant(
    variants: List[Dict[str, Any]],
    page_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Selects the best preprocessing variant deterministically.
    Prefers safe grayscale variants (clahe_denoised) when scores are within 0.05
    to avoid destructive binarization of sensitive legal features.
    """
    if not variants:
        raise ValueError("No preprocessing variants provided for selection")

    sorted_variants = sorted(variants, key=lambda v: v.get("quality_score", 0.0), reverse=True)
    top_variant = sorted_variants[0]

    # Safety policy: If adaptive_threshold is highest by only a small margin (< 0.06),
    # prefer the safe enhanced grayscale variant (clahe_denoised or clahe_enhanced).
    if "threshold" in top_variant["name"]:
        for v in sorted_variants[1:]:
            if "clahe_denoised" in v["name"] and (top_variant["quality_score"] - v["quality_score"] < 0.06):
                top_variant = v
                break

    candidates_summary = [
        {"name": v["name"], "score": v.get("quality_score", 0.0)}
        for v in sorted_variants
    ]

    return {
        "selected_variant": top_variant["name"],
        "selection_reason": f"best balanced quality score ({top_variant.get('quality_score', 0.0):.2f}) for page type '{page_type or 'general'}'",
        "variant_obj": top_variant,
        "candidate_variants": candidates_summary,
    }


# ---------------------------------------------------------------------------
# 6. OCR Preprocessing Master Function & Coordinate Mapping
# ---------------------------------------------------------------------------

def preprocess_for_ocr(
    image: np.ndarray,
    page_number: int = 1,
    page_type: Optional[str] = None,
    save_debug: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Preprocesses a scanned land-document page for optimal PaddleOCR inference.

    Returns:
        (
            selected_ocr_image,     # Preprocessed 3-channel image ready for OCR
            preprocessing_metadata  # Quality and transformation tracking dictionary
        )
    """
    if image is None or image.size == 0:
        return image, {"page_number": page_number, "scale": 1.0, "operations": []}

    # 1. Analyze initial quality
    quality_before = analyze_image_quality(image)

    # 2. Build candidate variants
    variants = build_preprocessing_variants(image, page_type=page_type)

    # 3. Select best variant safely
    selection = select_best_preprocessed_variant(variants, page_type=page_type)
    best_variant = selection["variant_obj"]
    selected_image = best_variant["image"]

    # 4. Analyze quality after processing
    quality_after = analyze_image_quality(selected_image)

    # 5. Assemble metadata
    metadata = {
        "page_number": page_number,
        "page_type": page_type or "unknown",
        "selected_variant": best_variant["name"],
        "operations": best_variant["operations"],
        "scale": best_variant.get("scale", 1.0),
        "deskew_applied": bool(abs(best_variant.get("deskew_angle", 0.0)) >= 0.5),
        "skew_angle_degrees": best_variant.get("deskew_angle", 0.0),
        "quality_before": quality_before,
        "quality_after": quality_after,
        "selection_reason": selection["selection_reason"],
        "candidate_variants": selection["candidate_variants"],
    }

    # Record in active runtime state for telemetry & dashboard
    record_runtime_preprocessing(metadata)

    # Optional hackathon debug image export
    if save_debug:
        save_preprocessing_debug_image(image, page_number, "01_original")
        save_preprocessing_debug_image(selected_image, page_number, f"02_selected_{best_variant['name']}")
        for v in variants:
            if "threshold" in v["name"]:
                save_preprocessing_debug_image(v["image"], page_number, "03_adaptive_threshold")
                break

    return selected_image, metadata


# ---------------------------------------------------------------------------
# 6. Runtime Telemetry State & Storage
# ---------------------------------------------------------------------------

_LAST_RUNTIME_PREPROCESSING: Optional[Dict[str, Any]] = None
_RUNTIME_PAGE_PREPROCESSING: Dict[int, Dict[str, Any]] = {}
RUNTIME_TELEMETRY_PATH = DEBUG_PREPROCESSING_DIR / "runtime_telemetry.json"


def record_runtime_preprocessing(meta: Dict[str, Any]) -> None:
    """Records the latest preprocessed page metadata in memory and safely caches to scratch/."""
    global _LAST_RUNTIME_PREPROCESSING, _RUNTIME_PAGE_PREPROCESSING
    import json
    _LAST_RUNTIME_PREPROCESSING = meta
    pg = int(meta.get("page_number", 1))
    _RUNTIME_PAGE_PREPROCESSING[pg] = meta

    try:
        DEBUG_PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
        # Create a JSON-serializable snapshot (strip numpy types)
        clean_snapshot = {
            "latest": meta,
            "pages": list(_RUNTIME_PAGE_PREPROCESSING.values()),
        }
        RUNTIME_TELEMETRY_PATH.write_text(json.dumps(clean_snapshot, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def get_latest_runtime_preprocessing() -> Optional[Dict[str, Any]]:
    """Retrieves the most recent page preprocessing metadata from memory or disk cache."""
    global _LAST_RUNTIME_PREPROCESSING
    if _LAST_RUNTIME_PREPROCESSING is not None:
        return _LAST_RUNTIME_PREPROCESSING

    if RUNTIME_TELEMETRY_PATH.exists():
        try:
            import json
            data = json.loads(RUNTIME_TELEMETRY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "latest" in data:
                _LAST_RUNTIME_PREPROCESSING = data["latest"]
                return _LAST_RUNTIME_PREPROCESSING
        except Exception:
            pass
    return None


def get_all_runtime_preprocessing() -> List[Dict[str, Any]]:
    """Retrieves all recorded page preprocessing sessions in the current runtime."""
    global _RUNTIME_PAGE_PREPROCESSING
    if _RUNTIME_PAGE_PREPROCESSING:
        return list(_RUNTIME_PAGE_PREPROCESSING.values())

    if RUNTIME_TELEMETRY_PATH.exists():
        try:
            import json
            data = json.loads(RUNTIME_TELEMETRY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "pages" in data:
                return data["pages"]
        except Exception:
            pass
    return []


def map_coordinates_to_original(
    bbox: List[int | float],
    scale: float = 1.0,
    offset_x: int = 0,
    offset_y: int = 0,
) -> List[int]:
    """
    Maps bounding-box coordinates [x_min, y_min, x_max, y_max] from preprocessed/upscaled
    image space back into original source page coordinate space.
    """
    if scale <= 0:
        scale = 1.0

    x_min, y_min, x_max, y_max = bbox
    orig_x_min = int(round((x_min - offset_x) / scale))
    orig_y_min = int(round((y_min - offset_y) / scale))
    orig_x_max = int(round((x_max - offset_x) / scale))
    orig_y_max = int(round((y_max - offset_y) / scale))

    return [orig_x_min, orig_y_min, orig_x_max, orig_y_max]


def save_preprocessing_debug_image(
    image: np.ndarray,
    page_number: int,
    label: str,
    output_dir: Optional[Path | str] = None,
) -> str:
    """
    Safely saves inspection/debug image under scratch/preprocessing_debug/.
    """
    out_dir = Path(output_dir) if output_dir else DEBUG_PREPROCESSING_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"page_{page_number}_{label}.png"
    cv2.imwrite(str(out_file), image)
    return str(out_file)
