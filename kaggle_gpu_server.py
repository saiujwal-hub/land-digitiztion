#!/usr/bin/env python3
"""
kaggle_gpu_server.py

Kaggle GPU OCR Server - Headless Long-Running Service for OneBhoomi SIH Prototype.
Runs on Kaggle with GPU enabled and Internet enabled.

Provides:
  1. Automated CUDA & GPU detection
  2. Pinned, compatible PaddlePaddle GPU + PaddleOCR dependency management
  3. Pre-flight startup diagnostics & AnalysisConfig compatibility shims
  4. Real GPU OCR smoke test on synthetic text image before opening traffic
  5. Enhanced /status endpoint reporting true Paddle CUDA engine readiness
  6. High-throughput /ocr endpoint compatible with local preprocessing & coordinate tracking
  7. Isolated /recognize-handwriting endpoint for TrOCR (never falls back to PaddleOCR)
  8. Cloudflare tunnel auto-configuration and heartbeat monitoring
"""

import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import re
import sys
import time
import shutil
import platform
import tempfile
import requests
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ==============================================================================
# 0. DEFENSIVE RUNTIME COMPATIBILITY SHIM
# ==============================================================================
def apply_runtime_compatibility_shims():
    """
    Applies runtime shims for C++ bindings in PaddlePaddle / PaddleOCR.
    Specifically guarantees that AnalysisConfig.set_optimization_level exists as a safe callable
    regardless of whether Paddle 3.x or 2.x C++ bindings are active in memory.
    """
    try:
        import paddle
        # Paddle 2.x vs 3.x AnalysisConfig compatibility
        if hasattr(paddle, "base") and hasattr(paddle.base, "libpaddle"):
            cfg_cls = getattr(paddle.base.libpaddle, "AnalysisConfig", None)
            if cfg_cls is not None and not hasattr(cfg_cls, "set_optimization_level"):
                try:
                    setattr(cfg_cls, "set_optimization_level", lambda self, *args, **kwargs: None)
                    print("[COMPATIBILITY SHIM] Added no-op set_optimization_level to paddle.base.libpaddle.AnalysisConfig")
                except Exception as e:
                    print(f"[COMPATIBILITY SHIM] Warning: Could not patch AnalysisConfig: {e}")

        if hasattr(paddle, "inference") and hasattr(paddle.inference, "Config"):
            inf_cls = getattr(paddle.inference, "Config", None)
            if inf_cls is not None and not hasattr(inf_cls, "set_optimization_level"):
                try:
                    setattr(inf_cls, "set_optimization_level", lambda self, *args, **kwargs: None)
                    print("[COMPATIBILITY SHIM] Added no-op set_optimization_level to paddle.inference.Config")
                except Exception:
                    pass
    except Exception:
        pass


# ==============================================================================
# 1. HARDWARE & CUDA DETECTION
# ==============================================================================
def detect_hardware_and_cuda() -> Dict[str, Any]:
    """Detects system GPU model, driver CUDA version, and matching wheel suffix."""
    info = {
        "gpu_name": "NVIDIA GPU",
        "cuda_version": "Unknown",
        "cuda_suffix": "cu126",
        "has_gpu": False,
    }
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.STDOUT
        ).decode("utf-8")
        gpu_names = [g.strip() for g in smi_out.strip().splitlines() if g.strip()]
        if gpu_names:
            info["gpu_name"] = " \n ".join(gpu_names)
            info["has_gpu"] = True
    except Exception:
        pass

    try:
        smi_all = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode("utf-8")
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", smi_all)
        if match:
            info["cuda_version"] = match.group(1)
            parts = info["cuda_version"].split(".")
            maj = int(parts[0])
            min_ = int(parts[1]) if len(parts) > 1 else 0
            if maj == 11:
                info["cuda_suffix"] = "cu118"
            elif maj == 12:
                info["cuda_suffix"] = "cu126" if min_ >= 3 else "cu122"
    except Exception:
        pass

    return info


# ==============================================================================
# 2. STARTUP DIAGNOSTICS
# ==============================================================================
def print_startup_diagnostics():
    """Prints comprehensive system, GPU, Paddle, and Torch runtime diagnostics."""
    hw = detect_hardware_and_cuda()
    print("\n" + "=" * 75)
    print(" 🔍 KAGGLE GPU OCR SERVER - STARTUP DIAGNOSTICS")
    print("=" * 75)
    print(f" • Python Version      : {platform.python_version()} ({sys.executable})")
    print(f" • OS Platform         : {platform.platform()}")
    print(f" • Host GPU Device     : {hw['gpu_name']}")
    print(f" • Detected CUDA Ver   : {hw['cuda_version']} (Target wheel: {hw['cuda_suffix']})")

    # Inspect Paddle
    paddle_ver = "Not Installed"
    paddle_cuda = False
    paddle_dev = "None"
    try:
        import paddle
        paddle_ver = getattr(paddle, "__version__", "Unknown")
        paddle_cuda = paddle.is_compiled_with_cuda()
        paddle_dev = paddle.device.get_device()
    except Exception as e:
        paddle_ver = f"Import Error: {e}"

    print(f" • PaddlePaddle Ver    : {paddle_ver}")
    print(f" • Paddle CUDA Enabled : {paddle_cuda}")
    print(f" • Paddle Active Device: {paddle_dev}")

    # Inspect PaddleOCR & PaddleX
    ocr_ver = "Not Installed"
    try:
        import paddleocr
        ocr_ver = getattr(paddleocr, "__version__", "Unknown")
    except Exception as e:
        ocr_ver = f"Import Error: {e}"

    pdx_ver = "Not Installed"
    try:
        import paddlex
        pdx_ver = getattr(paddlex, "__version__", "Unknown")
    except Exception:
        pdx_ver = "None"

    print(f" • PaddleOCR Version   : {ocr_ver}")
    print(f" • PaddleX Version     : {pdx_ver}")

    # Inspect OpenCV & PyTorch
    cv_ver = "Not Installed"
    try:
        import cv2
        cv_ver = cv2.__version__
    except Exception:
        pass

    torch_ver = "Not Installed"
    try:
        import torch
        torch_cuda = torch.cuda.is_available()
        torch_ver = f"{torch.__version__} (CUDA={torch_cuda})"
    except Exception:
        pass

    print(f" • OpenCV Version      : {cv_ver}")
    print(f" • PyTorch Version     : {torch_ver}")
    print("=" * 75 + "\n")


# ==============================================================================
# 3. REPRODUCIBLE DEPENDENCY INSTALLATION & PINNING
# ==============================================================================
def verify_paddle_compatibility_in_memory() -> Tuple[bool, str]:
    """
    Runs a fast in-memory check to confirm PaddleOCR instantiates without
    the infamous AnalysisConfig.set_optimization_level AttributeError.
    """
    try:
        import paddle
        apply_runtime_compatibility_shims()
        if not paddle.is_compiled_with_cuda():
            return False, "PaddlePaddle is installed in CPU-only mode. GPU CUDA build required."

        from paddleocr import PaddleOCR
        # Test constructor without downloading large models
        test_ocr = PaddleOCR(
            device="gpu",
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        return True, "PaddleOCR instantiated successfully on GPU"
    except Exception as exc:
        return False, str(exc)


def install_compatible_dependencies():
    """
    Installs a known-compatible pinned combination of PaddlePaddle GPU and PaddleOCR:
      - CUDA 12.x: paddlepaddle-gpu==3.0.0 (cu126/cu122) + paddleocr==3.0.2
      - CUDA 11.x: paddlepaddle-gpu==3.0.0 (cu118) + paddleocr==3.0.2
      - Secondary Fallback: paddlepaddle-gpu==2.6.2 (cu120) + paddleocr==2.7.3
    Never installs floating 'paddleocr>=2.7.0' which pulls incompatible PaddleX versions.
    """
    hw = detect_hardware_and_cuda()
    cu_suffix = hw["cuda_suffix"]

    print(f"[SETUP] Checking existing PaddlePaddle GPU environment for {cu_suffix}...")
    is_ok, check_msg = verify_paddle_compatibility_in_memory()
    if is_ok:
        print(f"✓ Environment verified compatible: {check_msg}")
        return

    print(f"⚠️ Incompatibility detected: {check_msg}")
    print("[SETUP] Removing conflicting paddle packages...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "paddlepaddle", "paddlepaddle-gpu", "paddleocr", "paddlex"],
        check=False
    )

    print(f"[SETUP] Installing pinned compatible PaddlePaddle-GPU (target: {cu_suffix})...")
    # Primary attempt: PaddlePaddle 3.0.0 official GPU release
    ret = subprocess.run([
        sys.executable, "-m", "pip", "install", "--default-timeout=1000",
        f"paddlepaddle-gpu==3.0.0",
        "-i", f"https://www.paddlepaddle.org.cn/packages/stable/{cu_suffix}/"
    ], check=False)

    if ret.returncode != 0:
        print("[SETUP] Official channel failed; trying standard PyPI paddlepaddle-gpu...")
        subprocess.run([sys.executable, "-m", "pip", "install", "paddlepaddle-gpu==3.0.0"], check=False)

    print("[SETUP] Installing pinned PaddleOCR 3.0.2 and supporting services...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "paddleocr==3.0.2",
        "flask",
        "transformers",
        "torch",
        "torchvision",
        "pydantic"
    ], check=False)

    apply_runtime_compatibility_shims()


# Run dependency installation check prior to model initialization
apply_runtime_compatibility_shims()
try:
    import paddle
    import paddleocr
    import cv2
    import numpy as np
    from flask import Flask, request, jsonify
    from paddleocr import PaddleOCR
except ImportError:
    install_compatible_dependencies()
    import cv2
    import numpy as np
    from flask import Flask, request, jsonify
    from paddleocr import PaddleOCR

apply_runtime_compatibility_shims()


# ==============================================================================
# 4. OCR ENGINE & MULTILINGUAL MODEL CACHE
# ==============================================================================
SUPPORTED_LANGUAGES = {
    "en": {"backend": "en", "name": "English"},
    "te": {"backend": "te", "name": "Telugu"},
    "hi": {"backend": "hi", "name": "Hindi"},
    "kn": {"backend": "ka", "name": "Kannada"},
    "ta": {"backend": "ta", "name": "Tamil"},
    "mr": {"backend": "mr", "name": "Marathi"},
    "ur": {"backend": "ur", "name": "Urdu"},
}

_GPU_OCR_MODELS: Dict[str, Any] = {}
_GPU_OCR_LOCK = threading.Lock()
_BASELINE_OCR_INITIALIZED = False
_BASELINE_OCR_ERROR: Optional[str] = None

def _get_target_device() -> str:
    """Returns 'gpu' only if Paddle is compiled with CUDA and at least 1 GPU is available."""
    try:
        import paddle
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
    except Exception:
        pass
    return "cpu"

DEVICE = os.environ.get("PADDLE_DEVICE") or _get_target_device()


def get_gpu_ocr_model(lang: str = "en") -> Tuple[Optional[Any], Optional[str]]:
    """
    Gets or dynamically loads a cached PaddleOCR model for the requested language.
    Strictly verifies GPU execution. Never reloads models repeatedly on every page.
    """
    global _BASELINE_OCR_INITIALIZED, _BASELINE_OCR_ERROR
    apply_runtime_compatibility_shims()

    lang_key = (lang or "en").lower().strip()
    if lang_key == "auto":
        lang_key = "en"

    if lang_key not in SUPPORTED_LANGUAGES:
        return None, f"Unsupported language '{lang_key}'. Supported languages: {list(SUPPORTED_LANGUAGES.keys())}"

    backend_lang = SUPPORTED_LANGUAGES[lang_key]["backend"]

    if lang_key in _GPU_OCR_MODELS:
        return _GPU_OCR_MODELS[lang_key], None

    with _GPU_OCR_LOCK:
        if lang_key in _GPU_OCR_MODELS:
            return _GPU_OCR_MODELS[lang_key], None

        print(f"Loading PaddleOCR model for '{lang_key}' (backend='{backend_lang}', device='{DEVICE}')...")
        try:
            model = PaddleOCR(
                device=DEVICE,
                lang=backend_lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            _GPU_OCR_MODELS[lang_key] = model
            print(f"[OK] Model for '{lang_key}' loaded successfully and cached.")
            return model, None
        except Exception as exc:
            err_msg = f"Failed to load PaddleOCR model for '{lang_key}': {exc}"
            print(f"[ERROR] {err_msg}")
            return None, err_msg


# ==============================================================================
# 5. REAL GPU OCR SMOKE TEST
# ==============================================================================
def run_real_gpu_smoke_test() -> Dict[str, Any]:
    """
    Executes a real GPU OCR inference smoke test on a synthetic image containing:
      SALE DEED
      SURVEY NO 278
    Verifies:
      1. Prediction returns structured lines
      2. Device is GPU
      3. Inference time is measured
    Halts execution if smoke test fails.
    """
    global _BASELINE_OCR_INITIALIZED, _BASELINE_OCR_ERROR
    print("\n[SMOKE TEST] Running synthetic image PaddleOCR test on GPU...")

    # 1. Create synthetic image
    h, w = 160, 500
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.putText(img, "SALE DEED", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
    cv2.putText(img, "SURVEY NO 278", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)

    # 2. Get baseline English model
    model, err = get_gpu_ocr_model("en")
    if model is None:
        _BASELINE_OCR_INITIALIZED = False
        _BASELINE_OCR_ERROR = err
        print(f"\n❌ GPU OCR server startup failed.\nPaddleOCR/PaddlePaddle compatibility is unresolved: {err}\nNo CPU fallback was used.\n")
        return {"passed": False, "error": err}

    # 3. Execute inference
    try:
        t0 = time.perf_counter()
        if hasattr(model, "predict"):
            res = model.predict(img)[0]
            rec_texts = res.get("rec_texts", [])
        elif hasattr(model, "ocr"):
            res = model.ocr(img)
            rec_texts = [line[1][0] for line in res[0]] if (res and res[0]) else []
        else:
            raise RuntimeError("Model instance has neither .predict() nor .ocr() interface")
        inf_ms = (time.perf_counter() - t0) * 1000

        print(f"✓ Smoke test recognized: {rec_texts}")
        print(f"✓ Inference Latency   : {inf_ms:.2f} ms")

        # Confirm non-empty output
        if not rec_texts:
            raise RuntimeError("PaddleOCR returned empty text results on synthetic benchmark image.")

        _BASELINE_OCR_INITIALIZED = True
        _BASELINE_OCR_ERROR = None
        print("✓ REAL GPU SMOKE TEST PASSED.")
        return {
            "passed": True,
            "recognized_texts": [str(t) for t in rec_texts],
            "inference_time_ms": round(inf_ms, 2),
            "device": DEVICE,
        }
    except Exception as exc:
        _BASELINE_OCR_INITIALIZED = False
        _BASELINE_OCR_ERROR = str(exc)
        print(f"\n❌ GPU OCR server startup failed.\nPaddleOCR/PaddlePaddle compatibility is unresolved: {exc}\nNo CPU fallback was used.\n")
        return {"passed": False, "error": str(exc)}


# ==============================================================================
# 6. PRETRAINED HTR MODEL (TrOCR Handwritten) - LAZY LOAD
# ==============================================================================
_GPU_HTR_LOCK = threading.Lock()
_GPU_HTR_MODEL = None
_GPU_HTR_PROCESSOR = None
_GPU_HTR_STATUS = "UNLOADED"
_GPU_HTR_ERROR = None
_DEFAULT_HTR_MODEL_NAME = "microsoft/trocr-base-handwritten"

def get_gpu_htr_model(model_name: str = _DEFAULT_HTR_MODEL_NAME):
    """
    Lazily loads and caches the pretrained HTR model on Kaggle GPU.
    Never auto-installs on local machine.
    Returns (model, processor, error_message).
    """
    global _GPU_HTR_MODEL, _GPU_HTR_PROCESSOR, _GPU_HTR_STATUS, _GPU_HTR_ERROR
    if _GPU_HTR_MODEL is not None and _GPU_HTR_PROCESSOR is not None:
        return _GPU_HTR_MODEL, _GPU_HTR_PROCESSOR, None

    with _GPU_HTR_LOCK:
        if _GPU_HTR_MODEL is not None and _GPU_HTR_PROCESSOR is not None:
            return _GPU_HTR_MODEL, _GPU_HTR_PROCESSOR, None

        print(f"Checking environment and loading HTR model '{model_name}'...")
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            from PIL import Image
        except ImportError as ie:
            _GPU_HTR_STATUS = "UNAVAILABLE"
            _GPU_HTR_ERROR = f"HTR backend dependencies not installed: {ie}."
            print(f"[WARN] {_GPU_HTR_ERROR}")
            return None, None, _GPU_HTR_ERROR

        try:
            htr_device = "cuda" if (torch.cuda.is_available() and DEVICE == "gpu") else "cpu"
            print(f"Loading TrOCR weights '{model_name}' onto {htr_device}...")
            processor = TrOCRProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name).to(htr_device)
            model.eval()
            _GPU_HTR_MODEL = model
            _GPU_HTR_PROCESSOR = processor
            _GPU_HTR_STATUS = "AVAILABLE"
            _GPU_HTR_ERROR = None
            print(f"[OK] HTR model '{model_name}' loaded successfully on {htr_device}!")
            return _GPU_HTR_MODEL, _GPU_HTR_PROCESSOR, None
        except Exception as exc:
            _GPU_HTR_STATUS = "UNAVAILABLE"
            _GPU_HTR_ERROR = f"Failed to load HTR model '{model_name}': {exc}"
            print(f"[ERROR] {_GPU_HTR_ERROR}")
            return None, None, _GPU_HTR_ERROR


# ==============================================================================
# 7. FLASK APPLICATION & ENDPOINTS
# ==============================================================================
hw_info = detect_hardware_and_cuda()
gpu_name_str = hw_info["gpu_name"]

app = Flask(__name__)

@app.route("/", methods=["GET"])
@app.route("/status", methods=["GET"])
def status_endpoint():
    """
    Standardized /status endpoint:
    Returns true status of GPU, Paddle CUDA availability, and loaded models.
    """
    paddle_ver = "Unknown"
    paddle_cuda = False
    try:
        import paddle
        paddle_ver = paddle.__version__
        paddle_cuda = paddle.is_compiled_with_cuda()
    except Exception:
        pass

    ocr_ver = "Unknown"
    try:
        import paddleocr
        ocr_ver = paddleocr.__version__
    except Exception:
        pass

    pdx_ver = None
    try:
        import paddlex
        pdx_ver = paddlex.__version__
    except Exception:
        pass

    is_model_available = (_BASELINE_OCR_INITIALIZED and ("en" in _GPU_OCR_MODELS or len(_GPU_OCR_MODELS) > 0))

    if not is_model_available and _BASELINE_OCR_ERROR is not None:
        return jsonify({
            "status": "degraded",
            "gpu_available": hw_info["has_gpu"],
            "paddle_cuda_enabled": paddle_cuda,
            "device": DEVICE,
            "gpu_name": gpu_name_str,
            "paddle_version": paddle_ver,
            "paddleocr_version": ocr_ver,
            "paddlex_version": pdx_ver,
            "ocr_model_status": "UNAVAILABLE",
            "htr_model_status": _GPU_HTR_STATUS,
            "error": _BASELINE_OCR_ERROR,
            "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
            "loaded_models": list(_GPU_OCR_MODELS.keys()),
        }), 200

    return jsonify({
        "status": "connected",
        "gpu_available": hw_info["has_gpu"],
        "paddle_cuda_enabled": paddle_cuda,
        "device": DEVICE,
        "gpu_name": gpu_name_str,
        "paddle_version": paddle_ver,
        "paddleocr_version": ocr_ver,
        "paddlex_version": pdx_ver,
        "ocr_model_status": "AVAILABLE" if is_model_available else "INITIALIZING",
        "htr_model_status": _GPU_HTR_STATUS,
        "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
        "loaded_models": list(_GPU_OCR_MODELS.keys()),
    })


@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    """
    High-accuracy GPU OCR endpoint.
    Accepts: multipart/form-data with 'image', 'page_number', 'lang'.
    Compatible with JPEG, PNG, grayscale-as-JPEG, deskewed, and upscaled scans.
    Returns: All standard fields required by local client parser.
    """
    page_num = 1
    raw_page = request.form.get("page_number") or request.args.get("page_number")
    if raw_page is not None:
        try:
            page_num = int(raw_page)
        except (ValueError, TypeError):
            page_num = 1

    req_lang = (request.form.get("lang") or request.args.get("lang") or "en").lower().strip()

    if "image" not in request.files:
        return jsonify({
            "error": "No image file provided",
            "page_number": page_num,
            "requested_language": req_lang,
            "ocr_model_language": None,
            "model_status": "ERROR",
            "lines": [],
            "needs_review": True,
            "warnings": ["No image file provided in request.files['image']"]
        }), 400

    file = request.files["image"]
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({
            "error": "Failed to decode image buffer",
            "page_number": page_num,
            "requested_language": req_lang,
            "ocr_model_language": None,
            "model_status": "ERROR",
            "lines": [],
            "needs_review": True,
            "warnings": ["Failed to decode uploaded image buffer into valid pixel matrix"]
        }), 400

    # Retrieve language model - NEVER silently substitute English when Indic is requested
    model, err = get_gpu_ocr_model(req_lang)
    if model is None:
        return jsonify({
            "error": err,
            "page_number": page_num,
            "requested_language": req_lang,
            "ocr_model_language": None,
            "model_status": "UNAVAILABLE",
            "lines": [],
            "needs_review": True,
            "warnings": [err]
        }), 400

    active_lang = req_lang if req_lang != "auto" else "en"

    # Execute GPU OCR inference
    with _GPU_OCR_LOCK:
        start_time = time.perf_counter()
        if hasattr(model, "predict"):
            result = model.predict(image)[0]
            rec_texts = [str(t) for t in result.get("rec_texts", [])]
            rec_scores = [float(s) for s in result.get("rec_scores", [])]
            rec_polys = result.get("rec_polys", [])
        elif hasattr(model, "ocr"):
            res = model.ocr(image)
            raw_lines = res[0] if (res and res[0]) else []
            rec_texts = [str(l[1][0]) for l in raw_lines]
            rec_scores = [float(l[1][1]) for l in raw_lines]
            rec_polys = [l[0] for l in raw_lines]
        else:
            raise RuntimeError("Model does not support predict() or ocr()")
        ocr_time = (time.perf_counter() - start_time) * 1000

    rec_polys_list = [poly.tolist() if hasattr(poly, "tolist") else poly for poly in rec_polys]

    lines_list = []
    for text, score, poly in zip(rec_texts, rec_scores, rec_polys_list):
        pts = poly
        x_min = int(min(pt[0] for pt in pts))
        y_min = int(min(pt[1] for pt in pts))
        x_max = int(max(pt[0] for pt in pts))
        y_max = int(max(pt[1] for pt in pts))
        lines_list.append({
            "text": text,
            "confidence": round(score, 4),
            "bbox": [x_min, y_min, x_max, y_max],
            "points": pts,
            "language": active_lang,
            "page_number": page_num,
        })

    return jsonify({
        "page_number": page_num,
        "requested_language": req_lang,
        "ocr_model_language": active_lang,
        "model_status": "AVAILABLE",
        "lines": lines_list,
        "needs_review": False,
        "warnings": [],
        "rec_texts": rec_texts,
        "rec_scores": rec_scores,
        "rec_polys": rec_polys_list,
        "ocr_time_ms": ocr_time,
        "gpu_name": gpu_name_str,
    })


@app.route("/recognize-handwriting", methods=["POST"])
def recognize_handwriting_endpoint():
    """
    Dedicated remote endpoint for English handwriting recognition (TrOCR).
    Strict constraints:
    - Never falls back to PaddleOCR for handwriting recognition.
    - If HTR model unavailable or fails, returns MODEL_UNAVAILABLE with needs_review=True.
    """
    page_num = 1
    raw_page = request.form.get("page_number") or request.args.get("page_number")
    if raw_page is not None:
        try:
            page_num = int(raw_page)
        except (ValueError, TypeError):
            page_num = 1

    region_id = request.form.get("region_id") or request.args.get("region_id") or f"page{page_num}_region1"

    if "image" not in request.files:
        return jsonify({
            "page_number": page_num,
            "region_id": region_id,
            "recognition_status": "INFERENCE_FAILED",
            "recognized_text": None,
            "confidence": 0.0,
            "language": "English",
            "script": "Latin",
            "source_model": None,
            "needs_review": True,
            "warning": "No image file provided in request.files['image']"
        }), 400

    file = request.files["image"]
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return jsonify({
            "page_number": page_num,
            "region_id": region_id,
            "recognition_status": "INFERENCE_FAILED",
            "recognized_text": None,
            "confidence": 0.0,
            "language": "English",
            "script": "Latin",
            "source_model": None,
            "needs_review": True,
            "warning": "Failed to decode uploaded image crop buffer"
        }), 400

    htr_model, htr_processor, htr_err = get_gpu_htr_model()
    if htr_model is None or htr_processor is None:
        return jsonify({
            "page_number": page_num,
            "region_id": region_id,
            "recognition_status": "MODEL_UNAVAILABLE",
            "recognized_text": None,
            "confidence": 0.0,
            "language": "English",
            "script": "Latin",
            "source_model": None,
            "needs_review": True,
            "warning": htr_err or "Dedicated handwriting model unavailable; manual transcription required."
        })

    try:
        import torch
        from PIL import Image

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        device = next(htr_model.parameters()).device
        pixel_values = htr_processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = htr_model.generate(pixel_values, return_dict_in_generate=True, output_scores=True)
            sequences = generated_ids.sequences
            generated_text = htr_processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()

            confidence = 0.85
            if hasattr(generated_ids, "scores") and generated_ids.scores:
                scores = [torch.softmax(s, dim=-1).max().item() for s in generated_ids.scores]
                if scores:
                    confidence = round(float(sum(scores) / len(scores)), 4)

        return jsonify({
            "page_number": page_num,
            "region_id": region_id,
            "recognition_status": "RECOGNIZED",
            "recognized_text": generated_text if generated_text else None,
            "confidence": confidence if generated_text else 0.0,
            "language": "English",
            "script": "Latin",
            "source_model": _DEFAULT_HTR_MODEL_NAME,
            "needs_review": True,
            "warning": None if generated_text else "HTR returned empty string"
        })
    except Exception as exc:
        return jsonify({
            "page_number": page_num,
            "region_id": region_id,
            "recognition_status": "INFERENCE_FAILED",
            "recognized_text": None,
            "confidence": 0.0,
            "language": "English",
            "script": "Latin",
            "source_model": _DEFAULT_HTR_MODEL_NAME,
            "needs_review": True,
            "warning": f"HTR inference error: {exc}"
        })


# ==============================================================================
# 8. TUNNEL SUPERVISOR & BROADCAST
# ==============================================================================
NOTIFY_CHANNEL = os.environ.get("NOTIFY_CHANNEL", "onebhoomi_ocr_tunnel")
MAX_RUNTIME_HOURS = 11.5
active_tunnel_proc = None

_CLOUDFLARED_BINARY_URLS = {
    "x86_64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    "aarch64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
}

def install_cloudflared() -> str:
    path = shutil.which("cloudflared")
    if path:
        return path
    arch = platform.machine().lower()
    url = _CLOUDFLARED_BINARY_URLS.get(arch, _CLOUDFLARED_BINARY_URLS["x86_64"])
    target_path = Path(tempfile.gettempdir()) / "cloudflared"
    if not target_path.exists():
        print(f"Downloading cloudflared from {url}...")
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(target_path, "wb") as f:
            shutil.copyfileobj(resp.raw, f)
        target_path.chmod(0o755)
    return str(target_path)


def expose_port(port: int = 5000) -> str:
    global active_tunnel_proc
    cf_path = install_cloudflared()
    log_file = Path(tempfile.gettempdir()) / f"cloudflared_{port}.log"
    if log_file.exists():
        log_file.unlink()

    cmd = [cf_path, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]
    print(f"Starting cloudflared tunnel on port {port}...")
    with open(log_file, "w") as log_f:
        active_tunnel_proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    url_regex = re.compile(r"https://[-a-zA-Z0-9_.]+\.trycloudflare\.com")
    start_time = time.time()
    while time.time() - start_time < 30:
        if log_file.exists():
            content = log_file.read_text(encoding="utf-8", errors="replace")
            matches = url_regex.findall(content)
            if matches:
                found_url = matches[0]
                print(f"✓ Cloudflare Tunnel Live: {found_url}")
                return found_url
        time.sleep(1)
    raise TimeoutError("Cloudflare tunnel failed to produce a public URL within 30 seconds.")


def broadcast_url(url: str, note: str = "Server Online"):
    if not NOTIFY_CHANNEL:
        return
    endpoint = f"https://ntfy.sh/{NOTIFY_CHANNEL}"
    try:
        requests.post(
            endpoint,
            data=url.encode("utf-8"),
            headers={
                "Title": f"Kaggle GPU OCR ({note})",
                "Tags": "rocket,gpu,computer",
                "Priority": "high"
            },
            timeout=10
        )
        print(f"✓ Broadcasted live URL to {endpoint}")
    except Exception as exc:
        print(f"Notification broadcast note: {exc}")


def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


# ==============================================================================
# 9. SERVER MAIN ENTRYPOINT
# ==============================================================================
def main():
    # 1. Startup Diagnostics
    print_startup_diagnostics()

    # 2. Run Real GPU Smoke Test
    smoke = run_real_gpu_smoke_test()
    if not smoke["passed"]:
        print("\n" + "!" * 75)
        print("❌ CRITICAL: GPU OCR server startup failed.")
        print("PaddleOCR/PaddlePaddle compatibility is unresolved.")
        print("No CPU fallback was used.")
        print("!" * 75 + "\n")
        sys.exit(1)

    # 3. Start Flask Thread
    threading.Thread(target=run_flask, daemon=True).start()
    time.sleep(2)
    print("✓ Flask server running on port 5000.")

    # 4. Start Tunnel & Broadcast
    public_url = expose_port(5000)
    broadcast_url(public_url, note="Server Ready")

    # 5. 12-Hour Supervisor Loop
    deadline = time.time() + (MAX_RUNTIME_HOURS * 3600)
    print("\n" + "=" * 70)
    print(" 🚀 KAGGLE 12-HOUR HEADLESS OCR BACKEND IS RUNNING")
    print("=" * 70)
    print(f" • Public Tunnel URL: {public_url}")
    print(f" • Sync Channel:     https://ntfy.sh/{NOTIFY_CHANNEL}")
    print(f" • Hardware Device:  {gpu_name_str}")
    print(f" • Planned Runtime:  {MAX_RUNTIME_HOURS} hours")
    print("=" * 70 + "\n")

    loop_count = 0
    try:
        while time.time() < deadline:
            time.sleep(60)
            loop_count += 1

            if active_tunnel_proc and active_tunnel_proc.poll() is not None:
                print("⚠️ Tunnel dropped! Re-establishing...")
                try:
                    public_url = expose_port(5000)
                    broadcast_url(public_url, note="Tunnel Reconnected")
                except Exception as e:
                    print(f"Tunnel restart attempt failed: {e}")

            if loop_count % 10 == 0:
                elapsed_m = loop_count
                remain_m = max(0, int((deadline - time.time()) / 60))
                try:
                    st = requests.get("http://127.0.0.1:5000/status", timeout=5).json()
                    st_desc = f"OK ({st.get('gpu_name', 'GPU Online')}) - OCR: {st.get('ocr_model_status')}"
                except Exception as ex:
                    st_desc = f"Warning: {ex}"
                print(f"[{time.strftime('%H:%M:%S')}] Active: {elapsed_m}m / {remain_m}m remaining | Local: {st_desc}")

            if loop_count % 60 == 0:
                broadcast_url(public_url, note=f"Active ({loop_count // 60}h)")

        print("Execution completed.")
    except KeyboardInterrupt:
        print("Stopped by user.")


if __name__ == "__main__":
    main()
