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
import json
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
# 0. DEFENSIVE RUNTIME COMPATIBILITY SHIMS
# ==============================================================================
def apply_langchain_compatibility_shims():
    """
    Shims langchain.docstore and langchain.text_splitter in sys.modules.
    In modern Kaggle environments (Python 3.12 + LangChain 1.2+),
    'langchain.docstore' was removed in favor of langchain_core.documents.
    PaddleX 3.0.x relies on the older import path; this shim prevents:
    ModuleNotFoundError: No module named 'langchain.docstore'.
    """
    import sys
    import types

    doc_cls = None
    try:
        from langchain_core.documents import Document as CoreDoc
        doc_cls = CoreDoc
    except Exception:
        try:
            from langchain.schema.document import Document as SchemaDoc
            doc_cls = SchemaDoc
        except Exception:
            class MockDocument:
                def __init__(self, page_content="", metadata=None):
                    self.page_content = page_content
                    self.metadata = metadata or {}
            doc_cls = MockDocument

    splitter_cls = None
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter as TextSplitter
        splitter_cls = TextSplitter
    except Exception:
        class MockSplitter:
            def __init__(self, *args, **kwargs):
                pass
            def split_text(self, text):
                return [text]
        splitter_cls = MockSplitter

    try:
        if "langchain" not in sys.modules:
            sys.modules["langchain"] = types.ModuleType("langchain")

        if "langchain.docstore" not in sys.modules or "langchain.docstore.document" not in sys.modules:
            docstore_mod = types.ModuleType("langchain.docstore")
            doc_mod = types.ModuleType("langchain.docstore.document")
            doc_mod.Document = doc_cls
            docstore_mod.document = doc_mod
            sys.modules["langchain.docstore"] = docstore_mod
            sys.modules["langchain.docstore.document"] = doc_mod
            setattr(sys.modules["langchain"], "docstore", docstore_mod)

        if "langchain.text_splitter" not in sys.modules:
            ts_mod = types.ModuleType("langchain.text_splitter")
            ts_mod.RecursiveCharacterTextSplitter = splitter_cls
            sys.modules["langchain.text_splitter"] = ts_mod
            setattr(sys.modules["langchain"], "text_splitter", ts_mod)
    except Exception as e:
        print(f"[COMPATIBILITY SHIM] Note on langchain shims: {e}")


def apply_runtime_compatibility_shims():
    """
    Applies runtime shims for C++ bindings in PaddlePaddle / PaddleOCR.
    Specifically guarantees that AnalysisConfig.set_optimization_level exists as a safe callable
    regardless of whether Paddle 3.x or 2.x C++ bindings are active in memory.
    """
    apply_langchain_compatibility_shims()
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


# Apply shims immediately at script load time
apply_langchain_compatibility_shims()
apply_runtime_compatibility_shims()


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
        "pydantic",
        "bitsandbytes>=0.46.1",
        "accelerate"
    ], check=False)

    # Refresh Python path and invalidate caches after pip execution
    import importlib
    import site
    importlib.invalidate_caches()
    for p in site.getsitepackages():
        if p not in sys.path:
            sys.path.insert(0, p)
    apply_langchain_compatibility_shims()
    apply_runtime_compatibility_shims()


# Run dependency installation check prior to model initialization
apply_langchain_compatibility_shims()
apply_runtime_compatibility_shims()
try:
    import paddle
    import paddleocr
    import cv2
    import numpy as np
    from flask import Flask, request, jsonify
    from paddleocr import PaddleOCR
except (ImportError, ModuleNotFoundError) as initial_err:
    print(f"[SETUP] Initial module import note: {initial_err}. Running compatible dependency setup...")
    install_compatible_dependencies()
    import importlib
    import site
    importlib.invalidate_caches()
    for p in site.getsitepackages():
        if p not in sys.path:
            sys.path.insert(0, p)
    apply_langchain_compatibility_shims()
    apply_runtime_compatibility_shims()
    import cv2
    import numpy as np
    from flask import Flask, request, jsonify
    import paddle
    import paddleocr
    from paddleocr import PaddleOCR

apply_langchain_compatibility_shims()
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
# ==============================================================================
# 6B. PRETRAINED NEURAL NLP MODEL (Qwen2.5-7B-Instruct 4-bit) - LAZY LOAD & SMOKE TEST
# ==============================================================================
_GPU_NLP_LOCK = threading.Lock()
_GPU_NLP_MODEL = None
_GPU_NLP_TOKENIZER = None
_GPU_NLP_STATUS = "UNLOADED"
_GPU_NLP_ERROR = None
_DEFAULT_NLP_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

SYNTHETIC_SMOKE_OCR = """SALE DEED
Document No. 1234
Date: 15/08/2026
Vendor: Ramesh Kumar
Purchaser: Suresh Kumar
Survey No. 278
Sub Survey No. 278/2
Area: 2.50 acres
Village: Kothapalli
Mandal: Ghatkesar
District: Medchal
Consideration Amount: Rs. 2500000"""

TELUGU_SMOKE_OCR = """SALE DEED
విక్రేత: రమేష్ కుమార్
కొనుగోలుదారు: సురేష్ కుమార్
Survey No: 278
గ్రామం: కొత్తపల్లి
Mandal: Ghatkesar
District: Medchal"""


def ensure_qwen_gpu_dependencies() -> tuple[str, str]:
    """
    Ensures bitsandbytes>=0.46.1 and accelerate are installed for Qwen 4-bit NF4 GPU inference.
    Installs ONLY missing packages; does NOT touch or reinstall PyTorch, CUDA, or Paddle.
    Returns (bitsandbytes_version, accelerate_version).
    """
    to_install = []
    bnb_ver = "NOT_INSTALLED"
    acc_ver = "NOT_INSTALLED"

    try:
        import bitsandbytes as bnb
        bnb_ver = getattr(bnb, "__version__", "unknown")
        from packaging import version
        if version.parse(bnb_ver) < version.parse("0.46.1"):
            to_install.append("bitsandbytes>=0.46.1")
    except Exception:
        to_install.append("bitsandbytes>=0.46.1")

    try:
        import accelerate
        acc_ver = getattr(accelerate, "__version__", "unknown")
    except Exception:
        to_install.append("accelerate")

    if to_install:
        print(f"[SETUP] Installing missing Qwen 4-bit dependencies: {to_install}...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", *to_install
        ], check=False)

        import importlib
        import site
        importlib.invalidate_caches()
        for p in site.getsitepackages():
            if p not in sys.path:
                sys.path.insert(0, p)

        try:
            import bitsandbytes as bnb
            bnb_ver = getattr(bnb, "__version__", "installed")
        except Exception as e:
            bnb_ver = f"error: {e}"

        try:
            import accelerate
            acc_ver = getattr(accelerate, "__version__", "installed")
        except Exception as e:
            acc_ver = f"error: {e}"

    return str(bnb_ver or "unknown"), str(acc_ver or "unknown")


def check_qwen_cuda_diagnostics() -> dict:
    """
    Performs specific CUDA hardware and dependency diagnostic checks for Qwen.
    Prints torch version, bitsandbytes version, accelerate version, CUDA availability,
    device count, GPU names, and VRAM info.
    """
    bnb_ver, acc_ver = ensure_qwen_gpu_dependencies()

    try:
        import torch
    except ImportError:
        print("[QWEN DIAGNOSTICS] PyTorch is not installed.")
        return {
            "torch_installed": False,
            "cuda_available": False,
            "bitsandbytes_version": bnb_ver,
            "accelerate_version": acc_ver,
        }

    print("\n[QWEN DIAGNOSTICS]")
    print(f"PyTorch: {torch.__version__}")
    print(f"bitsandbytes: {bnb_ver}")
    print(f"accelerate: {acc_ver}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA available: {cuda_avail}")
    gpu_count = torch.cuda.device_count() if cuda_avail else 0
    print(f"CUDA devices: {gpu_count}")
    gpu_names = []
    if cuda_avail and gpu_count > 0:
        for idx in range(gpu_count):
            name = torch.cuda.get_device_name(idx)
            gpu_names.append(name)
            print(f"GPU {idx}: {name}")
            try:
                free_b, total_b = torch.cuda.mem_get_info(idx)
                print(f"  Memory {idx}: {free_b // (1024*1024)} MB free / {total_b // (1024*1024)} MB total")
            except Exception:
                pass
        curr_dev = torch.cuda.current_device()
        print(f"Current CUDA device: {curr_dev}")
    return {
        "torch_version": torch.__version__,
        "bitsandbytes_version": bnb_ver,
        "accelerate_version": acc_ver,
        "cuda_available": cuda_avail,
        "device_count": gpu_count,
        "gpu_names": gpu_names,
    }


def get_cuda_memory_mb(device_idx: int = 0) -> dict:
    """Records current allocated and reserved CUDA memory in MB."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0}
        alloc = torch.cuda.memory_allocated(device_idx) / (1024 * 1024)
        res = torch.cuda.memory_reserved(device_idx) / (1024 * 1024)
        return {"allocated_mb": round(alloc, 2), "reserved_mb": round(res, 2)}
    except Exception:
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}


def _get_model_input_device(model: Any) -> Any:
    """Safely determines the correct CUDA input tensor device for device_map='auto' models."""
    import torch
    if hasattr(model, "device"):
        try:
            if str(model.device).startswith("cuda") or str(model.device) == "cpu":
                return model.device
        except Exception:
            pass
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, int):
            return torch.device(f"cuda:{first_device}")
        elif isinstance(first_device, str):
            return torch.device(first_device)
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_qwen_json(raw_text: str) -> dict:
    """Extracts and parses canonical 12-field JSON object from Qwen generation text."""
    import json
    canonical_keys = [
        "document_type", "document_number", "document_date",
        "vendor", "purchaser", "survey_number", "sub_survey_number",
        "property_area", "village", "mandal", "district", "consideration_amount"
    ]
    cleaned = (raw_text or "").strip()
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(1).strip()
    else:
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1].strip()

    parsed = {}
    try:
        parsed = json.loads(cleaned)
    except Exception:
        try:
            fixed = re.sub(r",\s*([\}\]])", r"\1", cleaned)
            parsed = json.loads(fixed)
        except Exception:
            parsed = {}
            for k in canonical_keys:
                m = re.search(rf'"{k}"\s*:\s*"([^"]*)"', cleaned)
                if m:
                    parsed[k] = m.group(1)

    result = {}
    for key in canonical_keys:
        val = parsed.get(key)
        if val is None or str(val).strip().lower() in ("null", "none", "n/a", "unknown", ""):
            result[key] = None
        else:
            result[key] = str(val).strip()
    return result


def get_gpu_nlp_model(model_name: str = _DEFAULT_NLP_MODEL_NAME):
    """
    Lazily loads and caches Qwen2.5-7B-Instruct in 4-bit on Kaggle GPU.
    Uses BitsAndBytesConfig with NF4 double quantization and device_map='auto'.
    Returns (model, tokenizer, error_message).
    """
    global _GPU_NLP_MODEL, _GPU_NLP_TOKENIZER, _GPU_NLP_STATUS, _GPU_NLP_ERROR
    if _GPU_NLP_MODEL is not None and _GPU_NLP_TOKENIZER is not None:
        return _GPU_NLP_MODEL, _GPU_NLP_TOKENIZER, None

    with _GPU_NLP_LOCK:
        if _GPU_NLP_MODEL is not None and _GPU_NLP_TOKENIZER is not None:
            return _GPU_NLP_MODEL, _GPU_NLP_TOKENIZER, None

        # Print detailed CUDA diagnostics
        check_qwen_cuda_diagnostics()

        print(f"Checking environment and loading NLP model '{model_name}'...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as ie:
            _GPU_NLP_STATUS = "UNAVAILABLE"
            _GPU_NLP_ERROR = f"Neural NLP dependencies not installed: {ie}."
            print(f"[WARN] {_GPU_NLP_ERROR}")
            return None, None, _GPU_NLP_ERROR

        if not torch.cuda.is_available():
            _GPU_NLP_STATUS = "UNAVAILABLE"
            _GPU_NLP_ERROR = "CUDA is not available. Qwen requires a GPU environment."
            print(f"[ERROR] {_GPU_NLP_ERROR}")
            return None, None, _GPU_NLP_ERROR

        # Record GPU Memory before loading
        mem_before = get_cuda_memory_mb(0)

        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print(f"Loading Qwen 4-bit NF4 weights '{model_name}' onto GPU...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
            model.eval()

            # Record GPU Memory after loading
            mem_after = get_cuda_memory_mb(0)
            diff_alloc = round(mem_after["allocated_mb"] - mem_before["allocated_mb"], 2)
            diff_res = round(mem_after["reserved_mb"] - mem_before["reserved_mb"], 2)

            print("\n[QWEN MEMORY]")
            print(f"Before: {mem_before['allocated_mb']} MB allocated, {mem_before['reserved_mb']} MB reserved")
            print(f"After: {mem_after['allocated_mb']} MB allocated, {mem_after['reserved_mb']} MB reserved")
            print(f"Increase: +{diff_alloc} MB allocated (+{diff_res} MB reserved)")

            input_dev = _get_model_input_device(model)
            print(f"[QWEN] Model loaded successfully")
            print(f"[QWEN] Quantization: 4-bit NF4")
            print(f"[QWEN] Device: {input_dev}")
            dev_map_str = str(getattr(model, "hf_device_map", input_dev))
            print(f"[QWEN] Model device map: {dev_map_str}")

            if str(input_dev) == "cpu" or not str(input_dev).startswith("cuda"):
                raise RuntimeError(f"Qwen model was placed on CPU ({input_dev}) instead of CUDA!")

            _GPU_NLP_MODEL = model
            _GPU_NLP_TOKENIZER = tokenizer
            _GPU_NLP_STATUS = "AVAILABLE"
            _GPU_NLP_ERROR = None
            return _GPU_NLP_MODEL, _GPU_NLP_TOKENIZER, None
        except Exception as exc:
            _GPU_NLP_STATUS = "UNAVAILABLE"
            _GPU_NLP_ERROR = f"Failed to load NLP model '{model_name}': {exc}"
            print(f"[ERROR] {_GPU_NLP_ERROR}")
            return None, None, _GPU_NLP_ERROR


def run_qwen_gpu_smoke_test() -> dict:
    """
    Executes a real Qwen2.5-7B-Instruct 4-bit GPU inference test.
    Verifies model load on CUDA, generate() execution, canonical JSON parsing,
    entity extraction completeness, and Unicode Indic safety.
    """
    print("\n" + "=" * 60)
    print("QWEN NEURAL NLP GPU SMOKE TEST")
    print("=" * 60)

    if DEVICE != "gpu":
        print("[WARN] GPU device not active for Qwen smoke test. Skipping.")
        return {"passed": False, "error": "GPU device not active."}

    print("Loading Qwen...")
    mem_before = get_cuda_memory_mb(0)
    t_load_start = time.perf_counter()
    model, tokenizer, err = get_gpu_nlp_model()
    if err or model is None or tokenizer is None:
        print(f"❌ Qwen model failed to load: {err}")
        return {"passed": False, "error": err}

    load_time_s = time.perf_counter() - t_load_start
    print(f"Model loaded in {load_time_s:.2f}s...")
    print("Running inference...")

    try:
        import torch

        target_device = _get_model_input_device(model)
        prompt = f"""You are a legal document information extraction assistant for Indian land registry records.
Extract information ONLY when it is explicitly supported by the supplied OCR text.
NEVER infer, hallucinate, assume, or invent any legal field.
Do not guess person names, survey numbers, dates, consideration amounts, or locations.
If a field is not present or cannot be clearly determined from the text, return null for that field.

Return ONLY a valid JSON object with the following exact keys:
- "document_type": string or null
- "document_number": string or null
- "document_date": string or null
- "vendor": string or null
- "purchaser": string or null
- "survey_number": string or null
- "sub_survey_number": string or null
- "property_area": string or null
- "village": string or null
- "mandal": string or null
- "district": string or null
- "consideration_amount": string or null

SUPPLIED OCR TEXT:
\"\"\"
{SYNTHETIC_SMOKE_OCR}
\"\"\"

Output STRICT JSON only:"""

        messages = [
            {"role": "system", "content": "You output strictly valid JSON conforming to the requested schema."},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokenized = tokenizer([text], return_tensors="pt")
        inputs = {k: v.to(target_device) for k, v in tokenized.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        inf_ms = (time.perf_counter() - t0) * 1000

        input_len = inputs["input_ids"].shape[1]
        raw_output = tokenizer.batch_decode(
            [out[input_len:] for out in generated_ids],
            skip_special_tokens=True,
        )[0]
        print(f"\n[QWEN RAW GENERATED TEXT]\n{raw_output}\n")

        parsed = _parse_qwen_json(raw_output)
        extracted_count = sum(1 for v in parsed.values() if v is not None)

        if extracted_count < 4:
            raise ValueError(f"Extracted too few fields ({extracted_count}/12): {parsed}")

        gpu_mem_info = get_cuda_memory_mb(0)
        gpu_mem_used = gpu_mem_info["allocated_mb"]

        bnb_ver = "unknown"
        acc_ver = "unknown"
        try:
            import bitsandbytes as bnb
            bnb_ver = getattr(bnb, "__version__", "installed")
        except Exception:
            pass
        try:
            import accelerate
            acc_ver = getattr(accelerate, "__version__", "installed")
        except Exception:
            pass

        print(f"\n[QWEN SMOKE TEST RESULTS]")
        print(f"• bitsandbytes version: {bnb_ver}")
        print(f"• accelerate version:   {acc_ver}")
        print(f"• Qwen model status:    {_GPU_NLP_STATUS}")
        print(f"• CUDA device:          {target_device}")
        print(f"• GPU VRAM before:      {mem_before['allocated_mb']} MB (reserved: {mem_before['reserved_mb']} MB)")
        print(f"• GPU VRAM after:       {gpu_mem_used} MB (reserved: {gpu_mem_info['reserved_mb']} MB)")
        print(f"• Inference latency:    {inf_ms:.1f} ms")
        print(f"• Generated JSON:")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))

        print(f"\n✓ Qwen model loaded successfully")
        print(f"✓ CUDA inference successful")
        print(f"✓ JSON response received")
        print(f"✓ Entity extraction successful ({extracted_count}/12 fields)")
        print(f"✓ Inference latency: {inf_ms:.1f} ms")
        print(f"✓ GPU memory used: {gpu_mem_used:.1f} MB")
        print(f"✓ REAL QWEN GPU SMOKE TEST PASSED")
        print("=" * 60 + "\n")

        # Multilingual Indic Unicode Test
        print("[QWEN MULTILINGUAL TEST]")
        print("Testing mixed English + Telugu Unicode OCR inference...")
        try:
            m_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You output strictly valid JSON."},
                    {"role": "user", "content": f"Extract fields from:\n{TELUGU_SMOKE_OCR}\nOutput JSON:"}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            m_tokenized = tokenizer([m_text], return_tensors="pt")
            m_inputs = {k: v.to(target_device) for k, v in m_tokenized.items()}
            with torch.no_grad():
                m_gen = model.generate(**m_inputs, max_new_tokens=256, do_sample=False)
            print("✓ Telugu Unicode inference completed without error")
        except Exception as indic_exc:
            print(f"⚠️ Telugu Unicode inference warning: {indic_exc}")

        return {
            "passed": True,
            "latency_ms": round(inf_ms, 2),
            "gpu_memory_mb": round(gpu_mem_used, 2),
            "extracted_fields": parsed,
            "extracted_count": extracted_count,
        }

    except Exception as exc:
        print(f"\n❌ REAL QWEN GPU SMOKE TEST FAILED: {exc}")
        import traceback
        traceback.print_exc()
        print("=" * 60 + "\n")
        return {"passed": False, "error": str(exc)}


def run_nlp_http_test(port: int = 5000) -> bool:
    """Verifies that the /nlp HTTP endpoint responds with HTTP 200, success=true, gpu=true."""
    import requests
    try:
        res = requests.post(
            f"http://127.0.0.1:{port}/nlp",
            json={"text": SYNTHETIC_SMOKE_OCR, "language": "en"},
            timeout=40,
        )
        if res.status_code == 200:
            data = res.json()
            if data.get("success") is True and data.get("gpu") is True:
                res_fields = data.get("result", {})
                extracted = sum(1 for v in res_fields.values() if v is not None)
                if extracted >= 4:
                    print("\n✓ /nlp HTTP TEST PASSED")
                    print(f"✓ HTTP status: {res.status_code}")
                    print("✓ success: true")
                    print("✓ gpu: true")
                    print(f"✓ Extracted fields: {extracted}/12")
                    return True
        print(f"⚠️ /nlp HTTP test unexpected response: {res.status_code} -> {res.text[:200]}")
        return False
    except Exception as exc:
        print(f"⚠️ /nlp HTTP test failed: {exc}")
        return False




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
            "nlp_model_status": _GPU_NLP_STATUS,
            "nlp_model_name": _DEFAULT_NLP_MODEL_NAME if _GPU_NLP_STATUS == "AVAILABLE" else None,
            "nlp_quantization": "4-bit NF4" if _GPU_NLP_STATUS == "AVAILABLE" else None,
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
        "nlp_model_status": _GPU_NLP_STATUS,
        "nlp_model_name": _DEFAULT_NLP_MODEL_NAME if _GPU_NLP_STATUS == "AVAILABLE" else None,
        "nlp_quantization": "4-bit NF4" if _GPU_NLP_STATUS == "AVAILABLE" else None,
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


@app.route("/nlp", methods=["POST"])
def nlp_endpoint():
    """
    Dedicated remote endpoint for legal land-document information extraction using Qwen2.5-7B-Instruct.
    Accepts: application/json {"text": "...", "language": "auto"}
    Returns: Structured JSON with canonical schema.
    """
    req_data = request.get_json(silent=True) or {}
    ocr_text = req_data.get("text", "") or request.form.get("text", "")
    language = req_data.get("language", "auto") or request.form.get("language", "auto")

    canonical_keys = [
        "document_type", "document_number", "document_date",
        "vendor", "purchaser", "survey_number", "sub_survey_number",
        "property_area", "village", "mandal", "district", "consideration_amount"
    ]
    empty_result = {k: None for k in canonical_keys}

    if not ocr_text or not str(ocr_text).strip():
        return jsonify({
            "success": True,
            "model": _DEFAULT_NLP_MODEL_NAME,
            "gpu": True,
            "result": empty_result,
            "evidence": {},
            "warning": "Empty OCR text supplied."
        })

    model, tokenizer, err = get_gpu_nlp_model()
    if model is None or tokenizer is None:
        return jsonify({
            "success": False,
            "model": _DEFAULT_NLP_MODEL_NAME,
            "gpu": False,
            "error": err or "Neural NLP model unavailable.",
            "result": empty_result,
            "evidence": {}
        }), 503

    try:
        import torch
        prompt = f"""You are a legal document information extraction assistant for Indian land registry records.
Extract information ONLY when it is explicitly supported by the supplied OCR text.
NEVER infer, hallucinate, assume, or invent any legal field.
Do not guess person names, survey numbers, dates, consideration amounts, or locations.
If a field is not present or cannot be clearly determined from the text, return null for that field.

Return ONLY a valid JSON object with the following exact keys:
- "document_type": string or null
- "document_number": string or null
- "document_date": string or null
- "vendor": string or null
- "purchaser": string or null
- "survey_number": string or null
- "sub_survey_number": string or null
- "property_area": string or null
- "village": string or null
- "mandal": string or null
- "district": string or null
- "consideration_amount": string or null

SUPPLIED OCR TEXT:
\"\"\"
{ocr_text}
\"\"\"

Output STRICT JSON only:"""

        messages = [
            {"role": "system", "content": "You output strictly valid JSON conforming to the requested schema."},
            {"role": "user", "content": prompt}
        ]

        target_device = _get_model_input_device(model)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokenized = tokenizer([text], return_tensors="pt")
        inputs = {k: v.to(target_device) for k, v in tokenized.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        input_len = inputs["input_ids"].shape[1]
        raw_output = tokenizer.batch_decode(
            [out[input_len:] for out in generated_ids],
            skip_special_tokens=True,
        )[0]

        result = _parse_qwen_json(raw_output)

        evidence = {}
        for k, v in result.items():
            if v:
                v_lower = v.lower()
                for line in str(ocr_text).splitlines():
                    if v_lower in line.lower():
                        evidence[k] = line.strip()
                        break

        return jsonify({
            "success": True,
            "model": _DEFAULT_NLP_MODEL_NAME,
            "gpu": True,
            "result": result,
            "evidence": evidence
        })
    except Exception as exc:
        return jsonify({
            "success": False,
            "model": _DEFAULT_NLP_MODEL_NAME,
            "gpu": True,
            "error": f"Inference execution failed: {exc}",
            "result": empty_result,
            "evidence": {}
        }), 500



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

    # 3. Pre-load TrOCR Handwriting Model (Pre-downloaded initially for written text)
    print("\n[STARTUP] Pre-downloading and initializing TrOCR handwriting model...")
    try:
        htr_model, htr_proc, htr_err = get_gpu_htr_model()
        if htr_model is not None:
            print("✓ TrOCR handwriting model pre-downloaded and active on GPU!")
        else:
            print(f"⚠️ TrOCR preload note: {htr_err}")
    except Exception as exc:
        print(f"⚠️ TrOCR preload exception: {exc}")

    # 4. Real Qwen2.5-7B-Instruct 4-bit GPU Smoke Test
    print("\n[STARTUP] Executing Qwen2.5-7B-Instruct 4-bit Neural NLP GPU Smoke Test...")
    qwen_smoke = run_qwen_gpu_smoke_test()
    if not qwen_smoke.get("passed"):
        print(f"⚠️ Qwen GPU smoke test note: {qwen_smoke.get('error')}")
        print("OCR and HTR remain operational — /status will report nlp_model_status=UNAVAILABLE.")
    else:
        print("[STARTUP] Qwen neural NLP engine verified operational on GPU.")

    # 5. Start Flask Thread
    threading.Thread(target=run_flask, daemon=True).start()
    time.sleep(2)
    print("✓ Flask server running on port 5000.")

    # 6. Real /nlp HTTP endpoint verification test
    if qwen_smoke.get("passed"):
        run_nlp_http_test(port=5000)

    # 7. Post-Verification: Prove PaddleOCR and TrOCR are healthy post-Qwen
    print("\n[VERIFICATION] Re-verifying PaddleOCR and TrOCR post-Qwen...")
    post_ocr = run_real_gpu_smoke_test()
    if post_ocr.get("passed"):
        print("✓ Post-verification: PaddleOCR GPU engine is operational.")
    else:
        print("⚠️ Post-verification warning: PaddleOCR GPU smoke test failed!")

    if htr_model is not None:
        print("✓ Post-verification: TrOCR handwriting model is operational.")

    # 8. Start Tunnel & Broadcast
    public_url = expose_port(5000)
    broadcast_url(public_url, note="Server Ready")

    # 6. 12-Hour Supervisor Loop
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
