#!/usr/bin/env python3
"""
Kaggle GPU OCR Server - 12-Hour Headless Mode
Can be run directly as a script on Kaggle or pasted into a single Kaggle Notebook cell.

Run headless without keeping browser open:
1. Turn GPU ON (T4 x2 or P100)
2. Turn Internet ON
3. Click "Save Version" -> "Save & Run All (Commit)"
4. Close your browser tab or computer!
5. On your laptop, run: ./update_ocr_url.sh --sync
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
# 0. AUTOMATIC DEPENDENCY INSTALLATION (PaddlePaddle GPU + PaddleOCR)
# ==============================================================================
def install_dependencies():
    print("Checking / Installing PaddlePaddle GPU & PaddleOCR...")
    try:
        import paddle
        import paddleocr
        print("✓ Paddle & PaddleOCR already installed.")
        return
    except ImportError:
        pass

    # Detect CUDA
    cu_suffix = "cu120"
    try:
        smi_out = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", smi_out)
        if match:
            v = match.group(1)
            parts = v.split(".")
            maj, min_ = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
            if maj == 11:
                cu_suffix = "cu118"
            elif maj == 12 and min_ >= 2:
                cu_suffix = "cu122" if min_ < 6 else "cu126"
    except Exception:
        pass

    print(f"Installing PaddlePaddle-GPU ({cu_suffix})...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--default-timeout=1000", "paddlepaddle-gpu",
        "-i", f"https://www.paddlepaddle.org.cn/packages/stable/{cu_suffix}/"
    ], check=False)

    print("Installing PaddleOCR, Flask, Transformers, and Torch...")
    subprocess.run([sys.executable, "-m", "pip", "install", "paddleocr>=2.7.0", "flask", "transformers", "torch", "torchvision"], check=False)

# Ensure dependencies are present when running as notebook or script
try:
    import paddleocr
    import paddle
except ImportError:
    install_dependencies()

import cv2
import numpy as np
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR

# ==============================================================================
# 1. NOTIFICATION & 12-HOUR SETTINGS
# ==============================================================================
NOTIFY_CHANNEL = os.environ.get("NOTIFY_CHANNEL", "onebhoomi_ocr_tunnel")
MAX_RUNTIME_HOURS = 11.5

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

# ==============================================================================
# 2. TUNNEL SUPERVISOR
# ==============================================================================
_CLOUDFLARED_BINARY_URLS = {
    "x86_64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    "amd64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    "aarch64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
    "arm64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
}

active_tunnel_proc = None

def _stream_until_url(proc, patterns, timeout_s=120):
    global active_tunnel_proc
    active_tunnel_proc = proc
    if proc.stdout is None:
        raise RuntimeError("Tunnel process stdout is not available.")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line:
            print(line, end="")
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    return match.group(0)
        elif proc.poll() is not None:
            break
        else:
            time.sleep(0.2)
    raise TimeoutError("Timed out while waiting for a public tunnel URL.")

def _download_cloudflared_binary():
    machine = platform.machine().lower()
    asset_url = _CLOUDFLARED_BINARY_URLS.get(machine)
    if not asset_url:
        raise RuntimeError(f"Unsupported architecture for cloudflared: {machine}")
    target_dir = Path(tempfile.gettempdir()) / "codex-cloudflared"
    target_dir.mkdir(parents=True, exist_ok=True)
    binary_path = target_dir / "cloudflared"
    if binary_path.exists():
        return str(binary_path)
    print(f"Downloading cloudflared from {asset_url}...")
    response = requests.get(asset_url, timeout=120)
    response.raise_for_status()
    binary_path.write_bytes(response.content)
    binary_path.chmod(0o755)
    return str(binary_path)

def expose_port(port: int = 5000) -> str:
    binary = shutil.which("cloudflared") or _download_cloudflared_binary()
    print("Starting cloudflared quick tunnel...")
    proc = subprocess.Popen(
        [binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    url = _stream_until_url(proc, [re.compile(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", re.IGNORECASE)])
    print(f"\nPublic OCR URL: {url}")
    return url

# ==============================================================================
# 3. PADDLEOCR ENGINE & MULTILINGUAL MODEL CACHE
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

_GPU_OCR_MODELS = {}
_GPU_OCR_LOCK = threading.Lock()

def _detect_device():
    try:
        import paddle
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
    except Exception:
        pass
    return "cpu"

DEVICE = os.environ.get("PADDLE_DEVICE") or _detect_device()

def get_gpu_ocr_model(lang: str = "en") -> tuple:
    """
    Get or dynamically load a cached PaddleOCR model for the requested language.
    Does NOT reload models on every page.
    Returns (model_instance, error_message).
    """
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

# Pre-load English model at startup
print(f"Initializing baseline PaddleOCR model (device='{DEVICE}')...")
_init_model, _init_err = get_gpu_ocr_model("en")
if _init_model is not None:
    print("[OK] Baseline English PaddleOCR model initialized successfully!")
else:
    print(f"[WARN] Warning initializing baseline model: {_init_err}")

# ==============================================================================
# 3b. PRETRAINED HTR MODEL (TrOCR Handwritten) - LAZY CACHED LOAD
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
            _GPU_HTR_ERROR = f"HTR backend dependencies not installed: {ie}. Transformers and PyTorch required on Kaggle."
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

gpu_name = "NVIDIA GPU"
try:
    smi_out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode("utf-8")
    gpu_name = smi_out.strip()
except Exception:
    pass

app = Flask(__name__)

@app.route("/", methods=["GET"])
@app.route("/status", methods=["GET"])
def status_endpoint():
    return jsonify({
        "status": "connected",
        "gpu_name": gpu_name,
        "device": DEVICE,
        "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
        "loaded_models": list(_GPU_OCR_MODELS.keys()),
        "htr_model_status": _GPU_HTR_STATUS,
        "htr_source_model": _DEFAULT_HTR_MODEL_NAME if _GPU_HTR_MODEL is not None else None,
    })

@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    # Extract page_number
    page_num = 1
    raw_page = request.form.get("page_number") or request.args.get("page_number")
    if raw_page is not None:
        try:
            page_num = int(raw_page)
        except (ValueError, TypeError):
            page_num = 1

    # Extract lang
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
            "error": "Failed to decode image",
            "page_number": page_num,
            "requested_language": req_lang,
            "ocr_model_language": None,
            "model_status": "ERROR",
            "lines": [],
            "needs_review": True,
            "warnings": ["Failed to decode uploaded image buffer"]
        }), 400

    # Load requested language model - NEVER silently fall back to English
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

    # Run inference with the language-specific model
    with _GPU_OCR_LOCK:
        start_time = time.perf_counter()
        result = model.predict(image)[0]
        ocr_time = (time.perf_counter() - start_time) * 1000

    rec_texts = [str(t) for t in result.get("rec_texts", [])]
    rec_scores = [float(s) for s in result.get("rec_scores", [])]
    rec_polys = result.get("rec_polys", [])
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
        # Backward-compatible fields
        "rec_texts": rec_texts,
        "rec_scores": rec_scores,
        "rec_polys": rec_polys_list,
        "ocr_time_ms": ocr_time,
        "gpu_name": gpu_name,
    })

@app.route("/recognize-handwriting", methods=["POST"])
def recognize_handwriting_endpoint():
    """
    Separate remote endpoint for English handwriting recognition.
    Strict constraints:
    - Never falls back to PaddleOCR for handwriting recognition.
    - If HTR model unavailable or fails, returns MODEL_UNAVAILABLE with warning.
    - Preserves region_id, page_number, confidence, and sets needs_review=True.
    """
    page_num = 1
    raw_page = request.form.get("page_number") or request.args.get("page_number")
    if raw_page is not None:
        try:
            page_num = int(raw_page)
        except (ValueError, TypeError):
            page_num = 1

    region_id = request.form.get("region_id") or request.args.get("region_id") or f"page{page_num}_region1"
    req_lang = (request.form.get("language") or request.args.get("language") or "en").lower().strip()

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

    # Lazily fetch HTR model - never fall back to PaddleOCR!
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

        # Convert OpenCV BGR image to PIL RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        # Run TrOCR inference
        device = next(htr_model.parameters()).device
        pixel_values = htr_processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = htr_model.generate(pixel_values, return_dict_in_generate=True, output_scores=True)
            sequences = generated_ids.sequences
            generated_text = htr_processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()

            # Estimate sequence confidence from token scores if available
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

def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

def main():
    install_dependencies()
    threading.Thread(target=run_flask, daemon=True).start()
    time.sleep(2)
    print("Flask server running on port 5000.")

    public_url = expose_port(5000)
    broadcast_url(public_url, note="Server Ready")

    # ==============================================================================
    # 4. 12-HOUR SUPERVISOR LOOP
    # ==============================================================================
    deadline = time.time() + (MAX_RUNTIME_HOURS * 3600)
    print("\n" + "=" * 65)
    print(" 🚀 KAGGLE 12-HOUR HEADLESS OCR BACKEND IS RUNNING")
    print("=" * 65)
    print(f" • Public Tunnel URL: {public_url}")
    print(f" • Sync Channel:     https://ntfy.sh/{NOTIFY_CHANNEL}")
    print(f" • Planned Runtime:  {MAX_RUNTIME_HOURS} hours")
    print(" • You can close your browser tab or turn off your PC safely!")
    print(f" • Local sync:       ./update_ocr_url.sh --sync")
    print("=" * 65 + "\n")

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
                    st_desc = f"OK ({st.get('gpu_name', 'GPU Online')})"
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
