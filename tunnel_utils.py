from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import requests


_CLOUDFLARED_BINARY_URLS = {
    "x86_64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    "amd64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    "aarch64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
    "arm64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
}


def _stream_until_url(proc: subprocess.Popen[str], patterns: list[re.Pattern[str]], timeout_s: int = 120) -> str:
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


def _start_localtunnel(port: int) -> str:
    lt_cmd = None
    npx_cmd = shutil.which("npx.cmd") or shutil.which("npx")
    lt_cmd_bin = shutil.which("lt.cmd") or shutil.which("lt")
    if npx_cmd:
        lt_cmd = [npx_cmd, "-y", "localtunnel", "--port", str(port)]
    elif lt_cmd_bin:
        lt_cmd = [lt_cmd_bin, "--port", str(port)]

    if not lt_cmd:
        raise FileNotFoundError("Neither 'npx' nor 'lt' is available for localtunnel.")

    print("Starting localtunnel...")
    proc = subprocess.Popen(
        lt_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return _stream_until_url(
        proc,
        [
            re.compile(r"https://[^\s]+(?:localtunnel\.me|loca\.lt)[^\s]*", re.IGNORECASE),
        ],
    )


def _download_cloudflared_binary() -> str:
    machine = platform.machine().lower()
    asset_url = _CLOUDFLARED_BINARY_URLS.get(machine)
    if not asset_url:
        raise RuntimeError(f"Unsupported Kaggle architecture for cloudflared: {machine}")

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


def _start_cloudflared(port: int) -> str:
    binary = shutil.which("cloudflared")
    if not binary:
        binary = _download_cloudflared_binary()

    print("Starting cloudflared quick tunnel...")
    proc = subprocess.Popen(
        [binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return _stream_until_url(
        proc,
        [
            re.compile(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", re.IGNORECASE),
        ],
    )


def expose_port(port: int, prefer_kaggle: bool | None = None) -> str:
    kaggle_env = any(
        key in os.environ
        for key in (
            "KAGGLE_KERNEL_RUN_TYPE",
            "KAGGLE_URL_BASE",
            "KAGGLE_WORKING_DIR",
        )
    )
    prefer_kaggle = kaggle_env if prefer_kaggle is None else prefer_kaggle

    attempts = [_start_cloudflared, _start_localtunnel] if prefer_kaggle else [_start_localtunnel, _start_cloudflared]
    errors: list[str] = []

    for starter in attempts:
        try:
            url = starter(port)
            print(f"\nPublic OCR URL: {url}")
            return url
        except Exception as exc:
            errors.append(f"{starter.__name__}: {exc}")
            print(f"{starter.__name__} failed: {exc}")

    raise RuntimeError("Unable to create a public tunnel. " + " | ".join(errors))
