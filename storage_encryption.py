"""
storage_encryption.py - Authenticated Symmetric Storage Encryption Layer for OneBhoomi

Provides encryption at rest for:
1. Stored scanned document previews (scratch/preview_cache/<id>.html).
2. Database-persisted records containing extracted land-record data (PostgreSQL JSONB/BYTEA).

CRYPTOGRAPHIC SPECIFICATION:
- Uses Fernet from cryptography (authenticated AES-128-CBC + PKCS7 padding + HMAC-SHA256).
- Fail-closed: invalid ciphertext, tampered bytes, or unexpected plaintext strictly raises
  InvalidToken or ValueError. Normal read APIs NEVER silently accept or return plaintext.

STRICT KEY SEPARATION INVARIANT:
- The storage encryption key is stored at verification_keys/storage_encryption.key.
- The RSA-PSS private key (verification_keys/private_key.pem) is used EXCLUSIVELY for
  digital signatures and integrity verification.
- The storage encryption key is NOT derived from the RSA key, signatures, user IDs,
  or passwords.
"""

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet, InvalidToken

ENVELOPE_MAGIC_V1 = b"ONEBHOOMI-ENC-V1:"
JSON_ENVELOPE_KEY = "__onebhoomi_encrypted__"
JSON_ENVELOPE_VERSION = 1

DEFAULT_KEYS_DIR = Path("verification_keys")
DEFAULT_KEY_FILE = DEFAULT_KEYS_DIR / "storage_encryption.key"

_key_lock = threading.Lock()
_cached_fernet: Optional[Fernet] = None
_cached_key: Optional[bytes] = None


def get_storage_encryption_key_path() -> Path:
    """Returns the resolved path to the symmetric storage encryption key file."""
    custom_path = os.environ.get("STORAGE_ENCRYPTION_KEY_PATH")
    if custom_path:
        return Path(custom_path)
    return DEFAULT_KEY_FILE


def ensure_storage_encryption_key() -> Path:
    """
    Ensures that a dedicated symmetric storage encryption key exists.
    If missing, generates a new Fernet key independently using
    Fernet.generate_key() and writes it atomically with restrictive
    filesystem permissions (0o600 on POSIX).
    """
    key_path = get_storage_encryption_key_path()
    if key_path.exists():
        return key_path

    key_path.parent.mkdir(parents=True, exist_ok=True)
    new_key = Fernet.generate_key()

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY

    try:
        fd = os.open(key_path, flags, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(new_key)
    except FileExistsError:
        pass

    return key_path


def get_storage_encryption_key() -> bytes:
    """Retrieves the raw Fernet key, cached in memory."""
    global _cached_key, _cached_fernet
    if _cached_key is not None:
        return _cached_key

    with _key_lock:
        if _cached_key is not None:
            return _cached_key

        key_path = ensure_storage_encryption_key()
        try:
            with open(key_path, "rb") as f:
                raw_key = f.read().strip()
            if not raw_key or len(raw_key) != 44:
                raise ValueError(
                    f"Invalid storage encryption key at {key_path}. Expected 44-byte base64 Fernet key."
                )
            _cached_key = raw_key
            _cached_fernet = Fernet(_cached_key)
            return _cached_key
        except Exception as e:
            raise RuntimeError(f"Failed to load storage encryption key from {key_path}: {e}")


def get_fernet() -> Fernet:
    """Returns the singleton Fernet cipher instance."""
    get_storage_encryption_key()
    assert _cached_fernet is not None
    return _cached_fernet


def reset_key_cache() -> None:
    """Clears the cached key and cipher instance (for tests)."""
    global _cached_key, _cached_fernet
    with _key_lock:
        _cached_key = None
        _cached_fernet = None


# =====================================================================
# Byte Encryption & Decryption (Documents, Blobs, BYTEA)
# =====================================================================

def is_encrypted_bytes(data: bytes) -> bool:
    """Returns True if the binary payload has the authenticated storage envelope marker."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        return False
    return bytes(data).startswith(ENVELOPE_MAGIC_V1)


def encrypt_bytes(data: bytes) -> bytes:
    """
    Encrypts raw bytes using authenticated Fernet cipher with the ONEBHOOMI-ENC-V1 envelope.
    Idempotent: if data is already encrypted, returns it unchanged.
    """
    if not data:
        return b""
    raw_bytes = bytes(data)
    if is_encrypted_bytes(raw_bytes):
        return raw_bytes

    f = get_fernet()
    token = f.encrypt(raw_bytes)
    return ENVELOPE_MAGIC_V1 + token


def decrypt_bytes(ciphertext: bytes) -> bytes:
    """
    Decrypts an authenticated storage envelope payload into original plaintext bytes.
    FAIL-CLOSED: If data is not encrypted or envelope is missing, raises ValueError.
    If HMAC authentication fails or data is tampered, raises InvalidToken.
    NEVER returns plaintext on decryption failure.
    """
    if not ciphertext:
        return b""
    b = bytes(ciphertext)
    if not b.startswith(ENVELOPE_MAGIC_V1):
        raise ValueError(
            "Decryption failed: data is not encrypted or missing ONEBHOOMI-ENC-V1 storage envelope. "
            "Plaintext fallback is forbidden."
        )

    token = b[len(ENVELOPE_MAGIC_V1):]
    f = get_fernet()
    return f.decrypt(token)


# =====================================================================
# JSON Payload Encryption & Decryption (PostgreSQL JSONB columns)
# =====================================================================

def is_encrypted_json(obj: Any) -> bool:
    """Checks if an object is an encrypted JSON envelope dictionary."""
    return isinstance(obj, dict) and obj.get(JSON_ENVELOPE_KEY) is True


def encrypt_json(obj: Any) -> Any:
    """
    Serializes a Python object to JSON, encrypts with Fernet, and wraps in an envelope dict.
    Idempotent: if already an encrypted envelope, returns it unchanged.
    """
    if obj is None:
        return None
    if is_encrypted_json(obj):
        return obj

    def default_serializer(o):
        if isinstance(o, (bytes, bytearray, memoryview)):
            return bytes(o).hex()
        return str(o)

    serialized = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=default_serializer)
    f = get_fernet()
    ciphertext_token = f.encrypt(serialized.encode("utf-8")).decode("ascii")

    return {
        JSON_ENVELOPE_KEY: True,
        "version": JSON_ENVELOPE_VERSION,
        "ciphertext": ciphertext_token,
    }


def decrypt_json(envelope: Any) -> Any:
    """
    Unwraps an encrypted JSON envelope dictionary, decrypts ciphertext, and deserializes.
    FAIL-CLOSED: If envelope is not an encrypted dictionary, raises ValueError.
    If HMAC authentication fails or data is tampered, raises InvalidToken.
    NEVER returns plaintext on decryption failure.
    """
    if envelope is None:
        return None
    if not is_encrypted_json(envelope):
        raise ValueError(
            "Decryption failed: database column does not contain an encrypted JSON envelope. "
            "Plaintext fallback is forbidden."
        )

    ciphertext_token = envelope.get("ciphertext")
    if not ciphertext_token or not isinstance(ciphertext_token, str):
        raise ValueError("Corrupted encrypted JSON envelope: missing ciphertext")

    f = get_fernet()
    try:
        plaintext_bytes = f.decrypt(ciphertext_token.encode("ascii"))
        return json.loads(plaintext_bytes.decode("utf-8"))
    except InvalidToken:
        raise InvalidToken("Decryption failed: invalid or tampered JSON ciphertext")
    except Exception as e:
        raise ValueError(f"Failed to deserialize decrypted JSON payload: {e}")
