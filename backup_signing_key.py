#!/usr/bin/env python3
"""
backup_signing_key.py - OneBhoomi RSA Signing Private Key Backup Utility

Exports the live RSA signing private key into an encrypted, password-protected
PEM backup file using the standard PKCS#8 BestAvailableEncryption primitive.

Usage:
    python backup_signing_key.py <destination_path> [--source <source_path>]
"""

import argparse
import getpass
import os
import sys
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

DEFAULT_SOURCE_PATH = Path("verification_keys") / "private_key.pem"


def create_backup(
    destination_path: Path,
    source_path: Path = DEFAULT_SOURCE_PATH,
    password: Optional[str] = None
) -> None:
    """
    Reads the live private key and writes an encrypted PKCS#8 PEM backup.
    Does NOT modify the live key.
    """
    src = Path(source_path)
    dst = Path(destination_path)

    if not src.exists():
        raise FileNotFoundError(f"Source private key not found at '{src}'.")

    if not src.is_file():
        raise ValueError(f"Source path '{src}' is not a regular file.")

    # 1. Read live private key without modifying it
    try:
        with open(src, "rb") as f:
            key_data = f.read()
    except Exception as e:
        raise IOError(f"Failed to read source key: {e}")

    # 2. Parse live private key (unencrypted PEM)
    try:
        private_key = serialization.load_pem_private_key(key_data, password=None)
    except Exception:
        raise ValueError("Failed to parse source private key. Ensure it is a valid PEM private key.")

    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise ValueError("Source key is not an RSA private key.")

    # 3. Validate password
    if not password:
        raise ValueError("Backup encryption password cannot be empty.")

    # 4. Encrypt private key using cryptography's PKCS#8 BestAvailableEncryption
    try:
        encrypted_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(password.encode("utf-8"))
        )
    except Exception as e:
        raise RuntimeError(f"Encryption failed: {e}")

    # 5. Ensure parent destination directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # 6. Write backup file with restricted permissions (0o600 where supported)
    try:
        if hasattr(os, "O_WRONLY") and hasattr(os, "O_CREAT"):
            fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(encrypted_pem)
                f.flush()
                if hasattr(os, "fsync"):
                    os.fsync(f.fileno())
        else:
            with open(dst, "wb") as f:
                f.write(encrypted_pem)
                f.flush()
    except Exception as e:
        raise IOError(f"Failed to write encrypted backup to '{dst}': {e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export OneBhoomi RSA signing private key to an encrypted backup file."
    )
    parser.add_argument(
        "destination",
        help="Destination path for the encrypted backup file (e.g. /path/to/backup.pem.enc)"
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE_PATH),
        help=f"Source private key path (default: {DEFAULT_SOURCE_PATH})"
    )
    parser.add_argument(
        "--password-env",
        default=None,
        help="Environment variable name to read backup password from (for automation/tests)"
    )

    args = parser.parse_args()

    if args.password_env and os.getenv(args.password_env):
        pwd = os.environ[args.password_env]
    else:
        try:
            pwd = getpass.getpass("Enter backup encryption password: ")
            if not pwd:
                print("Error: Password cannot be empty.", file=sys.stderr)
                return 1
            confirm = getpass.getpass("Confirm backup encryption password: ")
            if pwd != confirm:
                print("Error: Passwords do not match.", file=sys.stderr)
                return 1
        except (KeyboardInterrupt, EOFError):
            print("\nBackup cancelled by user.", file=sys.stderr)
            return 1

    try:
        create_backup(destination_path=Path(args.destination), source_path=Path(args.source), password=pwd)
        print(f"Success: Encrypted backup created at '{args.destination}'.")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
