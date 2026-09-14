#!/usr/bin/env python3
"""
restore_signing_key.py - OneBhoomi RSA Signing Private Key Restore Utility

Restores the RSA private signing key from an encrypted backup into
verification_keys/private_key.pem with atomic replacement, public-key consistency
validation, and explicit overwrite confirmation.

Usage:
    python restore_signing_key.py <backup_path> [--dest <destination_path>] [--public-key <pubkey_path>] [--overwrite]
"""

import argparse
import getpass
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

DEFAULT_DEST_PATH = Path("verification_keys") / "private_key.pem"
DEFAULT_PUBLIC_KEY_PATH = Path("verification_keys") / "public_key.pem"


def restore_backup(
    backup_path: Path,
    destination_path: Path = DEFAULT_DEST_PATH,
    public_key_path: Optional[Path] = DEFAULT_PUBLIC_KEY_PATH,
    password: Optional[str] = None,
    overwrite_confirmed: bool = False
) -> None:
    """
    Decrypts the backup private key, validates public-key consistency against
    the installed public verification key, and atomically restores the live key.
    """
    if not password:
        raise ValueError("Decryption password cannot be empty.")

    b_path = Path(backup_path)
    dst = Path(destination_path)

    if not b_path.exists():
        raise FileNotFoundError(f"Backup file not found at '{b_path}'.")

    if not b_path.is_file():
        raise ValueError(f"Backup path '{b_path}' is not a regular file.")

    # 1. Read encrypted backup file
    try:
        with open(b_path, "rb") as f:
            encrypted_data = f.read()
    except Exception as e:
        raise IOError(f"Failed to read backup file: {e}")

    # 2. Decrypt and load private key
    try:
        private_key = serialization.load_pem_private_key(
            encrypted_data,
            password=password.encode("utf-8")
        )
    except Exception:
        raise ValueError("Decryption failed: incorrect password or corrupted backup file.")

    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise ValueError("Restored key is not an RSA private key.")

    # 3. Derive canonical SubjectPublicKeyInfo PEM bytes from restored private key
    derived_public_key = private_key.public_key()
    derived_pub_bytes = derived_public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    # 4. Consistency check against existing public verification key if present
    if public_key_path:
        pub_file = Path(public_key_path)
        if pub_file.exists():
            try:
                with open(pub_file, "rb") as f:
                    installed_pub_data = f.read()
                installed_pub = serialization.load_pem_public_key(installed_pub_data)
                installed_pub_bytes = installed_pub.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
            except Exception:
                raise ValueError(f"Failed to parse existing public verification key at '{pub_file}'.")

            if derived_pub_bytes != installed_pub_bytes:
                raise ValueError(
                    "Public key consistency check failed: The restored private key does NOT "
                    f"match the currently installed public verification key at '{pub_file}'. "
                    "Restoration aborted to prevent invalidating existing signatures."
                )

    # 5. Check if destination already exists
    if dst.exists() and not overwrite_confirmed:
        raise PermissionError(
            f"Destination private key '{dst}' already exists and overwrite was not confirmed."
        )

    # 6. Format restored private key in TraditionalOpenSSL (PKCS#1) format matching live key
    pem_private = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    )

    # 7. Atomic write: write to temporary sibling file in same directory
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dst.parent / f"{dst.name}.tmp.{uuid.uuid4().hex[:8]}"

    try:
        if hasattr(os, "O_WRONLY") and hasattr(os, "O_CREAT"):
            fd = os.open(temp_dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(pem_private)
                f.flush()
                if hasattr(os, "fsync"):
                    os.fsync(f.fileno())
        else:
            with open(temp_dest, "wb") as f:
                f.write(pem_private)
                f.flush()

        # Atomically replace destination
        os.replace(temp_dest, dst)
    except Exception as e:
        if temp_dest.exists():
            try:
                temp_dest.unlink()
            except Exception:
                pass
        raise IOError(f"Atomic key replacement failed: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore OneBhoomi RSA signing private key from an encrypted backup."
    )
    parser.add_argument(
        "backup",
        help="Path to encrypted backup file (e.g. /path/to/backup.pem.enc)"
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST_PATH),
        help=f"Destination private key path (default: {DEFAULT_DEST_PATH})"
    )
    parser.add_argument(
        "--public-key",
        default=str(DEFAULT_PUBLIC_KEY_PATH),
        help=f"Public verification key path for consistency check (default: {DEFAULT_PUBLIC_KEY_PATH})"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow overwriting existing destination private key"
    )
    parser.add_argument(
        "--password-env",
        default=None,
        help="Environment variable name to read decryption password from"
    )

    args = parser.parse_args()
    dst = Path(args.dest)

    overwrite = args.overwrite
    if dst.exists() and not overwrite:
        print(f"WARNING: Destination private key '{dst}' already exists!", file=sys.stderr)
        print("Overwriting will replace the live signing key.", file=sys.stderr)
        try:
            confirmation = input("Type 'OVERWRITE' to proceed: ").strip()
            if confirmation != "OVERWRITE":
                print("Restore cancelled: key was NOT overwritten.", file=sys.stderr)
                return 1
            overwrite = True
        except (KeyboardInterrupt, EOFError):
            print("\nRestore cancelled.", file=sys.stderr)
            return 1

    if args.password_env and os.getenv(args.password_env):
        pwd = os.environ[args.password_env]
    else:
        try:
            pwd = getpass.getpass("Enter backup decryption password: ")
            if not pwd:
                print("Error: Password cannot be empty.", file=sys.stderr)
                return 1
        except (KeyboardInterrupt, EOFError):
            print("\nRestore cancelled by user.", file=sys.stderr)
            return 1

    try:
        restore_backup(
            backup_path=Path(args.backup),
            destination_path=dst,
            public_key_path=Path(args.public_key) if args.public_key else None,
            password=pwd,
            overwrite_confirmed=overwrite
        )
        print(f"Success: Private signing key restored to '{dst}'.")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
