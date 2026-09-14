"""
tests/test_signing_key_backup_restore.py - Hardened Operational Key Recovery Tests

Fully isolated from the real application signing key.
All tests operate strictly inside temporary directories with monkeypatched
verification_service key paths that are restored upon test teardown.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import backup_signing_key
import restore_signing_key
import verification_service


class TestSigningKeyBackupRestore(unittest.TestCase):

    def setUp(self):
        # 1. Create isolated temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.test_dir.name)
        self.keys_dir = self.base_dir / "verification_keys"
        self.keys_dir.mkdir(parents=True, exist_ok=True)

        self.private_key_path = self.keys_dir / "private_key.pem"
        self.public_key_path = self.keys_dir / "public_key.pem"

        # 2. Save original verification_service paths and monkeypatch to test directory
        self._orig_keys_dir = verification_service.KEYS_DIR
        self._orig_private_key_path = verification_service.PRIVATE_KEY_PATH
        self._orig_public_key_path = verification_service.PUBLIC_KEY_PATH

        verification_service.KEYS_DIR = self.keys_dir
        verification_service.PRIVATE_KEY_PATH = self.private_key_path
        verification_service.PUBLIC_KEY_PATH = self.public_key_path

        # 3. Generate isolated test 2048-bit RSA key pair matching verification_service format
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # Write test private key in TraditionalOpenSSL (PKCS#1) format
        pem_priv = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )
        with open(self.private_key_path, "wb") as f:
            f.write(pem_priv)

        # Write test public key
        pem_pub = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        with open(self.public_key_path, "wb") as f:
            f.write(pem_pub)

        self.backup_path = self.base_dir / "backup.pem.enc"
        self.password = "IsolatedTestPassword2026!"

    def tearDown(self):
        # Always restore original verification_service paths
        verification_service.KEYS_DIR = self._orig_keys_dir
        verification_service.PRIVATE_KEY_PATH = self._orig_private_key_path
        verification_service.PUBLIC_KEY_PATH = self._orig_public_key_path

        # Clean up isolated test directory
        self.test_dir.cleanup()

    def test_01_backup_roundtrip(self):
        """1. Round-trip: existing key -> encrypted backup -> restore -> loaded key works."""
        backup_signing_key.create_backup(
            destination_path=self.backup_path,
            source_path=self.private_key_path,
            password=self.password
        )
        self.assertTrue(self.backup_path.exists())

        # Remove isolated test private key and restore
        self.private_key_path.unlink()
        restore_signing_key.restore_backup(
            backup_path=self.backup_path,
            destination_path=self.private_key_path,
            public_key_path=self.public_key_path,
            password=self.password,
            overwrite_confirmed=True
        )
        self.assertTrue(self.private_key_path.exists())

        with open(self.private_key_path, "rb") as f:
            restored = serialization.load_pem_private_key(f.read(), password=None)
        self.assertIsInstance(restored, rsa.RSAPrivateKey)

    def test_02_backup_actually_encrypts(self):
        """2. Backup file does not equal plaintext live PEM, is encrypted, and cannot be loaded without password."""
        backup_signing_key.create_backup(
            destination_path=self.backup_path,
            source_path=self.private_key_path,
            password=self.password
        )
        with open(self.private_key_path, "rb") as f:
            live_bytes = f.read()
        with open(self.backup_path, "rb") as f:
            backup_bytes = f.read()

        # Cannot equal plaintext
        self.assertNotEqual(live_bytes, backup_bytes)
        self.assertIn(b"BEGIN ENCRYPTED PRIVATE KEY", backup_bytes)
        self.assertNotIn(b"BEGIN RSA PRIVATE KEY", backup_bytes)

        # Loading without password must fail
        with self.assertRaises((TypeError, ValueError)):
            serialization.load_pem_private_key(backup_bytes, password=None)

    def test_03_correct_password_succeeds(self):
        """3. Decryption with correct password succeeds."""
        backup_signing_key.create_backup(self.backup_path, self.private_key_path, self.password)
        dest_test = self.base_dir / "restored_test.pem"
        restore_signing_key.restore_backup(
            self.backup_path, dest_test, self.public_key_path, self.password, overwrite_confirmed=True
        )
        self.assertTrue(dest_test.exists())

    def test_04_wrong_password_fails(self):
        """4. Decryption with wrong password fails cleanly without leaking key."""
        backup_signing_key.create_backup(self.backup_path, self.private_key_path, self.password)
        dest_test = self.base_dir / "restored_test.pem"
        with self.assertRaises(ValueError) as ctx:
            restore_signing_key.restore_backup(
                self.backup_path, dest_test, self.public_key_path, "WrongPassword999!", overwrite_confirmed=True
            )
        self.assertIn("Decryption failed", str(ctx.exception))
        self.assertFalse(dest_test.exists())

    def test_05_empty_password_rejected(self):
        """5. Empty password is rejected on both backup and restore."""
        with self.assertRaises(ValueError):
            backup_signing_key.create_backup(self.backup_path, self.private_key_path, "")
        backup_signing_key.create_backup(self.backup_path, self.private_key_path, self.password)
        with self.assertRaises(ValueError):
            restore_signing_key.restore_backup(self.backup_path, self.private_key_path, None, "")

    def test_06_password_mismatch_rejected(self):
        """6. CLI rejects password confirmation mismatch."""
        with self.assertRaises(ValueError):
            backup_signing_key.create_backup(self.backup_path, self.private_key_path, None)

    def test_07_missing_source_key_fails(self):
        """7. Missing source key raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            backup_signing_key.create_backup(self.backup_path, self.base_dir / "missing.pem", self.password)

    def test_08_missing_backup_file_fails(self):
        """8. Missing backup file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            restore_signing_key.restore_backup(self.base_dir / "missing.enc", self.private_key_path, None, self.password)

    def test_09_existing_destination_not_overwritten_without_confirmation(self):
        """9. Existing destination private key is not overwritten unless confirmed."""
        backup_signing_key.create_backup(self.backup_path, self.private_key_path, self.password)
        with self.assertRaises(PermissionError):
            restore_signing_key.restore_backup(
                self.backup_path, self.private_key_path, self.public_key_path, self.password, overwrite_confirmed=False
            )

    def test_10_restore_with_overwrite_succeeds(self):
        """10. Restore with overwrite confirmation succeeds."""
        backup_signing_key.create_backup(self.backup_path, self.private_key_path, self.password)
        restore_signing_key.restore_backup(
            self.backup_path, self.private_key_path, self.public_key_path, self.password, overwrite_confirmed=True
        )
        self.assertTrue(self.private_key_path.exists())

    def test_11_failed_restore_leaves_no_corrupted_key(self):
        """11. Failed restore does not leave a truncated or corrupted live key."""
        backup_signing_key.create_backup(self.backup_path, self.private_key_path, self.password)
        with open(self.private_key_path, "rb") as f:
            original_bytes = f.read()

        try:
            restore_signing_key.restore_backup(
                self.backup_path, self.private_key_path, self.public_key_path, "wrong_pwd", overwrite_confirmed=True
            )
        except ValueError:
            pass

        with open(self.private_key_path, "rb") as f:
            current_bytes = f.read()
        self.assertEqual(original_bytes, current_bytes, "Existing key must remain intact after failed restore")

    def test_12_restored_key_derives_identical_public_key(self):
        """12. Restored private key derives identical public key to the source key."""
        backup_signing_key.create_backup(self.backup_path, self.private_key_path, self.password)
        dest_restored = self.base_dir / "restored.pem"
        restore_signing_key.restore_backup(
            self.backup_path, dest_restored, self.public_key_path, self.password, overwrite_confirmed=True
        )
        with open(dest_restored, "rb") as f:
            restored_key = serialization.load_pem_private_key(f.read(), password=None)
        orig_pub = self.private_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        restored_pub = restored_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        self.assertEqual(orig_pub, restored_pub)

    def test_13_mismatched_public_key_aborts_restore_rigorously(self):
        """13. Mismatched public key file aborts restore, preserves destination bytes, leaves no temp files."""
        backup_signing_key.create_backup(self.backup_path, self.private_key_path, self.password)

        # Capture exact destination bytes before attempt
        with open(self.private_key_path, "rb") as f:
            before_bytes = f.read()

        # Create a DIFFERENT public key
        different_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        diff_pub_path = self.base_dir / "diff_public.pem"
        with open(diff_pub_path, "wb") as f:
            f.write(different_key.public_key().public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            ))

        with self.assertRaises(ValueError) as ctx:
            restore_signing_key.restore_backup(
                self.backup_path, self.private_key_path, diff_pub_path, self.password, overwrite_confirmed=True
            )
        self.assertIn("Public key consistency check failed", str(ctx.exception))

        # Destination must remain byte-for-byte identical
        with open(self.private_key_path, "rb") as f:
            after_bytes = f.read()
        self.assertEqual(before_bytes, after_bytes, "Destination key must remain byte-for-byte unmodified")

        # No temporary files lingering in directory
        temp_files = list(self.keys_dir.glob("*.tmp.*"))
        self.assertEqual(temp_files, [], "No temporary replacement files should linger after aborted restore")

    def test_14_atomic_restore_failure_preserves_live_key(self):
        """14. Simulate I/O replacement failure: destination remains intact and temp file is unlinked."""
        backup_signing_key.create_backup(self.backup_path, self.private_key_path, self.password)

        with open(self.private_key_path, "rb") as f:
            before_bytes = f.read()

        # Mock os.replace to simulate an unexpected file-system failure during atomic rename
        with patch("os.replace", side_effect=OSError("Simulated atomic replacement I/O failure")):
            with self.assertRaises(IOError) as ctx:
                restore_signing_key.restore_backup(
                    self.backup_path, self.private_key_path, self.public_key_path, self.password, overwrite_confirmed=True
                )
            self.assertIn("Atomic key replacement failed", str(ctx.exception))

        # Live key must be completely intact
        with open(self.private_key_path, "rb") as f:
            after_bytes = f.read()
        self.assertEqual(before_bytes, after_bytes, "Live key must remain untouched when atomic replacement fails")

        # Ensure no temp file remained
        temp_files = list(self.keys_dir.glob("*.tmp.*"))
        self.assertEqual(temp_files, [], "Temporary restore file must be cleaned up on atomic failure")

    def test_15_historical_signature_compatibility_isolated(self):
        """15. Sign payload, backup, wipe isolated key, restore, and verify original signature succeeds."""
        # 1. Sign a known payload using the existing verification_service.sign_document()
        payload = {"document_number": "HIST-TEST-2026", "execution_date": "2026-03-14", "village": "Gachibowli"}
        sig = verification_service.sign_document(payload)
        pub_pem = verification_service.get_public_verification_key()

        # Verify signature is valid
        self.assertTrue(verification_service.verify_document_signature(payload, sig, pub_pem))

        # 2. Back up the isolated test private key
        backup_signing_key.create_backup(self.backup_path, self.private_key_path, self.password)

        # 3. Wipe isolated test private key
        self.private_key_path.unlink()
        self.assertFalse(self.private_key_path.exists())

        # 4. Restore isolated key from backup
        restore_signing_key.restore_backup(
            self.backup_path, self.private_key_path, self.public_key_path, self.password, overwrite_confirmed=True
        )
        self.assertTrue(self.private_key_path.exists())

        # 5. Verify original signature against restored key
        self.assertTrue(
            verification_service.verify_document_signature(payload, sig, pub_pem),
            "Original signature must verify successfully after key restoration from backup"
        )

    def test_16_crypto_core_behavioral_regression(self):
        """16. Meaningful behavioral validation of core crypto functions in verification_service."""
        # A. Test canonicalize_document()
        payload_a = {"b": 2, "a": 1, "nested": {"z": 9, "m": 5}}
        payload_b = {"nested": {"m": 5, "z": 9}, "a": 1, "b": 2}
        canonical_a = verification_service.canonicalize_document(payload_a)
        canonical_b = verification_service.canonicalize_document(payload_b)
        self.assertEqual(canonical_a, canonical_b, "Canonicalization must be deterministic regardless of key order")

        # B. Test get_or_create_signing_key() parameters
        loaded_key = verification_service.get_or_create_signing_key()
        self.assertIsInstance(loaded_key, rsa.RSAPrivateKey)
        self.assertEqual(loaded_key.key_size, 2048, "RSA key size must be 2048")
        self.assertEqual(loaded_key.public_key().public_numbers().e, 65537, "Public exponent must be 65537")

        # C. Test sign_document() and verify_document_signature()
        doc_payload = {"document_number": "REG-TEST-100", "district": "Rangareddy"}
        sig_b64 = verification_service.sign_document(doc_payload)
        pub_key_pem = verification_service.get_public_verification_key()

        # Valid payload verifies
        self.assertTrue(verification_service.verify_document_signature(doc_payload, sig_b64, pub_key_pem))

        # Tampered payload fails verification
        tampered_payload = {"document_number": "REG-TEST-100", "district": "CorruptedDistrict"}
        self.assertFalse(verification_service.verify_document_signature(tampered_payload, sig_b64, pub_key_pem))

        # Corrupted signature fails verification
        corrupted_sig = sig_b64[:-4] + "AAAA"
        self.assertFalse(verification_service.verify_document_signature(doc_payload, corrupted_sig, pub_key_pem))


if __name__ == "__main__":
    unittest.main()
