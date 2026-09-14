# OneBhoomi RSA Signing Key Backup & Recovery Runbook

## 1. Purpose of the Signing Private Key
The OneBhoomi digital land record platform uses an asymmetric RSA private key to cryptographically seal approved land records. Each approved record receives an RSA-PSS / SHA-256 digital signature over its canonical payload. The matching public key verifies authenticity across government registries.

## 2. Live Key Custody Location
- **Private Key**: `verification_keys/private_key.pem` (mode `0600`)
- **Public Verification Key**: `verification_keys/public_key.pem`

The live signing, verification, and canonicalization mechanisms are completely unchanged. The active private key remains in local filesystem custody on the registry host.

## 3. When Backups Must Be Made
An encrypted backup must be executed:
1. **Immediately after initial key generation** during system commissioning.
2. **Immediately after every key rotation** ceremony.
3. **After verifying that the backup is usable** (e.g. testing recovery into an isolated staging environment).

## 4. Recommended Backup Storage
- **Operator-controlled encrypted USB storage**: Kept in physical custody or a registry fireproof safe.
- **Protected secondary offline storage**: Dedicated access-restricted backup share.
- **Air-gapped physical media**: Disconnected from the network.

## 5. Security Rules
- **NEVER commit private keys or encrypted backups to Git**: `.gitignore` strictly blocks `verification_keys/`, `*.pem.enc`, `*.key.enc`, and `backups/`.
- **NEVER store the backup password in the repository or alongside the backup file**: Maintain strict dual-custody (the encrypted backup file and its password must be stored separately).
- **Restrict filesystem permissions**: Always maintain `0600` permissions on key files.
- **NEVER transmit plaintext key material**: Never email keys, post keys into issue trackers, or paste key contents into chat or AI interfaces.

## 6. Backup Procedure
To export the live private key to an encrypted backup:
```bash
python backup_signing_key.py /path/to/secure-storage/onebhoomi-signing-key.pem.enc
```
1. The utility prompts interactively for a password without echoing (`getpass`).
2. Password confirmation is required.
3. `cryptography`'s `serialization.BestAvailableEncryption` is used to create a password-protected encrypted PKCS#8 private-key backup in PEM format.
4. The live key is never modified during backup.

## 7. Restore Procedures

### A. Restoring the Current Active Key
To restore the current active private key into `verification_keys/private_key.pem`:
```bash
python restore_signing_key.py /path/to/secure-storage/onebhoomi-signing-key.pem.enc
```
1. The operator is prompted for the decryption password without echoing.
2. **Public Key Consistency Check**: The tool derives the public key from the backup and verifies byte-for-byte identity against `verification_keys/public_key.pem`. If the keys do not match, the restore aborts immediately.
3. **Overwrite Confirmation**: If `verification_keys/private_key.pem` already exists, the utility halts and requires the operator to type `OVERWRITE` explicitly.
4. **Atomic Restoration**: The key is written to a temporary sibling file, flushed, synced (`fsync`), and atomically moved into place using `os.replace`. A failed restore will never leave a truncated or corrupted file.

### B. Historical / Previous Key Recovery
After a legitimate key rotation, an archived private-key backup corresponds to its archived historical public key, NOT necessarily the current active public key.

To recover an archived historical key for offline audits, re-verification, or inspection:
1. **Identify the historical keypair**: Locate the archived backup file and the matching historical public key.
2. **Provide matching historical public key**: Specify the historical public key path using `--public-key` and restore into an isolated staging location using `--dest`:
   ```bash
   python restore_signing_key.py /path/to/archive/key-2023.pem.enc \
       --public-key /path/to/archive/pubkey-2023.pem \
       --dest /path/to/isolated-staging/historical_private_key.pem
   ```
3. **Do not overwrite the current public key**: The public-key consistency check is strictly an **identity check**, not a key-rotation mechanism. The utility will **never** automatically modify `verification_keys/public_key.pem`.
4. **Positive identification required**: Only restore historical keys when the operator has positively confirmed the intended keypair.

## 8. Critical Warning: Consequence of Key Loss
> [!CAUTION]
> **Permanent Loss of Private Key**:
> If the original signing private key is permanently lost and there is no valid backup, **records already sealed with that key cannot be cryptographically verified using a newly generated replacement key**.
>
> Generating a replacement key allows signing *future* records, but it **DOES NOT** recover or validate historical signatures. Historical certificates validated against the old public key will fail verification if the private key or matching public key is lost.

## 9. Key Rotation Guidance
1. When rotating keys, the **old private key and old public key must be securely archived**, never destroyed, so historical document signatures can still be verified against historical records.
2. The newly generated keypair must be backed up immediately following the procedure above.
3. A new key must never be assumed capable of validating old signatures.
