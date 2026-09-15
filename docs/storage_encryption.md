# OneBhoomi — Storage Encryption Architecture & Operations Guide

## 1. Cryptographic Key Separation

OneBhoomi strictly separates cryptographic keys based on their distinct security objectives:

| Key Attribute | RSA-PSS Signing Key | Storage Encryption Key |
| :--- | :--- | :--- |
| **File Path** | `verification_keys/private_key.pem` | `verification_keys/storage_encryption.key` |
| **Cryptographic Type** | Asymmetric RSA (2048-bit) | Symmetric Fernet (AES-128-CBC + HMAC-SHA256) |
| **Security Objective** | **Authenticity & Integrity** | **Confidentiality at Rest** |
| **Used By** | `verification_service.py` (signing & verification) | `storage_encryption.py` (file & database crypto) |
| **Compromise Impact** | Digital signatures cannot be trusted | Data confidentiality at rest is breached |
| **Loss Impact** | Cannot issue new signed records | Cannot decrypt stored documents or records |

> [!IMPORTANT]
> The storage encryption key is completely independent. It is **NEVER** derived from the RSA private key, public key, signatures, user IDs, or passwords.

---

## 2. Invariant: RSA-PSS Signing Input is Unchanged

Digital signatures are calculated exclusively over the canonical plaintext logical record:

```text
Logical Record
      │
      ▼
Canonicalization (canonicalize_document)
      │
      ▼
Canonical Plaintext Bytes
      ├──────────────────────────────► RSA-PSS ──► Digital Signature
      │
      ▼ (Storage Boundary Only)
Storage Encryption Layer (Fernet)
      │
      ▼
PostgreSQL & Document Disk Ciphertext
```

The digital signature represents the authenticity of the canonical logical record and **NEVER** signs ciphertext.

---

## 3. Database Columns: Encrypted vs. Plaintext Searchable Metadata

### Encrypted Columns (`verification_records`)
- `document_payload` (`JSONB`): Encrypted envelope `{"__onebhoomi_encrypted__": true, "version": 1, "ciphertext": "..."}`.
- `raw_ocr` (`JSONB`): Encrypted envelope.
- `field_provenance` (`JSONB`): Encrypted envelope.
- `neural_nlp` (`JSONB`): Encrypted envelope.
- `canonical_sealed_payload` (`BYTEA`): Encrypted with `ONEBHOOMI-ENC-V1:` envelope.

### Plaintext Searchable & Relational Metadata
The following columns remain plaintext for SQL indexing, joins, and filtering:
- `verification_id` (PRIMARY KEY)
- `status`, `is_land_document`, `file_hash`, `filename`, `uploaded_by_user_id`, `clerk_submitted`
- `created_at`, `submitted_at`, `approved_at`
- `signature`, `public_key`, `qr_code`
- `decision`, `populated_fields`, `checks`, `duplicate_info`

---

## 4. Document File Storage & Preview Cache

Original raw scan uploads are processed in transient memory and temporary files (`NamedTemporaryFile`), which are unlinked immediately after OCR execution (`web_app.py:5832`).
The persistent artifact holding document scan data is `scratch/preview_cache/<id>.html`.
This preview cache is encrypted at rest using the `ONEBHOOMI-ENC-V1:` envelope.

---

## 5. Fail-Closed Read Invariant

Normal persistent-storage reads (`storage_encryption.decrypt_bytes` and `storage_encryption.decrypt_json`) strictly enforce envelope markers:
- Files missing `ONEBHOOMI-ENC-V1:` raise `ValueError`.
- Database JSON columns missing `__onebhoomi_encrypted__` raise `ValueError`.
- Tampered or invalid ciphertext raises `cryptography.fernet.InvalidToken`.
- **No silent fallback to plaintext is permitted.**

---

## 6. Migration Operations

To migrate existing plaintext data in-place:
```bash
# Audit / Dry-run:
python migrate_storage_to_encryption.py --dry-run

# Execute in-place migration:
python migrate_storage_to_encryption.py
```

- **Idempotency:** Re-running migration safely skips already encrypted items.
- **Atomicity:** PostgreSQL migration executes inside a single transaction and rolls back entirely on any error. Files are replaced atomically using `os.replace`.
