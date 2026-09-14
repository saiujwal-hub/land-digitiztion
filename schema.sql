-- schema.sql
-- OneBhoomi PostgreSQL Schema

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(32) NOT NULL,
    assigned_officer_id VARCHAR(64),
    identities JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_identities ON users USING GIN (identities);

-- 2. Auth Credentials Table
CREATE TABLE IF NOT EXISTS auth_credentials (
    identity_key VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_auth_creds_user ON auth_credentials(user_id);

-- 3. Sessions Table
CREATE TABLE IF NOT EXISTS sessions (
    session_token VARCHAR(128) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

-- 4. Verification Records Table
CREATE TABLE IF NOT EXISTS verification_records (
    verification_id VARCHAR(64) PRIMARY KEY,
    status VARCHAR(32) NOT NULL,
    is_land_document BOOLEAN NOT NULL DEFAULT FALSE,
    file_hash VARCHAR(128),
    filename VARCHAR(512),
    uploaded_by_user_id VARCHAR(64),
    clerk_submitted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TEXT NOT NULL,
    submitted_at TEXT,
    approved_at TEXT,
    signature TEXT,
    public_key TEXT,
    qr_code TEXT,
    decision JSONB,
    populated_fields JSONB NOT NULL DEFAULT '[]'::jsonb,
    document_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    canonical_sealed_payload BYTEA,
    raw_ocr JSONB,
    field_provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    checks JSONB NOT NULL DEFAULT '[]'::jsonb,
    duplicate_info JSONB,
    neural_nlp JSONB
);

CREATE INDEX IF NOT EXISTS idx_verif_status ON verification_records(status);
CREATE INDEX IF NOT EXISTS idx_verif_file_hash ON verification_records(file_hash);
CREATE INDEX IF NOT EXISTS idx_verif_uploaded_by ON verification_records(uploaded_by_user_id);
CREATE INDEX IF NOT EXISTS idx_verif_created_at ON verification_records(created_at);
CREATE INDEX IF NOT EXISTS idx_verif_doc_payload ON verification_records USING GIN (document_payload);
