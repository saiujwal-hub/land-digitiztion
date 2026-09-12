import sys
import os
_venv_site = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Lib", "site-packages")
if os.path.exists(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

import html
import json
import logging
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

import accounts_store

logger = logging.getLogger(__name__)

# Credentials Persistence Layer (auth_credentials.json)
AUTH_CREDS_PATH = Path(os.environ.get("AUTH_CREDS_PATH", "auth_credentials.json"))
_creds_lock = threading.RLock()

# Argon2 password hasher (memory-hard, resistant to GPU attacks)
_hasher = PasswordHasher(
    time_cost=2,
    memory_cost=65536,
    parallelism=1,
    hash_len=32,
    salt_len=16
)

# ---------------------------------------------------------------------
# Feature Flags & Provider Configurations
# ---------------------------------------------------------------------
# Phone OTP feature flag: OFF by default
ENABLE_PHONE_OTP = os.environ.get("ENABLE_PHONE_OTP", "false").lower() in ("true", "1", "yes")

# SMS Provider Credentials (Twilio / MSG91)
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")

MSG91_AUTH_KEY = os.environ.get("MSG91_AUTH_KEY", "")
MSG91_SENDER_ID = os.environ.get("MSG91_SENDER_ID", "ONEBHO")
MSG91_TEMPLATE_ID = os.environ.get("MSG91_TEMPLATE_ID", "")

# In-memory transient OTP store: phone -> {"otp": "...", "expires_at": float, "attempts": int}
_otp_store: Dict[str, Dict[str, Any]] = {}
_otp_lock = threading.Lock()
OTP_EXPIRATION_SECONDS = 300  # 5 minutes

# Google OAuth Credentials
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")


# =====================================================================
# Credentials Storage (Argon2 Hashed)
# =====================================================================

def load_credentials_db() -> Dict[str, Dict[str, Any]]:
    """Loads hashed credentials from auth_credentials.json."""
    with _creds_lock:
        if not AUTH_CREDS_PATH.exists():
            return {}
        try:
            with open(AUTH_CREDS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def save_credentials_db(creds: Dict[str, Dict[str, Any]]) -> None:
    """Saves hashed credentials safely to auth_credentials.json."""
    with _creds_lock:
        tmp_path = AUTH_CREDS_PATH.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(creds, f, indent=2, ensure_ascii=False)
            tmp_path.replace(AUTH_CREDS_PATH)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed to persist credentials: {e}")


# =====================================================================
# Method 1: Email + Password Authentication
# =====================================================================

def hash_password(plain_password: str) -> str:
    """Hashes a password with Argon2 (never plain text)."""
    return _hasher.hash(plain_password)


def verify_password(hashed: str, plain_password: str) -> bool:
    """Verifies a plain password against an Argon2 hash."""
    try:
        return _hasher.verify(hashed, plain_password)
    except (VerifyMismatchError, Exception):
        return False


def register_email_password(
    email: str,
    password: str,
    name: str,
    role: str,
) -> Tuple[bool, str, Optional[Dict[str, Any]], Optional[str]]:
    """
    Registers a new user with email + password and selected role ('clerk' or 'officer').
    Returns (success, message, user_dict, session_token).
    """
    email_clean = (email or "").strip().lower()
    if not email_clean or "@" not in email_clean:
        return False, "Please enter a valid email address.", None, None

    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long.", None, None

    role_clean = (role or "").strip().lower()
    if role_clean not in accounts_store.VALID_ROLES:
        return False, f"Invalid role '{role}'. Must be either 'clerk' or 'officer'.", None, None

    name_clean = (name or "").strip() or email_clean.split("@")[0].title()

    # Check if user already exists with this email
    existing_user = accounts_store.get_user_by_identity("email", email_clean)
    if existing_user:
        return False, "An account with this email already exists. Please sign in.", None, None

    # Check credentials db
    creds_key = f"email:{email_clean}"
    creds = load_credentials_db()
    if creds_key in creds:
        return False, "Credentials already registered for this email.", None, None

    # Create user in accounts_store
    identities = [{"type": "email", "identifier": email_clean}]
    user = accounts_store.create_user(
        name=name_clean,
        role=role_clean,
        identities=identities,
    )

    # Hash and save password
    password_hash = hash_password(password)
    creds[creds_key] = {
        "user_id": user["user_id"],
        "password_hash": password_hash,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    save_credentials_db(creds)

    # Issue session
    session = accounts_store.create_session(user["user_id"])
    return True, "Registration successful.", user, session["session_token"]


def signin_email_password(
    email: str,
    password: str,
) -> Tuple[bool, str, Optional[Dict[str, Any]], Optional[str]]:
    """
    Authenticates an existing user with email + password.
    Returns (success, message, user_dict, session_token).
    """
    email_clean = (email or "").strip().lower()
    if not email_clean or not password:
        return False, "Email and password are required.", None, None

    creds_key = f"email:{email_clean}"
    creds = load_credentials_db()
    cred_entry = creds.get(creds_key)

    if not cred_entry or "password_hash" not in cred_entry:
        return False, "Invalid email or password.", None, None

    if not verify_password(cred_entry["password_hash"], password):
        return False, "Invalid email or password.", None, None

    user = accounts_store.get_user(cred_entry["user_id"])
    if not user:
        return False, "Associated user account was not found.", None, None

    session = accounts_store.create_session(user["user_id"])
    return True, "Login successful.", user, session["session_token"]


# =====================================================================
# Method 2: Phone Number + OTP Authentication (Behind Feature Flag)
# =====================================================================

def is_phone_otp_enabled() -> bool:
    """Returns True if Phone OTP feature is enabled and configured."""
    return ENABLE_PHONE_OTP


def get_phone_provider_status() -> Dict[str, Any]:
    """
    Returns diagnostics on the phone OTP provider configuration.
    Explains which credentials are required if disabled or unconfigured.
    """
    has_twilio = bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER)
    has_msg91 = bool(MSG91_AUTH_KEY)

    return {
        "enabled": ENABLE_PHONE_OTP,
        "active_provider": "twilio" if has_twilio else ("msg91" if has_msg91 else None),
        "required_credentials": {
            "Twilio": ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER"],
            "MSG91": ["MSG91_AUTH_KEY", "MSG91_TEMPLATE_ID"],
        },
        "instruction": (
            "To enable Phone OTP, set environment variable ENABLE_PHONE_OTP=true and supply "
            "either Twilio (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER) or "
            "MSG91 (MSG91_AUTH_KEY) credentials."
        ) if not ENABLE_PHONE_OTP else "Phone OTP is active.",
    }


def send_sms_via_twilio(to_phone: str, message: str) -> Tuple[bool, str]:
    """Sends an SMS message using Twilio REST API."""
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER):
        return False, "Twilio credentials are incomplete (need SID, Auth Token, and Twilio Number)."

    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    data = {
        "From": TWILIO_PHONE_NUMBER,
        "To": to_phone,
        "Body": message,
    }
    try:
        resp = requests.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=10)
        if resp.status_code in (200, 201):
            return True, "SMS sent successfully."
        err_msg = resp.json().get("message", resp.text)
        return False, f"Twilio error: {err_msg}"
    except Exception as e:
        return False, f"Failed to dispatch SMS: {e}"


def send_sms_via_msg91(to_phone: str, otp: str) -> Tuple[bool, str]:
    """Sends an OTP using MSG91 API."""
    if not MSG91_AUTH_KEY:
        return False, "MSG91_AUTH_KEY is not configured."

    url = "https://api.msg91.com/api/v5/otp"
    payload = {
        "template_id": MSG91_TEMPLATE_ID,
        "mobile": to_phone.replace("+", ""),
        "authkey": MSG91_AUTH_KEY,
        "otp": otp,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return True, "OTP dispatched via MSG91."
        return False, f"MSG91 error: {resp.text}"
    except Exception as e:
        return False, f"Failed to dispatch MSG91 OTP: {e}"


def request_phone_otp(phone: str) -> Tuple[bool, str]:
    """
    Dispatches a 6-digit OTP to the phone number.
    If feature flag is disabled, returns an informative message.
    """
    clean_phone = re.sub(r"[^\d+]", "", phone.strip())
    if len(clean_phone) < 10:
        return False, "Please enter a valid 10-digit phone number."

    if not ENABLE_PHONE_OTP:
        status = get_phone_provider_status()
        return False, f"Phone OTP is currently disabled in demo mode. {status['instruction']}"

    otp = f"{secrets.randbelow(900000) + 100000}"
    expires_at = time.time() + OTP_EXPIRATION_SECONDS

    with _otp_lock:
        _otp_store[clean_phone] = {
            "otp": otp,
            "expires_at": expires_at,
            "attempts": 0,
        }

    message = f"Your OneBhoomi Verification OTP is {otp}. Valid for 5 minutes."

    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        sent, err = send_sms_via_twilio(clean_phone, message)
        if sent:
            return True, "OTP sent successfully to your phone."
        return False, err
    elif MSG91_AUTH_KEY:
        sent, err = send_sms_via_msg91(clean_phone, otp)
        if sent:
            return True, "OTP sent successfully to your phone."
        return False, err
    else:
        return False, "No SMS provider configured. Set TWILIO_* or MSG91_* credentials."


def verify_phone_otp_and_login(
    phone: str,
    otp: str,
    name: Optional[str] = None,
    role: Optional[str] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]], Optional[str], bool]:
    """
    Verifies phone OTP and logs user in.
    If user doesn't exist yet and role is missing, returns needs_role_selection=True.
    Returns (success, message, user_dict, session_token, needs_role_selection).
    """
    clean_phone = re.sub(r"[^\d+]", "", phone.strip())
    clean_otp = otp.strip()

    if not clean_phone or not clean_otp:
        return False, "Phone number and OTP are required.", None, None, False

    if not ENABLE_PHONE_OTP:
        return False, "Phone OTP is currently disabled.", None, None, False

    with _otp_lock:
        entry = _otp_store.get(clean_phone)
        if not entry:
            return False, "No OTP requested for this phone number. Please request an OTP first.", None, None, False

        if time.time() > entry["expires_at"]:
            del _otp_store[clean_phone]
            return False, "OTP has expired. Please request a new one.", None, None, False

        entry["attempts"] += 1
        if entry["attempts"] > 4:
            del _otp_store[clean_phone]
            return False, "Too many failed attempts. Please request a new OTP.", None, None, False

        if entry["otp"] != clean_otp:
            return False, "Invalid OTP. Please check and try again.", None, None, False

        del _otp_store[clean_phone]

    # Look up existing user by phone
    user = accounts_store.get_user_by_identity("phone", clean_phone)

    if user:
        session = accounts_store.create_session(user["user_id"])
        return True, "Phone verification successful.", user, session["session_token"], False

    # First sign-in: require role selection
    if not role or role.strip().lower() not in accounts_store.VALID_ROLES:
        return True, "OTP verified! Please select your role to finalize account creation.", None, None, True

    user_name = (name or "").strip() or f"User {clean_phone[-4:]}"
    user = accounts_store.create_user(
        name=user_name,
        role=role.strip().lower(),
        identities=[{"type": "phone", "identifier": clean_phone}],
    )
    session = accounts_store.create_session(user["user_id"])
    return True, "Account created successfully.", user, session["session_token"], False


# =====================================================================
# Method 3: Google OAuth Sign-in (google-auth & google-auth-oauthlib)
# =====================================================================

def is_google_auth_configured() -> bool:
    """Returns True if Google OAuth Client ID and Secret are configured."""
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)


def get_google_oauth_status() -> Dict[str, Any]:
    """Returns diagnostics on Google OAuth configuration."""
    return {
        "configured": is_google_auth_configured(),
        "client_id_set": bool(GOOGLE_CLIENT_ID),
        "client_secret_set": bool(GOOGLE_CLIENT_SECRET),
        "instruction": (
            "To activate Google Sign-in, set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET "
            "environment variables from your Google Cloud Console OAuth 2.0 Client credentials."
        ) if not is_google_auth_configured() else "Google OAuth is ready.",
    }


def verify_google_id_token(id_token_str: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Verifies a Google ID token using Google's official google-auth library:
    google.oauth2.id_token.verify_oauth2_token.
    """
    if not GOOGLE_CLIENT_ID:
        return False, "Google OAuth is not configured (missing GOOGLE_CLIENT_ID).", None

    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests

        req = google_requests.Request()
        id_info = id_token.verify_oauth2_token(id_token_str, req, GOOGLE_CLIENT_ID)

        if id_info.get("iss") not in ["accounts.google.com", "https://accounts.google.com"]:
            return False, "Invalid token issuer.", None

        return True, "Token verified successfully.", id_info
    except Exception as e:
        return False, f"Google token verification failed: {e}", None


def get_google_auth_url(redirect_uri: str, state: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Generates standard Google OAuth 2.0 authorization URL using google_auth_oauthlib.
    """
    if not is_google_auth_configured():
        return None, "Google OAuth credentials not configured."

    try:
        from google_auth_oauthlib.flow import Flow

        client_config = {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [redirect_uri],
            }
        }
        flow = Flow.from_client_config(
            client_config,
            scopes=[
                "openid",
                "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/userinfo.profile",
            ],
            redirect_uri=redirect_uri,
        )
        auth_url, flow_state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
            state=state,
        )
        return auth_url, None
    except Exception as e:
        return None, f"Failed to create Google Auth URL: {e}"


def exchange_google_code_for_user_info(code: str, redirect_uri: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Exchanges an OAuth authorization code for Google user information
    using google_auth_oauthlib.flow.Flow.
    """
    if not is_google_auth_configured():
        return False, "Google OAuth credentials not configured.", None

    try:
        from google_auth_oauthlib.flow import Flow

        client_config = {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [redirect_uri],
            }
        }
        flow = Flow.from_client_config(
            client_config,
            scopes=[
                "openid",
                "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/userinfo.profile",
            ],
            redirect_uri=redirect_uri,
        )
        flow.fetch_token(code=code)
        credentials = flow.credentials

        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests

        id_info = id_token.verify_oauth2_token(
            credentials.id_token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
        return True, "Google authentication successful.", id_info
    except Exception as e:
        return False, f"OAuth code exchange failed: {e}", None


def complete_google_login(
    google_info: Dict[str, Any],
    role: Optional[str] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]], Optional[str], bool]:
    """
    Completes sign-in with verified Google user information.
    If the user has never logged in before and role is not given, returns needs_role_selection=True.
    Returns (success, message, user_dict, session_token, needs_role_selection).
    """
    email = (google_info.get("email") or "").strip().lower()
    sub = str(google_info.get("sub") or "").strip()
    name = google_info.get("name") or email.split("@")[0].title()

    if not email:
        return False, "Google account did not provide an email address.", None, None, False

    user = accounts_store.get_user_by_identity("google", sub) or accounts_store.get_user_by_identity("email", email)

    if user:
        accounts_store.link_identity_to_user(
            user["user_id"],
            identity_type="google",
            identifier=sub,
            metadata={"email": email, "picture": google_info.get("picture")},
        )
        session = accounts_store.create_session(user["user_id"])
        return True, "Google sign-in successful.", user, session["session_token"], False

    # First sign-in flow: check role
    role_clean = (role or "").strip().lower()
    if role_clean not in accounts_store.VALID_ROLES:
        return True, "Google account verified! Please select your role to complete setup.", None, None, True

    identities = [
        {"type": "google", "identifier": sub, "email": email, "picture": google_info.get("picture")},
        {"type": "email", "identifier": email},
    ]
    user = accounts_store.create_user(
        name=name,
        role=role_clean,
        identities=identities,
    )
    session = accounts_store.create_session(user["user_id"])
    return True, "Google account registered successfully.", user, session["session_token"], False


# =====================================================================
# UI Rendering (Telangana Portal Styling & Authentication)
# =====================================================================

def render_auth_page(
    page_type: str = "signin",  # 'signin', 'signup', or 'role_picker'
    error: Optional[str] = None,
    success: Optional[str] = None,
    active_tab: str = "password",
    form_data: Optional[Dict[str, Any]] = None,
    user_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Renders high-aesthetic authentication views conforming to OneBhoomi Telangana Government design standards.
    """
    form_data = form_data or {}
    phone_status = get_phone_provider_status()
    google_status = get_google_oauth_status()

    is_signup = (page_type == "signup")
    is_role_picker = (page_type == "role_picker")

    title = "OneBhoomi Authentication" if not is_signup else "OneBhoomi Account Registration"
    if is_role_picker:
        title = "Complete Your Account Setup · OneBhoomi"

    error_html = f"""
    <div class="auth-banner error-banner">
        <span class="icon">⚠️</span>
        <div class="banner-text"><b>Error:</b> {error}</div>
    </div>
    """ if error else ""

    success_html = f"""
    <div class="auth-banner success-banner">
        <span class="icon">✓</span>
        <div class="banner-text"><b>Success:</b> {success}</div>
    </div>
    """ if success else ""

    phone_badge_html = ""
    if not phone_status["enabled"]:
        phone_badge_html = """
        <div class="feature-flag-notice">
            <span class="flag-icon">ℹ️</span>
            <div>
                <b>Phone OTP in Demo Standby</b><br>
                <span>To enable live SMS, set <code>ENABLE_PHONE_OTP=true</code> and provide Twilio (SID/Token) or MSG91 credentials.</span>
            </div>
        </div>
        """

    google_badge_html = ""
    if not google_status["configured"]:
        google_badge_html = """
        <div class="feature-flag-notice">
            <span class="flag-icon">ℹ️</span>
            <div>
                <b>Google OAuth Ready</b><br>
                <span>Provide <code>GOOGLE_CLIENT_ID</code> and <code>GOOGLE_CLIENT_SECRET</code> in environment to activate one-click Google Sign-In.</span>
            </div>
        </div>
        """

    # Role selection widget
    role_cards_html = f"""
    <div class="role-selector-block">
        <label class="field-label">Portal Role Designation <span class="req">*</span></label>
        <div class="role-cards-grid">
            <label class="role-card">
                <input type="radio" name="role" value="clerk" checked>
                <div class="card-body">
                    <div class="card-icon">📋</div>
                    <div class="card-title">Clerk</div>
                    <div class="card-desc">Document ingestion, OCR verification, record amendments</div>
                </div>
            </label>
            <label class="role-card">
                <input type="radio" name="role" value="officer">
                <div class="card-body">
                    <div class="card-icon">🏛️</div>
                    <div class="card-title">Officer</div>
                    <div class="card-desc">Audit validation, official approval, RSA cryptographic sealing</div>
                </div>
            </label>
        </div>
    </div>
    """

    if is_role_picker:
        # First sign-in role confirmation page
        target_name = (user_context or {}).get("name", "User")
        target_email = (user_context or {}).get("email", "")
        method = (user_context or {}).get("method", "google")
        token_or_sub = (user_context or {}).get("identifier", "")

        body_content = f"""
        <div class="auth-header">
            <img src="/logo.png" alt="Government of Telangana" class="auth-logo" onerror="this.style.display='none'">
            <h2>Confirm Official Role</h2>
            <p class="subtitle">Welcome, <b>{target_name}</b> ({target_email}). Select your role to complete setup.</p>
        </div>

        {error_html}
        {success_html}

        <form method="POST" action="/auth/signup" class="auth-form">
            <input type="hidden" name="auth_method" value="{method}">
            <input type="hidden" name="name" value="{target_name}">
            <input type="hidden" name="email" value="{target_email}">
            <input type="hidden" name="identifier" value="{token_or_sub}">

            {role_cards_html}

            <button type="submit" class="submit-btn">Finalize Account & Enter Portal</button>
        </form>
        """
    else:
        # Sign-in or Sign-up page
        submit_btn_text = "Sign In" if not is_signup else "Create Account"
        action_url = "/auth/signin" if not is_signup else "/auth/signup"
        switch_link = f"""
        <p class="auth-switch">
            {'Don\'t have an account yet? <a href="/auth/signup">Register as Clerk or Officer →</a>' if not is_signup else 'Already have an authorized account? <a href="/auth/signin">Sign In here →</a>'}
        </p>
        """

        name_field_html = f"""
        <div class="form-group">
            <label class="field-label">Full Name <span class="req">*</span></label>
            <input type="text" name="name" class="text-input" placeholder="e.g. Ramesh Kumar" value="{form_data.get('name', '')}" {'required' if is_signup else ''}>
        </div>
        """ if is_signup else ""

        signup_role_html = role_cards_html if is_signup else ""

        body_content = f"""
        <div class="auth-header">
            <img src="/logo.png" alt="Government of Telangana" class="auth-logo" onerror="this.style.display='none'">
            <h2>{'Portal Sign-In' if not is_signup else 'Portal Registration'}</h2>
            <p class="subtitle">Government of Telangana · Land Records Digitization Registry</p>
        </div>

        {error_html}
        {success_html}

        <div class="tabs-nav">
            <button type="button" class="tab-btn active" onclick="switchAuthTab('password')">✉️ Email & Password</button>
            <button type="button" class="tab-btn" onclick="switchAuthTab('phone')">📱 Mobile & OTP</button>
            <button type="button" class="tab-btn" onclick="switchAuthTab('google')">🌐 Google Sign-In</button>
        </div>

        <!-- TAB 1: Email + Password -->
        <div id="tab-password" class="tab-pane active">
            <form method="POST" action="{action_url}" class="auth-form">
                <input type="hidden" name="auth_method" value="password">
                {name_field_html}
                {signup_role_html}

                <div class="form-group">
                    <label class="field-label">Email Address <span class="req">*</span></label>
                    <input type="email" name="email" class="text-input" placeholder="officer@revenue.telangana.gov.in" value="{form_data.get('email', '')}" required>
                </div>

                <div class="form-group">
                    <label class="field-label">Password <span class="req">*</span></label>
                    <input type="password" name="password" class="text-input" placeholder="Enter secure password (min 6 chars)" required>
                </div>

                <button type="submit" class="submit-btn">{submit_btn_text} with Password</button>
            </form>
        </div>

        <!-- TAB 2: Phone + OTP -->
        <div id="tab-phone" class="tab-pane">
            {phone_badge_html}
            <form method="POST" action="{action_url}" class="auth-form" id="phone-auth-form">
                <input type="hidden" name="auth_method" value="phone">
                {name_field_html}
                {signup_role_html}

                <div class="form-group">
                    <label class="field-label">Mobile Number (+91) <span class="req">*</span></label>
                    <div class="phone-input-row">
                        <input type="tel" name="phone" id="phone-number-input" class="text-input" placeholder="+91 98765 43210" value="{form_data.get('phone', '')}">
                        <button type="button" class="otp-request-btn" onclick="requestOtpViaAjax()" {'disabled' if not phone_status['enabled'] else ''}>
                            Request OTP
                        </button>
                    </div>
                </div>

                <div class="form-group">
                    <label class="field-label">6-Digit Verification Code</label>
                    <input type="text" name="otp" id="otp-input" class="text-input" placeholder="123456" maxlength="6" {'disabled' if not phone_status['enabled'] else ''}>
                </div>

                <button type="submit" class="submit-btn" {'disabled' if not phone_status['enabled'] else ''}>
                    {'Verify OTP & Sign In' if not is_signup else 'Verify OTP & Complete Registration'}
                </button>
            </form>
        </div>

        <!-- TAB 3: Google Sign-In -->
        <div id="tab-google" class="tab-pane">
            {google_badge_html}
            <div class="google-auth-box">
                <p class="google-desc">
                    Authenticate securely using your Google enterprise or institutional email account.
                </p>

                {signup_role_html}

                <form method="POST" action="{action_url}" id="google-auth-form">
                    <input type="hidden" name="auth_method" value="google">
                    <input type="hidden" name="google_credential" id="google_credential_field" value="">
                    
                    {'<div id="g_id_onload" data-client_id="' + GOOGLE_CLIENT_ID + '" data-callback="handleGoogleCredentialResponse"></div>' if GOOGLE_CLIENT_ID else ''}
                    
                    <a href="/auth/signin?method=google_redirect" class="google-login-btn {'disabled-link' if not google_status['configured'] else ''}">
                        <svg class="google-icon" viewBox="0 0 24 24">
                            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/>
                            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
                        </svg>
                        <span>Continue with Google</span>
                    </a>
                </form>
            </div>
        </div>

        {switch_link}
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..900;1,9..144,300..900&family=Archivo:wght@400;500;600;700&family=Courier+Prime:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
    <style>
        :root {{
            --paper: #F6F0E1;
            --paper-deep: #EFE6D0;
            --ink: #221D17;
            --ink-soft: #5A5142;
            --stamp: #A6193C;
            --stamp-deep: #7C1030;
            --rosette: #C99AA8;
            --green: #2E6B4F;
            --rule: #C9BC9F;
            --rule-soft: #DCD2B8;
            --serif: "Fraunces", Georgia, serif;
            --type: "Courier Prime", "Courier New", monospace;
            --sans: "Archivo", system-ui, sans-serif;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        html {{ scroll-behavior: smooth; }}
        body {{
            background: var(--paper);
            color: var(--ink);
            font-family: var(--sans);
            font-size: 15px;
            line-height: 1.6;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            position: relative;
            overflow-x: hidden;
        }}
        ::selection {{ background: var(--stamp); color: var(--paper); }}

        /* Security guilloche backdrop identical to landing page */
        .security-bg {{
            position: fixed; inset: 0; z-index: 0; pointer-events: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='420' height='420' viewBox='0 0 420 420'%3E%3Cg fill='none' stroke='%23C99AA8' stroke-width='1' opacity='.33'%3E%3Ccircle cx='210' cy='210' r='196'/%3E%3Ccircle cx='210' cy='210' r='188' stroke-dasharray='3 6'/%3E%3Ccircle cx='210' cy='210' r='172'/%3E%3Ccircle cx='210' cy='210' r='164' stroke-dasharray='10 4'/%3E%3Ccircle cx='210' cy='210' r='148'/%3E%3Ccircle cx='210' cy='210' r='140' stroke-dasharray='2 5'/%3E%3Ccircle cx='210' cy='210' r='124'/%3E%3Ccircle cx='210' cy='210' r='116' stroke-dasharray='8 5'/%3E%3Ccircle cx='210' cy='210' r='100'/%3E%3Ccircle cx='210' cy='210' r='92' stroke-dasharray='4 4'/%3E%3Ccircle cx='210' cy='210' r='76'/%3E%3Ccircle cx='210' cy='210' r='68' stroke-dasharray='12 3'/%3E%3Ccircle cx='210' cy='210' r='52'/%3E%3Ccircle cx='210' cy='210' r='44'/%3E%3Ccircle cx='210' cy='210' r='36' stroke-dasharray='3 4'/%3E%3Ccircle cx='210' cy='210' r='20'/%3E%3C/g%3E%3C/svg%3E");
            background-size: 420px 420px;
            opacity: .5;
        }}

        /* Perforated top & bottom edges */
        .perf {{
            height: 26px; width: 100%;
            background-image: radial-gradient(circle at 13px 13px, var(--paper) 6px, transparent 7px);
            background-size: 26px 26px;
            background-position: center top;
            position: relative; z-index: 2;
        }}
        .perf.bottom {{ background-position: center bottom; margin-top: auto; }}

        /* Marginal punch holes */
        .punches {{
            position: fixed; left: 18px; top: 0; bottom: 0; width: 22px; z-index: 2;
            display: flex; flex-direction: column; justify-content: space-evenly; pointer-events: none;
        }}
        .punches i {{
            width: 20px; height: 20px; border-radius: 50%;
            background: var(--paper-deep);
            box-shadow: inset 0 2px 4px rgba(0,0,0,.22), 0 1px 0 rgba(255,255,255,.5);
            display: block;
        }}
        @media(max-width:900px) {{ .punches {{ display: none; }} }}

        /* Site Header */
        header {{
            border-bottom: 3px double var(--rule);
            background: var(--paper);
            position: relative;
            z-index: 10;
        }}
        .wrap {{
            max-width: 1100px;
            margin: 0 auto;
            padding: 0 48px 0 68px;
        }}
        @media(max-width: 640px) {{ .wrap {{ padding: 0 22px; }} }}
        .reg-bar {{
            display: flex; align-items: center; justify-content: space-between;
            padding: 20px 0; gap: 24px; flex-wrap: nowrap;
        }}
        .brand {{
            display: flex; align-items: center; gap: 14px; text-decoration: none; color: var(--ink);
            white-space: nowrap; flex-shrink: 0;
        }}
        .brand b {{
            font-family: var(--serif); font-weight: 900; font-size: 26px; letter-spacing: .04em;
            white-space: nowrap;
        }}
        .brand span {{
            font-family: var(--type); font-size: 11px; letter-spacing: .14em; color: var(--stamp);
            text-transform: uppercase; white-space: nowrap; display: inline-block;
        }}
        nav {{
            display: flex; gap: 28px; align-items: center; white-space: nowrap;
        }}
        nav a {{
            font-family: var(--type); font-size: 12px; letter-spacing: .14em; text-transform: uppercase;
            color: var(--ink-soft); text-decoration: none; white-space: nowrap;
            transition: color .15s ease;
        }}
        nav a:hover {{ color: var(--stamp); }}
        .reg-right {{
            display: flex; align-items: center; gap: 14px; white-space: nowrap; flex-shrink: 0;
        }}
        .btn-head {{
            padding: 10px 18px; font-size: 11.5px; letter-spacing: .14em; word-spacing: .18em;
            white-space: nowrap; flex: none;
        }}
        @media(max-width: 820px) {{ nav {{ display: none; }} }}

        /* Button styles */
        .btn {{
            font-family: var(--type); font-size: 13px; letter-spacing: .16em; text-transform: uppercase;
            text-decoration: none; padding: 12px 24px; border-radius: 2px;
            transition: transform .15s ease, box-shadow .15s ease;
            display: inline-block; cursor: pointer;
        }}
        .btn-ghost {{
            color: var(--ink); border: 1.5px solid var(--ink);
        }}
        .btn-ghost:hover {{
            background: var(--ink); color: var(--paper);
        }}
        .btn-primary {{
            background: var(--stamp); color: var(--paper);
            box-shadow: 3px 3px 0 var(--stamp-deep);
            border: none;
        }}
        .btn-primary:hover {{
            transform: translate(-1px,-1px);
            box-shadow: 4px 4px 0 var(--stamp-deep);
        }}

        /* Auth Container & Layout */
        .auth-main-wrap {{
            position: relative;
            z-index: 1;
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 44px 16px 56px;
        }}
        .auth-container {{
            width: 100%;
            max-width: 530px;
            background: #FFFDF6;
            border: 2px solid var(--rule);
            border-radius: 4px;
            box-shadow: 5px 5px 0 var(--rule), 0 16px 36px rgba(34, 29, 23, 0.08);
            padding: 36px 34px;
            position: relative;
        }}
        @media(max-width: 540px) {{
            .auth-container {{ padding: 26px 20px; }}
        }}

        .auth-header {{
            text-align: center;
            margin-bottom: 24px;
            border-bottom: 1.5px solid var(--rule-soft);
            padding-bottom: 18px;
        }}
        .auth-logo {{
            max-height: 48px;
            margin-bottom: 12px;
        }}
        .auth-header h2 {{
            font-family: var(--serif);
            font-size: 1.85rem;
            font-weight: 700;
            color: var(--ink);
            letter-spacing: -0.015em;
            line-height: 1.2;
        }}
        .subtitle {{
            font-family: var(--type);
            font-size: 0.78rem;
            color: var(--stamp);
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-top: 6px;
        }}
        .subtitle b {{ color: var(--ink); }}

        /* Alerts / Banners */
        .auth-banner {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 11px 15px;
            border-radius: 3px;
            margin-bottom: 20px;
            font-size: 0.88rem;
        }}
        .error-banner {{
            background: #fdf2f2;
            border: 1.5px solid var(--stamp);
            color: var(--stamp-deep);
            font-weight: 500;
        }}
        .success-banner {{
            background: #f0fdf4;
            border: 1.5px solid var(--green);
            color: var(--green);
            font-weight: 500;
        }}

        /* Tabs */
        .tabs-nav {{
            display: flex;
            background: var(--paper-deep);
            border: 1.5px solid var(--rule);
            border-radius: 3px;
            padding: 4px;
            margin-bottom: 24px;
            gap: 4px;
        }}
        .tab-btn {{
            flex: 1;
            background: transparent;
            border: none;
            color: var(--ink-soft);
            font-family: var(--type);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: .05em;
            padding: 9px 8px;
            border-radius: 2px;
            cursor: pointer;
            transition: all 0.15s ease;
            white-space: nowrap;
        }}
        .tab-btn:hover {{
            color: var(--stamp);
        }}
        .tab-btn.active {{
            background: #FFFDF6;
            color: var(--stamp);
            box-shadow: 1px 1px 0 var(--rule);
            border: 1px solid var(--rule);
        }}
        .tab-pane {{
            display: none;
        }}
        .tab-pane.active {{
            display: block;
        }}

        /* Form Inputs */
        .form-group {{
            margin-bottom: 18px;
        }}
        .field-label {{
            display: block;
            font-family: var(--type);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin-bottom: 6px;
            color: var(--ink-soft);
        }}
        .req {{ color: var(--stamp); }}
        .text-input {{
            width: 100%;
            background: #FFFFFF;
            border: 1.5px solid var(--rule);
            border-radius: 3px;
            padding: 11px 14px;
            color: var(--ink);
            font-family: var(--sans);
            font-size: 0.95rem;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}
        .text-input:focus {{
            outline: none;
            border-color: var(--stamp);
            box-shadow: 0 0 0 3px rgba(166, 25, 60, 0.12);
        }}

        /* Submit Button */
        .submit-btn {{
            width: 100%;
            background: var(--stamp);
            border: none;
            border-radius: 2px;
            color: var(--paper);
            font-family: var(--type);
            font-size: 0.88rem;
            font-weight: 700;
            letter-spacing: .14em;
            text-transform: uppercase;
            padding: 13px 16px;
            cursor: pointer;
            margin-top: 10px;
            box-shadow: 3px 3px 0 var(--stamp-deep);
            transition: transform .15s ease, box-shadow .15s ease;
        }}
        .submit-btn:hover:not(:disabled) {{
            transform: translate(-1px, -1px);
            box-shadow: 4px 4px 0 var(--stamp-deep);
        }}
        .submit-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }}

        /* Role Picker Cards */
        .role-selector-block {{
            margin-bottom: 20px;
        }}
        .role-cards-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 6px;
        }}
        .role-card {{
            position: relative;
            cursor: pointer;
        }}
        .role-card input {{
            position: absolute;
            opacity: 0;
        }}
        .card-body {{
            background: var(--paper-deep);
            border: 1.5px solid var(--rule);
            border-radius: 3px;
            padding: 14px 12px;
            text-align: center;
            transition: all 0.2s ease;
        }}
        .role-card input:checked + .card-body {{
            background: #FFFDF6;
            border-color: var(--stamp);
            box-shadow: 2px 2px 0 var(--stamp-deep);
        }}
        .card-icon {{ font-size: 1.4rem; margin-bottom: 4px; }}
        .card-title {{
            font-family: var(--serif);
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--ink);
        }}
        .card-desc {{
            font-family: var(--sans);
            font-size: 0.75rem;
            color: var(--ink-soft);
            margin-top: 4px;
            line-height: 1.35;
        }}

        /* Phone & OTP */
        .phone-input-row {{
            display: flex;
            gap: 8px;
        }}
        .otp-request-btn {{
            background: var(--paper-deep);
            border: 1.5px solid var(--rule);
            color: var(--stamp);
            border-radius: 3px;
            font-family: var(--type);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
            padding: 0 14px;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.15s ease;
        }}
        .otp-request-btn:hover:not(:disabled) {{
            border-color: var(--stamp);
            box-shadow: 1px 1px 0 var(--stamp);
            background: #FFFDF6;
        }}
        .otp-request-btn:disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}

        .feature-flag-notice {{
            display: flex;
            align-items: flex-start;
            gap: 10px;
            background: rgba(166, 25, 60, 0.06);
            border: 1.5px solid var(--rule);
            border-radius: 3px;
            padding: 10px 14px;
            margin-bottom: 16px;
            font-size: 0.82rem;
            color: var(--ink-soft);
            line-height: 1.4;
        }}
        .flag-icon {{ font-size: 1.1rem; flex-shrink: 0; }}
        .feature-flag-notice code {{
            background: var(--paper-deep);
            border: 1px solid var(--rule);
            padding: 1px 5px;
            border-radius: 2px;
            font-size: 0.8rem;
            color: var(--ink);
            font-family: var(--type);
        }}

        /* Google Sign-in */
        .google-auth-box {{
            text-align: center;
            padding: 10px 0;
        }}
        .google-desc {{
            font-size: 0.88rem;
            color: var(--ink-soft);
            margin-bottom: 16px;
        }}
        .google-login-btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            width: 100%;
            background: #FFFFFF;
            color: var(--ink);
            font-family: var(--sans);
            font-size: 0.92rem;
            font-weight: 600;
            padding: 11px 16px;
            border-radius: 3px;
            border: 1.5px solid var(--rule);
            text-decoration: none;
            box-shadow: 2px 2px 0 var(--rule);
            transition: all 0.15s ease;
        }}
        .google-login-btn:hover:not(.disabled-link) {{
            border-color: var(--ink-soft);
            box-shadow: 3px 3px 0 var(--rule);
            transform: translate(-1px, -1px);
        }}
        .google-login-btn.disabled-link {{
            opacity: 0.5;
            pointer-events: none;
            filter: grayscale(1);
        }}
        .google-icon {{
            width: 20px;
            height: 20px;
        }}

        /* Switch link */
        .auth-switch {{
            text-align: center;
            font-family: var(--type);
            font-size: 0.82rem;
            color: var(--ink-soft);
            margin-top: 24px;
            letter-spacing: .02em;
        }}
        .auth-switch a {{
            color: var(--stamp);
            text-decoration: none;
            font-weight: 700;
        }}
        .auth-switch a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="security-bg" aria-hidden="true"></div>
    <div class="punches" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
    <div class="perf" aria-hidden="true"></div>

    <header>
        <div class="wrap reg-bar">
            <a class="brand" href="/">
                <img src="/logo.png?v=20260904d" alt="Government of Telangana Seal" style="height:44px; width:auto; display:block;" onerror="this.style.display='none'">
                <div>
                    <b>OneBhoomi</b>
                    <span>Offline Land Records Registry</span>
                </div>
            </a>
            <nav>
                <a href="/#extraction">1 · Read</a>
                <a href="/#workflow">2 · Check</a>
                <a href="/#sealing">3 · Seal</a>
                <a href="/verify">Verify Seal</a>
            </nav>
            <div class="reg-right">
                <a class="btn btn-ghost btn-head" href="/" style="font-weight:700;">&larr; Back to Portal</a>
            </div>
        </div>
    </header>

    <main class="auth-main-wrap">
        <div class="auth-container">
            {body_content}
        </div>
    </main>

    <div class="perf bottom" aria-hidden="true"></div>

    <script>
        function switchAuthTab(tabName) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
            
            const btn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabName));
            if (btn) btn.classList.add('active');
            
            const pane = document.getElementById('tab-' + tabName);
            if (pane) pane.classList.add('active');
        }}

        function requestOtpViaAjax() {{
            const phone = document.getElementById('phone-number-input').value;
            if (!phone) {{
                alert('Please enter your phone number first.');
                return;
            }}
            fetch('/auth/signin?action=request_otp', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ phone: phone }})
            }})
            .then(res => res.json())
            .then(data => {{
                alert(data.message || (data.status === 'ok' ? 'OTP sent!' : 'Could not send OTP'));
            }})
            .catch(err => alert('Failed to request OTP: ' + err));
        }}
    </script>
</body>
</html>
"""


def _get_role_destination_for_session(session_token: str) -> str:
    """
    Reads the user's role from accounts_store.py (the same lookup get_current_user() already performs).
    Redirects to "/clerk" if role == "clerk" or "/officer" if role == "officer".
    If a user somehow has no role set yet, defensively redirects to "/auth/choose-role".
    """
    if not session_token:
        return "/auth/signin"
    session = accounts_store.get_session(session_token)
    if not session:
        return "/auth/signin"
    user_id = session.get("user_id")
    if not user_id:
        return "/auth/signin"
    user = accounts_store.get_user(user_id)
    if not user:
        return "/auth/signin"
    role = (user.get("role") or "").strip().lower()
    if role == "clerk":
        return "/clerk"
    elif role == "officer":
        return "/officer"
    else:
        return "/auth/choose-role"


# =====================================================================
# Request Dispatchers: GET /auth/signin and GET /auth/signup
# =====================================================================

def handle_signin_get(handler: Any, query_params: Dict[str, List[str]]) -> None:
    """Handles GET /auth/signin and defensive GET /auth/choose-role."""
    parsed = urlparse(handler.path)
    if parsed.path == "/auth/choose-role":
        current_user = accounts_store.get_current_user(handler)
        user_name = (current_user or {}).get("name", "Authorized User")
        user_id = (current_user or {}).get("user_id", "")
        email = ""
        for ident in (current_user or {}).get("identities", []):
            if isinstance(ident, dict) and ident.get("type") == "email":
                email = ident.get("identifier") or ident.get("value") or ""
                break
        html = render_auth_page(
            "role_picker",
            user_context={
                "name": user_name,
                "email": email or user_id,
                "method": "choose_role",
                "identifier": user_id,
            }
        )
        _send_html_response(handler, html)
        return

    error = query_params.get("error", [None])[0]
    success = query_params.get("success", [None])[0]
    method = query_params.get("method", [None])[0]

    # Handle Google OAuth redirect flow
    if method == "google_redirect":
        host = handler.headers.get("Host", "localhost:8001")
        redirect_uri = f"http://{host}/auth/signin?method=google_callback"
        auth_url, err = get_google_auth_url(redirect_uri)
        if auth_url:
            handler.send_response(HTTPStatus.FOUND)
            handler.send_header("Location", auth_url)
            handler.end_headers()
            return
        else:
            status = get_google_oauth_status()
            html = render_auth_page("signin", error=err or status["instruction"])
            _send_html_response(handler, html)
            return

    # Handle Google OAuth callback flow
    if method == "google_callback":
        code = query_params.get("code", [None])[0]
        if code:
            host = handler.headers.get("Host", "localhost:8001")
            redirect_uri = f"http://{host}/auth/signin?method=google_callback"
            ok, msg, id_info = exchange_google_code_for_user_info(code, redirect_uri)
            if ok and id_info:
                success_login, login_msg, user, session_token, needs_role = complete_google_login(id_info)
                if needs_role:
                    # Render role selection page
                    html = render_auth_page(
                        "role_picker",
                        user_context={
                            "name": id_info.get("name", ""),
                            "email": id_info.get("email", ""),
                            "method": "google",
                            "identifier": id_info.get("sub", ""),
                        }
                    )
                    _send_html_response(handler, html)
                    return
                elif success_login and session_token:
                    dest = _get_role_destination_for_session(session_token)
                    _set_session_and_redirect(handler, session_token, dest)
                    return
                else:
                    html = render_auth_page("signin", error=login_msg)
                    _send_html_response(handler, html)
                    return
            else:
                html = render_auth_page("signin", error=msg)
                _send_html_response(handler, html)
                return

    html = render_auth_page("signin", error=error, success=success)
    _send_html_response(handler, html)


def handle_signup_get(handler: Any, query_params: Dict[str, List[str]]) -> None:
    """Handles GET /auth/signup."""
    error = query_params.get("error", [None])[0]
    success = query_params.get("success", [None])[0]
    html = render_auth_page("signup", error=error, success=success)
    _send_html_response(handler, html)


# =====================================================================
# Request Dispatchers: POST /auth/signin and POST /auth/signup
# =====================================================================

def handle_signin_post(handler: Any) -> None:
    """Handles POST /auth/signin."""
    content_type = handler.headers.get("Content-Type", "")
    length = int(handler.headers.get("Content-Length", "0"))
    body = handler.rfile.read(length)

    data = _parse_request_body(body, content_type)
    auth_method = data.get("auth_method", "password")

    # Ajax action: request_otp
    parsed = urlparse(handler.path)
    if "action=request_otp" in parsed.query or data.get("action") == "request_otp":
        phone = data.get("phone", "")
        sent, msg = request_phone_otp(phone)
        _send_json_response(handler, {"status": "ok" if sent else "error", "message": msg})
        return

    # Method 1: Email + Password
    if auth_method == "password":
        email = data.get("email", "")
        password = data.get("password", "")
        ok, msg, user, token = signin_email_password(email, password)
        if ok and token:
            dest = _get_role_destination_for_session(token)
            _set_session_and_redirect(handler, token, dest)
            return
        else:
            html = render_auth_page("signin", error=msg, form_data=data)
            _send_html_response(handler, html, status=HTTPStatus.UNAUTHORIZED)
            return

    # Method 2: Phone + OTP
    if auth_method == "phone":
        phone = data.get("phone", "")
        otp = data.get("otp", "")
        ok, msg, user, token, needs_role = verify_phone_otp_and_login(phone, otp)
        if needs_role:
            html = render_auth_page(
                "role_picker",
                user_context={"name": f"User {phone[-4:]}", "email": "", "method": "phone", "identifier": phone}
            )
            _send_html_response(handler, html)
            return
        elif ok and token:
            dest = _get_role_destination_for_session(token)
            _set_session_and_redirect(handler, token, dest)
            return
        else:
            html = render_auth_page("signin", error=msg, active_tab="phone", form_data=data)
            _send_html_response(handler, html, status=HTTPStatus.BAD_REQUEST)
            return

    # Method 3: Google ID Token
    if auth_method == "google":
        credential = data.get("google_credential", "")
        ok, msg, id_info = verify_google_id_token(credential)
        if ok and id_info:
            success_login, login_msg, user, token, needs_role = complete_google_login(id_info)
            if needs_role:
                html = render_auth_page(
                    "role_picker",
                    user_context={
                        "name": id_info.get("name", ""),
                        "email": id_info.get("email", ""),
                        "method": "google",
                        "identifier": id_info.get("sub", "")
                    }
                )
                _send_html_response(handler, html)
                return
            elif success_login and token:
                dest = _get_role_destination_for_session(token)
                _set_session_and_redirect(handler, token, dest)
                return
            else:
                html = render_auth_page("signin", error=login_msg, active_tab="google")
                _send_html_response(handler, html, status=HTTPStatus.UNAUTHORIZED)
                return
        else:
            html = render_auth_page("signin", error=msg, active_tab="google")
            _send_html_response(handler, html, status=HTTPStatus.BAD_REQUEST)
            return

    handler.send_error(HTTPStatus.BAD_REQUEST, "Unknown authentication method")


def handle_signup_post(handler: Any) -> None:
    """Handles POST /auth/signup."""
    content_type = handler.headers.get("Content-Type", "")
    length = int(handler.headers.get("Content-Length", "0"))
    body = handler.rfile.read(length)

    data = _parse_request_body(body, content_type)
    auth_method = data.get("auth_method", "password")
    role = data.get("role", "clerk")
    name = data.get("name", "")

    # Defensive Role Picker completion for user without role
    if auth_method == "choose_role":
        current_user = accounts_store.get_current_user(handler)
        if current_user:
            current_user["role"] = role
            accounts_store.save_user(current_user)
            token = accounts_store.extract_session_token_from_request(handler)
            dest = "/clerk" if role == "clerk" else "/officer"
            _set_session_and_redirect(handler, token, dest)
            return

    # Role Picker completion for Google or Phone
    if auth_method == "google" and data.get("identifier"):
        sub = data.get("identifier", "")
        email = data.get("email", "")
        google_info = {"sub": sub, "email": email, "name": name}
        ok, msg, user, token, _ = complete_google_login(google_info, role=role)
        if ok and token:
            dest = _get_role_destination_for_session(token)
            _set_session_and_redirect(handler, token, dest)
            return
        else:
            html = render_auth_page("signup", error=msg)
            _send_html_response(handler, html, status=HTTPStatus.BAD_REQUEST)
            return

    if auth_method == "phone" and data.get("identifier"):
        phone = data.get("identifier", "")
        user = accounts_store.create_user(
            name=name or f"User {phone[-4:]}",
            role=role,
            identities=[{"type": "phone", "identifier": phone}],
        )
        session = accounts_store.create_session(user["user_id"])
        dest = _get_role_destination_for_session(session["session_token"])
        _set_session_and_redirect(handler, session["session_token"], dest)
        return

    # Method 1: Email + Password Registration
    if auth_method == "password":
        ok, msg, user, token = register_email_password(
            email=data.get("email", ""),
            password=data.get("password", ""),
            name=name,
            role=role,
        )
        if ok and token:
            dest = _get_role_destination_for_session(token)
            _set_session_and_redirect(handler, token, dest)
            return
        else:
            html = render_auth_page("signup", error=msg, form_data=data)
            _send_html_response(handler, html, status=HTTPStatus.BAD_REQUEST)
            return

    # Method 2: Phone + OTP Registration
    if auth_method == "phone":
        phone = data.get("phone", "")
        otp = data.get("otp", "")
        ok, msg, user, token, needs_role = verify_phone_otp_and_login(
            phone,
            otp,
            name=name,
            role=role,
        )
        if ok and token:
            dest = _get_role_destination_for_session(token)
            _set_session_and_redirect(handler, token, dest)
            return
        else:
            html = render_auth_page("signup", error=msg, active_tab="phone", form_data=data)
            _send_html_response(handler, html, status=HTTPStatus.BAD_REQUEST)
            return

    handler.send_error(HTTPStatus.BAD_REQUEST, "Invalid registration request")


# =====================================================================
# HTTP Helpers
# =====================================================================

def _parse_request_body(body: bytes, content_type: str) -> Dict[str, Any]:
    """Parses application/x-www-form-urlencoded or application/json body."""
    if not body:
        return {}
    try:
        if "application/json" in content_type:
            return json.loads(body.decode("utf-8"))
        else:
            parsed = parse_qs(body.decode("utf-8"))
            return {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}
    except Exception:
        return {}


def _send_html_response(handler: Any, html_content: str, status: HTTPStatus = HTTPStatus.OK) -> None:
    data = html_content.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _send_json_response(handler: Any, data_dict: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
    data = json.dumps(data_dict, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _set_session_and_redirect(handler: Any, session_token: str, redirect_url: Optional[str] = None) -> None:
    if not redirect_url or redirect_url == "/":
        redirect_url = _get_role_destination_for_session(session_token)
    handler.send_response(HTTPStatus.SEE_OTHER)
    handler.send_header("Location", redirect_url)
    handler.send_header(
        "Set-Cookie",
        f"{accounts_store.SESSION_COOKIE_NAME}={session_token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=604800"
    )
    handler.end_headers()
