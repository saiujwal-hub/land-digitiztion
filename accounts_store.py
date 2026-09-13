import json
import os
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any, Dict, List, Optional

USERS_DB_PATH = Path(os.environ.get("USERS_DB_PATH", "users_db.json"))
SESSIONS_DB_PATH = Path(os.environ.get("SESSIONS_DB_PATH", "sessions_db.json"))

SESSION_COOKIE_NAME = "session_token"
DEFAULT_SESSION_DURATION_DAYS = 7
VALID_ROLES = {"user", "officer"}

_users_lock = threading.RLock()
_sessions_lock = threading.RLock()


# =====================================================================
# Users Store Persistence Layer
# =====================================================================

def load_users_db() -> Dict[str, Dict[str, Any]]:
    """Reads the users JSON database file."""
    with _users_lock:
        if not USERS_DB_PATH.exists():
            return {}
        try:
            with open(USERS_DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def save_users_db(users_db: Dict[str, Dict[str, Any]]) -> None:
    """Writes the users database structure to the JSON file."""
    with _users_lock:
        tmp_path = USERS_DB_PATH.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(users_db, f, indent=2, ensure_ascii=False)
            tmp_path.replace(USERS_DB_PATH)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed to write users database: {e}")


def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user by user_id."""
    if not user_id:
        return None
    users = load_users_db()
    return users.get(user_id)


def get_user_by_identity(identity_type: str, identifier: str) -> Optional[Dict[str, Any]]:
    """
    Finds a user linked with a specific auth identity (e.g. email, phone, google).
    Matches against 'identifier' or 'value' fields in identities list.
    """
    if not identity_type or not identifier:
        return None

    target_type = identity_type.strip().lower()
    target_id = identifier.strip().lower()

    users = load_users_db()
    for user in users.values():
        if not isinstance(user, dict):
            continue
        identities = user.get("identities", [])
        if not isinstance(identities, list):
            continue
        for identity in identities:
            if isinstance(identity, dict):
                itype = str(identity.get("type", "")).strip().lower()
                ival = str(identity.get("identifier") or identity.get("value") or "").strip().lower()
                if itype == target_type and ival == target_id:
                    return user
            elif isinstance(identity, str) and target_type == "email":
                if identity.strip().lower() == target_id:
                    return user
    return None


def save_user(user: Dict[str, Any]) -> None:
    """Saves or updates a user in the store."""
    user_id = user.get("user_id")
    if not user_id:
        raise ValueError("User must have a user_id")
    role = user.get("role")
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid user role: {role}. Must be one of {VALID_ROLES}")

    with _users_lock:
        users = load_users_db()
        users[user_id] = user
        save_users_db(users)


def create_user(
    name: str,
    role: str,
    identities: Optional[List[Dict[str, Any]]] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates and stores a new user record.
    Roles must be either 'clerk' or 'officer'.
    """
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid user role '{role}'. Must be one of: {', '.join(sorted(VALID_ROLES))}")

    uid = user_id or f"usr_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    normalized_identities: List[Dict[str, Any]] = []
    if identities:
        for ident in identities:
            if isinstance(ident, dict):
                normalized_identities.append(ident)
            elif isinstance(ident, str):
                normalized_identities.append({"type": "email", "identifier": ident})

    user = {
        "user_id": uid,
        "name": name.strip(),
        "role": role,
        "identities": normalized_identities,
        "created_at": now_iso,
    }
    save_user(user)
    return user


def link_identity_to_user(user_id: str, identity_type: str, identifier: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Links an additional auth identity (email, phone, google) to an existing user."""
    with _users_lock:
        user = get_user(user_id)
        if not user:
            raise ValueError(f"User with id '{user_id}' does not exist")

        identities = user.setdefault("identities", [])
        # Check if already linked
        target_type = identity_type.strip().lower()
        target_id = identifier.strip().lower()

        for ident in identities:
            if isinstance(ident, dict):
                if (
                    str(ident.get("type", "")).strip().lower() == target_type
                    and str(ident.get("identifier") or ident.get("value") or "").strip().lower() == target_id
                ):
                    return user

        new_identity = {
            "type": target_type,
            "identifier": target_id,
        }
        if metadata:
            new_identity.update(metadata)
        identities.append(new_identity)
        save_user(user)
        return user





# =====================================================================
# Sessions Store Persistence Layer
# =====================================================================

def load_sessions_db() -> Dict[str, Dict[str, Any]]:
    """Reads the sessions JSON database file."""
    with _sessions_lock:
        if not SESSIONS_DB_PATH.exists():
            return {}
        try:
            with open(SESSIONS_DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def save_sessions_db(sessions_db: Dict[str, Dict[str, Any]]) -> None:
    """Writes the sessions database structure to the JSON file."""
    with _sessions_lock:
        tmp_path = SESSIONS_DB_PATH.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(sessions_db, f, indent=2, ensure_ascii=False)
            tmp_path.replace(SESSIONS_DB_PATH)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed to write sessions database: {e}")


def _parse_iso_utc(ts_str: str) -> Optional[datetime]:
    """Safely parses ISO timestamp string to UTC datetime."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def create_session(
    user_id: str,
    duration_days: int = DEFAULT_SESSION_DURATION_DAYS,
    custom_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates a new authenticated session for user_id with an expiration date.
    Returns the created session dict.
    """
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=duration_days)
    token = custom_token or secrets.token_urlsafe(32)

    session = {
        "session_token": token,
        "user_id": user_id,
        "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at": expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    with _sessions_lock:
        sessions = load_sessions_db()
        sessions[token] = session
        save_sessions_db(sessions)

    return session


def get_session(session_token: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a session by token if it exists and has not expired.
    Returns None if invalid or expired.
    """
    if not session_token:
        return None

    sessions = load_sessions_db()
    session = sessions.get(session_token)
    if not session:
        return None

    expires_at = _parse_iso_utc(session.get("expires_at", ""))
    if not expires_at:
        return None

    if datetime.now(timezone.utc) >= expires_at:
        # Session expired - delete it
        delete_session(session_token)
        return None

    return session


def delete_session(session_token: str) -> bool:
    """Deletes an active session."""
    if not session_token:
        return False
    with _sessions_lock:
        sessions = load_sessions_db()
        if session_token in sessions:
            del sessions[session_token]
            save_sessions_db(sessions)
            return True
    return False


# =====================================================================
# Request Helper: get_current_user(request)
# =====================================================================

def extract_session_token_from_request(request: Any, cookie_name: str = SESSION_COOKIE_NAME) -> Optional[str]:
    """
    Extracts the session token from various HTTP request formats:
    - BaseHTTPRequestHandler (request.headers.get('Cookie'))
    - Dict with 'headers', 'Cookie', 'cookie', or 'session_token'
    - Request objects with a .cookies dictionary/mapping or .headers mapping
    - Direct cookie string or token string
    """
    if request is None:
        return None

    cookie_str: Optional[str] = None

    # Case 1: Raw string passed
    if isinstance(request, str):
        if f"{cookie_name}=" in request:
            cookie_str = request
        elif "=" not in request and len(request) >= 16:
            # Directly passed token string
            return request.strip()

    # Case 2: Object with .cookies attribute
    if hasattr(request, "cookies"):
        cookies_attr = request.cookies
        if isinstance(cookies_attr, dict):
            val = cookies_attr.get(cookie_name)
            if val:
                return str(val).strip()
        elif callable(cookies_attr):
            try:
                res = cookies_attr()
                if isinstance(res, dict) and cookie_name in res:
                    return str(res[cookie_name]).strip()
            except Exception:
                pass

    # Case 3: Object with .headers (e.g. BaseHTTPRequestHandler, Requests, Web frameworks)
    if hasattr(request, "headers"):
        headers = request.headers
        try:
            cookie_str = headers.get("Cookie") or headers.get("cookie")
        except Exception:
            pass

    # Case 4: Dict-like request structure
    if not cookie_str and isinstance(request, dict):
        if cookie_name in request:
            return str(request[cookie_name]).strip()
        headers = request.get("headers")
        if isinstance(headers, dict):
            cookie_str = headers.get("Cookie") or headers.get("cookie")
        elif "Cookie" in request:
            cookie_str = request["Cookie"]
        elif "cookie" in request:
            cookie_str = request["cookie"]

    # Parse standard Cookie header if found
    if cookie_str:
        try:
            cookie = SimpleCookie()
            cookie.load(cookie_str)
            if cookie_name in cookie:
                return cookie[cookie_name].value.strip()
        except Exception:
            # Fallback regex extraction if malformed cookie header
            import re
            m = re.search(rf"{re.escape(cookie_name)}=([a-zA-Z0-9_\-]+)", cookie_str)
            if m:
                return m.group(1).strip()

    return None


def get_current_user(request: Any) -> Optional[Dict[str, Any]]:
    """
    Reads the session cookie from request and returns the authenticated user dict,
    or None if unauthenticated, invalid, or expired.
    """
    token = extract_session_token_from_request(request)
    if not token:
        return None

    session = get_session(token)
    if not session:
        return None

    user_id = session.get("user_id")
    if not user_id:
        return None

    return get_user(user_id)
