import requests
import accounts_store

BASE_URL = "http://localhost:8001"

# 1. Unauthenticated GET /
resp_unauth = requests.get(f"{BASE_URL}/", timeout=5)
assert resp_unauth.status_code == 200
assert 'id="headerSignInBtn"' in resp_unauth.text, "Unauthenticated user must see headerSignInBtn"
assert "Sign In" in resp_unauth.text
assert "Signed in as" not in resp_unauth.text, "Unauthenticated user must not see Signed in as"
print("PASS 1: Unauthenticated user sees original 'Sign In' button.")

# 2. Authenticated GET / with session cookie
user = accounts_store.create_user(name="Srikanth Rao", role="clerk", identities=[{"type": "email", "identifier": "test_clerk_header@gov.in"}])
session = accounts_store.create_session(user["user_id"])
token = session["session_token"]

s_clerk = requests.Session()
s_clerk.cookies.set(accounts_store.SESSION_COOKIE_NAME, token)
resp_clerk = s_clerk.get(f"{BASE_URL}/", timeout=5)
assert resp_clerk.status_code == 200
assert 'id="headerSignInBtn"' not in resp_clerk.text, "Authenticated clerk must not see headerSignInBtn"
assert "Signed in as" in resp_clerk.text
assert "Srikanth Rao" in resp_clerk.text
assert "/auth/signout" in resp_clerk.text, "Authenticated clerk must see /auth/signout link"
print("PASS 2: Authenticated clerk sees 'Signed in as Srikanth Rao' and Sign Out link.")

# 3. Sign out and re-check /
s_clerk.get(f"{BASE_URL}/auth/signout", timeout=5)
resp_after_signout = s_clerk.get(f"{BASE_URL}/", timeout=5)
assert 'id="headerSignInBtn"' in resp_after_signout.text, "After signout, user must see headerSignInBtn again"
print("PASS 3: After signout, 'Sign In' button returns.")

print("ALL TESTS PASSED PERFECTLY!")
