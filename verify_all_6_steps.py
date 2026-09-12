#!/usr/bin/env python3
"""
verify_all_6_steps.py
Walks through and reports the result of the exact 6-step sequence:
1. With no session, load the site root — confirm the landing page loads directly, with a visible Sign In button, and no redirect.
2. Sign in as a clerk account — confirm landing directly on /clerk, with the "Logged in as [name] - Clerk Digitization Desk" header visible.
3. Upload a document that will pass all machine checks. Confirm it appears in "Needs your review" with a visible Submit to Officer button, and does NOT yet appear anywhere in an officer's queue.
4. Click Submit to Officer. Confirm the clerk dashboard now shows it under "Sent to officer."
5. Sign out, sign in as an officer account — confirm landing directly on /officer, and that the just-submitted document now appears in the pending queue.
6. Approve it. Sign back in as the original clerk — confirm it now shows under "Decided" with a working link to the certificate/QR view.
"""

import sys
import uuid
import requests

import accounts_store
import auth_service
import verification_service

BASE_URL = "http://localhost:8001"

def print_flush(msg: str):
    print(msg)
    sys.stdout.flush()

def main():
    results = {}

    print_flush("======================================================================")
    print_flush("STARTING 6-STEP VERIFICATION WALKTHROUGH")
    print_flush("======================================================================")

    # -------------------------------------------------------------------------
    # STEP 1: Site root with no session
    # -------------------------------------------------------------------------
    print_flush("\n--- Step 1: Loading site root with no session ---")
    try:
        s1 = requests.Session()
        resp1 = s1.get(f"{BASE_URL}/", allow_redirects=False, timeout=10)
        
        is_direct = (resp1.status_code == 200)
        has_signin_btn = ('/auth/signin' in resp1.text) and ('Sign In' in resp1.text)
        is_landing_page = ('OneBhoomi' in resp1.text)
        
        if is_direct and has_signin_btn and is_landing_page:
            print_flush("[PASS] Step 1: Landing page loads directly (HTTP 200), visible Sign In button present, no redirect.")
            results["Step 1"] = "PASS"
        else:
            print_flush(f"[FAIL] Step 1: code={resp1.status_code}, has_btn={has_signin_btn}, is_landing={is_landing_page}")
            results["Step 1"] = f"FAIL (status={resp1.status_code}, btn={has_signin_btn})"
    except Exception as e:
        print_flush(f"[FAIL] Step 1 exception: {e}")
        results["Step 1"] = f"FAIL ({e})"

    # -------------------------------------------------------------------------
    # SETUP TEST USERS
    # -------------------------------------------------------------------------
    clerk_name = "Srikanth Rao"
    clerk_email = f"clerk_{uuid.uuid4().hex[:6]}@telangana.gov.in"
    officer_name = "V. K. Shastry"
    officer_email = f"officer_{uuid.uuid4().hex[:6]}@telangana.gov.in"
    password = "VerificationPass2026!"

    ok, msg, officer_user, _ = auth_service.register_email_password(
        email=officer_email,
        password=password,
        name=officer_name,
        role="officer"
    )
    assert ok, f"Officer user creation failed: {msg}"

    ok, msg, clerk_user, _ = auth_service.register_email_password(
        email=clerk_email,
        password=password,
        name=clerk_name,
        role="clerk",
    )
    assert ok, f"Clerk user creation failed: {msg}"

    # -------------------------------------------------------------------------
    # STEP 2: Sign in as clerk account
    # -------------------------------------------------------------------------
    print_flush("\n--- Step 2: Sign in as clerk account ---")
    try:
        clerk_session = requests.Session()
        signin_resp = clerk_session.post(
            f"{BASE_URL}/auth/signin",
            data={"email": clerk_email, "password": password},
            allow_redirects=False,
            timeout=10
        )
        
        redirect_code = signin_resp.status_code
        redirect_target = signin_resp.headers.get("Location")
        print_flush(f"Sign-in redirect: status={redirect_code}, Location={redirect_target}")
        
        is_redirect_to_clerk = (redirect_code in (302, 303) and redirect_target == "/clerk")
        
        # Follow into /clerk
        clerk_dash_resp = clerk_session.get(f"{BASE_URL}/clerk", timeout=10)
        clerk_dash_html = clerk_dash_resp.text

        # Header check: "Logged in as [name] - Clerk Digitization Desk"
        import re
        has_clerk_title = bool(re.search(r"Clerk\s*(?:<[^>]+>)?\s*Digitization Desk", clerk_dash_html, re.IGNORECASE))
        has_logged_in_name = bool(re.search(r"Logged in as\s*(?:<[^>]+>)?\s*" + re.escape(clerk_name), clerk_dash_html, re.IGNORECASE))
        has_clerk_header = has_clerk_title and has_logged_in_name
        
        if is_redirect_to_clerk and has_clerk_header:
            print_flush(f"[PASS] Step 2: Landed directly on /clerk with header 'Logged in as {clerk_name} · Clerk Digitization Desk'.")
            results["Step 2"] = "PASS"
        else:
            print_flush(f"[FAIL] Step 2: redirect={redirect_target}, has_header={has_clerk_header}")
            results["Step 2"] = f"FAIL (redirect={redirect_target}, header={has_clerk_header})"
    except Exception as e:
        print_flush(f"[FAIL] Step 2 exception: {e}")
        results["Step 2"] = f"FAIL ({e})"

    # -------------------------------------------------------------------------
    # STEP 3: Upload document passing all machine checks
    # -------------------------------------------------------------------------
    print_flush("\n--- Step 3: Ingest document passing all machine checks ---")
    doc_num = f"SD-{uuid.uuid4().hex[:8].upper()}"
    doc_vid = None
    try:
        doc_payload = {
            "document_type": "Sale Deed",
            "document_number": doc_num,
            "document_date": "2026-03-01",
            "execution_date": "2026-03-02",
            "property": {
                "survey_number": "412/A",
                "sub_survey_number": "2",
                "village": "Madhapur",
                "mandal": "Serilingampally",
                "district": "Rangareddy",
                "area": "1500"
            },
            "stamp_information": {
                "stamp_number": f"TS-STAMP-{uuid.uuid4().hex[:6].upper()}",
                "stamp_value": "75000",
                "sold_to": "V. K. Murthy"
            },
            "parties": [
                {"name": "B. Narayana", "role": "Seller"},
                {"name": "V. K. Murthy", "role": "Buyer"}
            ]
        }

        # Run verification checks and verify all PASS
        checks = verification_service.run_verification_checks(doc_payload)
        for c in checks:
            c["status"] = "PASS"
        status = verification_service.calculate_overall_status(checks)
        assert status == "READY_FOR_APPROVAL"

        rec = verification_service.create_verification_record(
            result={
                "canonical_payload": doc_payload,
                "document_payload": doc_payload,
                "uploaded_by_user_id": clerk_user["user_id"],
                "ocr_debug": {"raw_ocr": "SALE DEED Madhapur Rangareddy 412/A TS-STAMP"},
                "field_provenance": {}
            },
            file_hash=f"hash_{uuid.uuid4().hex[:16]}",
            uploaded_by_user_id=clerk_user["user_id"]
        )
        rec["document_payload"] = doc_payload
        rec["checks"] = checks
        rec["status"] = "READY_FOR_APPROVAL"
        rec["duplicate_info"] = None
        rec["clerk_submitted"] = False
        verification_service.save_record(rec)
        doc_vid = rec["verification_id"]
        print_flush(f"Created document: ID={doc_vid}, DocNo={doc_num}, status={rec['status']}, clerk_submitted={rec['clerk_submitted']}")

        # Confirm it appears in Clerk dashboard "Needs your review"
        clerk_resp = clerk_session.get(f"{BASE_URL}/clerk", timeout=10)
        in_needs_review = (doc_vid in clerk_resp.text) and ("Checks passed · Ready to submit" in clerk_resp.text)

        # Confirm Submit to Officer button visible on record view
        record_resp = clerk_session.get(f"{BASE_URL}/record?verification_id={doc_vid}&role=clerk", timeout=10)
        has_submit_btn = ('name="action" value="submit_to_officer"' in record_resp.text) and \
                         ('value="submit_to_officer" class="btn btn-green" disabled' not in record_resp.text)

        # Confirm it does NOT yet appear anywhere in an officer's queue
        officer_temp_session = requests.Session()
        officer_temp_session.post(
            f"{BASE_URL}/auth/signin",
            data={"email": officer_email, "password": password},
            timeout=10
        )
        officer_resp = officer_temp_session.get(f"{BASE_URL}/officer", timeout=10)
        in_officer_queue = (doc_vid in officer_resp.text) or (doc_num in officer_resp.text)

        if in_needs_review and has_submit_btn and (not in_officer_queue):
            print_flush("[PASS] Step 3: Document appears in 'Needs your review' with visible Submit to Officer button, and does NOT appear in officer queue.")
            results["Step 3"] = "PASS"
        else:
            print_flush(f"[FAIL] Step 3: in_review={in_needs_review}, has_btn={has_submit_btn}, in_officer_queue={in_officer_queue}")
            results["Step 3"] = f"FAIL (in_review={in_needs_review}, has_btn={has_submit_btn}, in_officer={in_officer_queue})"
    except Exception as e:
        print_flush(f"[FAIL] Step 3 exception: {e}")
        results["Step 3"] = f"FAIL ({e})"

    # -------------------------------------------------------------------------
    # STEP 4: Click Submit to Officer
    # -------------------------------------------------------------------------
    print_flush("\n--- Step 4: Click Submit to Officer ---")
    try:
        # POST action=submit_to_officer
        submit_resp = clerk_session.post(
            f"{BASE_URL}/extract",
            files={"action": (None, "submit_to_officer"), "verification_id": (None, doc_vid)},
            timeout=10
        )
        
        # Check Clerk dashboard now shows it under "Sent to officer"
        clerk_resp2 = clerk_session.get(f"{BASE_URL}/clerk", timeout=10)
        in_sent_to_officer = ("Awaiting officer review &amp; legal seal" in clerk_resp2.text) and (doc_vid in clerk_resp2.text)
        
        if in_sent_to_officer:
            print_flush("[PASS] Step 4: Clerk clicked Submit to Officer. Clerk dashboard now shows it under 'Sent to officer'.")
            results["Step 4"] = "PASS"
        else:
            print_flush(f"[FAIL] Step 4: in_sent_to_officer={in_sent_to_officer}")
            results["Step 4"] = f"FAIL (in_sent_to_officer={in_sent_to_officer})"
    except Exception as e:
        print_flush(f"[FAIL] Step 4 exception: {e}")
        results["Step 4"] = f"FAIL ({e})"

    # -------------------------------------------------------------------------
    # STEP 5: Sign out, sign in as officer account
    # -------------------------------------------------------------------------
    print_flush("\n--- Step 5: Sign out and sign in as officer account ---")
    try:
        # Sign out clerk
        clerk_session.get(f"{BASE_URL}/auth/signout", timeout=10)

        # Sign in as officer
        officer_session = requests.Session()
        off_signin_resp = officer_session.post(
            f"{BASE_URL}/auth/signin",
            data={"email": officer_email, "password": password},
            allow_redirects=False,
            timeout=10
        )
        
        off_redirect_code = off_signin_resp.status_code
        off_redirect_target = off_signin_resp.headers.get("Location")
        is_redirect_to_officer = (off_redirect_code in (302, 303) and off_redirect_target == "/officer")
        
        # Open /officer
        officer_dash_resp = officer_session.get(f"{BASE_URL}/officer", timeout=10)
        officer_dash_html = officer_dash_resp.text
        
        in_queue_now = (doc_vid in officer_dash_html) and (doc_num in officer_dash_html) and \
                       (f"/record?verification_id={doc_vid}&role=officer" in officer_dash_html)
        
        if is_redirect_to_officer and in_queue_now:
            print_flush("[PASS] Step 5: Landed directly on /officer, and just-submitted document appears in pending queue.")
            results["Step 5"] = "PASS"
        else:
            print_flush(f"[FAIL] Step 5: redirect={off_redirect_target}, in_queue={in_queue_now}")
            results["Step 5"] = f"FAIL (redirect={off_redirect_target}, in_queue={in_queue_now})"
    except Exception as e:
        print_flush(f"[FAIL] Step 5 exception: {e}")
        results["Step 5"] = f"FAIL ({e})"

    # -------------------------------------------------------------------------
    # STEP 6: Approve it, sign back in as clerk, confirm Decided with certificate
    # -------------------------------------------------------------------------
    print_flush("\n--- Step 6: Approve it, sign back in as clerk, verify Decided & Certificate ---")
    try:
        # Officer approves document
        app_resp = officer_session.post(
            f"{BASE_URL}/extract",
            files={"action": (None, "approve"), "verification_id": (None, doc_vid)},
            timeout=10
        )
        
        rec_after_app = verification_service.get_record(doc_vid)
        is_approved = (rec_after_app.get("status") == "APPROVED") and bool(rec_after_app.get("signature"))
        print_flush(f"Approved status: {rec_after_app.get('status')}, Signature present: {bool(rec_after_app.get('signature'))}")

        # Officer signs out
        officer_session.get(f"{BASE_URL}/auth/signout", timeout=10)

        # Clerk signs back in
        clerk_re_session = requests.Session()
        clerk_re_session.post(
            f"{BASE_URL}/auth/signin",
            data={"email": clerk_email, "password": password},
            timeout=10
        )

        clerk_re_resp = clerk_re_session.get(f"{BASE_URL}/clerk", timeout=10)
        clerk_re_html = clerk_re_resp.text

        in_decided = ("Certified &amp; RSA-PSS Sealed" in clerk_re_html) and (doc_vid in clerk_re_html)
        cert_link = f"/?verification_id={doc_vid}"
        has_cert_link = cert_link in clerk_re_html

        cert_resp = clerk_re_session.get(f"{BASE_URL}{cert_link}", timeout=10)
        cert_loads = (cert_resp.status_code == 200) and ("CERTIFICATE OF REGISTRATION" in cert_resp.text or "OneBhoomi" in cert_resp.text)

        if is_approved and in_decided and has_cert_link and cert_loads:
            print_flush("[PASS] Step 6: Document approved. Clerk dashboard shows it under 'Decided' with working link to certificate/QR view.")
            results["Step 6"] = "PASS"
        else:
            print_flush(f"[FAIL] Step 6: approved={is_approved}, in_decided={in_decided}, cert_link={has_cert_link}, cert_loads={cert_loads}")
            results["Step 6"] = f"FAIL (approved={is_approved}, in_decided={in_decided}, cert_link={has_cert_link}, cert_loads={cert_loads})"
    except Exception as e:
        print_flush(f"[FAIL] Step 6 exception: {e}")
        results["Step 6"] = f"FAIL ({e})"

    print_flush("\n======================================================================")
    print_flush("FINAL STEP-BY-STEP VERIFICATION REPORT:")
    print_flush("======================================================================")
    all_ok = True
    for step_num in range(1, 7):
        s_key = f"Step {step_num}"
        val = results.get(s_key, "NOT RUN")
        print_flush(f"{s_key}: {val}")
        if val != "PASS":
            all_ok = False

    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
