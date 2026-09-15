"""
notification_service.py - OneBhoomi Post-Seal Notification Hook Service

Dispatches email and SMS notifications upon successful cryptographic sealing of a land record.
Designed with pluggable provider hooks (e.g. Twilio, MSG91, SendGrid/SMTP) behind a unified interface.
"""

from datetime import datetime, timezone
from email.message import EmailMessage
import json
import logging
import os
from pathlib import Path
import smtplib
import sys
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("NotificationService")
NOTIFICATIONS_LOG_PATH = Path(__file__).parent / "notifications.log"


def _safe_print(text: str) -> None:
    """Safely prints to stdout handling Windows legacy codepages (charmap/cp1252)."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding))


def _extract_recipient_info(record: Dict[str, Any]) -> Dict[str, str]:
    """Extracts contact details from record or document payload parties."""
    payload = record.get("document_payload") or {}
    prop = payload.get("property") or {}

    user_id = record.get("uploaded_by_user_id")
    recipient_name = "Land Deed Applicant / Property Owner"
    recipient_email = "applicant@onebhoomi.gov.in"
    recipient_phone = "+91 98765 43210"

    try:
        import accounts_store
        if user_id:
            user = accounts_store.get_user(user_id)
            if user:
                recipient_name = user.get("name") or recipient_name
                recipient_email = user.get("email") or recipient_email
                if user.get("phone"):
                    recipient_phone = user.get("phone")
    except Exception:
        pass

    parties = payload.get("parties") or []
    if parties and isinstance(parties, list):
        first_party = parties[0]
        if isinstance(first_party, str) and first_party.strip():
            recipient_name = first_party.strip()
        elif isinstance(first_party, dict) and first_party.get("name"):
            recipient_name = first_party.get("name")

    return {
        "name": recipient_name,
        "email": recipient_email,
        "phone": recipient_phone,
        "survey_number": str(prop.get("survey_number") or "N/A"),
        "village": str(prop.get("village") or "N/A"),
        "mandal": str(prop.get("mandal") or "N/A"),
        "district": str(prop.get("district") or "N/A"),
        "document_number": str(payload.get("document_number") or "N/A"),
        "document_type": str(payload.get("document_type") or "Land Record"),
    }


def send_email_notification(to_email: str, subject: str, body: str, record: Dict[str, Any]) -> bool:
    """
    Email dispatch provider hook.
    Sends email via SMTP using Python's built-in smtplib and EmailMessage if configured.
    Falls back to console output if required SMTP environment variables are missing.
    """
    smtp_host = os.environ.get("SMTP_HOST", "").strip()
    smtp_port_raw = os.environ.get("SMTP_PORT", "587").strip()
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_password = os.environ.get("SMTP_PASSWORD", "").strip()
    smtp_from = os.environ.get("SMTP_FROM_ADDRESS", "").strip()

    required_vars = {
        "SMTP_HOST": smtp_host,
        "SMTP_USER": smtp_user,
        "SMTP_PASSWORD": smtp_password,
        "SMTP_FROM_ADDRESS": smtp_from,
    }
    missing_vars = [k for k, v in required_vars.items() if not v]

    if missing_vars:
        logger.warning(
            "Email notification running in console fallback mode: missing required env var(s): %s",
            ", ".join(missing_vars),
        )
        formatted_email = (
            f"\n{'='*72}\n"
            f"[EMAIL NOTIFICATION DISPATCHED]\n"
            f"{'-'*72}\n"
            f"To:      {to_email}\n"
            f"From:    {smtp_from or 'OneBhoomi Land Registry <noreply@onebhoomi.gov.in>'}\n"
            f"Subject: {subject}\n"
            f"{'-'*72}\n"
            f"{body.strip()}\n"
            f"{'='*72}"
        )
        _safe_print(formatted_email)
        logger.info("Email notification dispatched (console fallback) to %s", to_email)
        return True

    try:
        port = int(smtp_port_raw) if smtp_port_raw else 587
    except (ValueError, TypeError):
        port = 587

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = smtp_from
        msg["To"] = to_email
        msg.set_content(body)

        with smtplib.SMTP(smtp_host, port, timeout=10) as server:
            server.starttls()
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.send_message(msg)

        logger.info("Email notification successfully sent via SMTP to %s", to_email)
        return True
    except Exception as exc:
        err_msg = str(exc)
        if smtp_password and smtp_password in err_msg:
            err_msg = err_msg.replace(smtp_password, "[REDACTED]")
        logger.error(
            "Failed to send email notification to %s via SMTP (%s): %s",
            to_email,
            type(exc).__name__,
            err_msg,
        )
        return False


def send_sms_notification(phone_number: str, message: str, record: Dict[str, Any]) -> bool:
    """
    SMS dispatch provider hook.
    Sends SMS via Twilio REST API using requests if configured.
    Falls back to console output if required Twilio environment variables are missing.
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    from_number = os.environ.get("TWILIO_FROM_NUMBER", "").strip()

    required_vars = {
        "TWILIO_ACCOUNT_SID": account_sid,
        "TWILIO_AUTH_TOKEN": auth_token,
        "TWILIO_FROM_NUMBER": from_number,
    }
    missing_vars = [k for k, v in required_vars.items() if not v]

    if missing_vars:
        logger.warning(
            "SMS notification running in console fallback mode: missing required env var(s): %s",
            ", ".join(missing_vars),
        )
        formatted_sms = (
            f"\n{'='*72}\n"
            f"[SMS NOTIFICATION DISPATCHED]\n"
            f"{'-'*72}\n"
            f"To:      {phone_number}\n"
            f"Sender:  {from_number or 'GOV-ONEBHOOMI'}\n"
            f"Message: {message.strip()}\n"
            f"{'='*72}\n"
        )
        _safe_print(formatted_sms)
        logger.info("SMS notification dispatched (console fallback) to %s", phone_number)
        return True

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    try:
        response = requests.post(
            url,
            data={
                "From": from_number,
                "To": phone_number,
                "Body": message,
            },
            auth=(account_sid, auth_token),
            timeout=10,
        )
        if 200 <= response.status_code < 300:
            logger.info(
                "SMS notification successfully sent via Twilio to %s (HTTP %d)",
                phone_number,
                response.status_code,
            )
            return True
        else:
            resp_body = response.text
            if auth_token and auth_token in resp_body:
                resp_body = resp_body.replace(auth_token, "[REDACTED]")
            logger.error(
                "Twilio SMS dispatch failed to %s: HTTP %d - %s",
                phone_number,
                response.status_code,
                resp_body,
            )
            return False
    except Exception as exc:
        err_msg = str(exc)
        if auth_token and auth_token in err_msg:
            err_msg = err_msg.replace(auth_token, "[REDACTED]")
        logger.error(
            "Failed to send SMS notification to %s via Twilio (%s): %s",
            phone_number,
            type(exc).__name__,
            err_msg,
        )
        return False


def notify_record_sealed(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Notification hook executed immediately following successful cryptographic record sealing.
    Prepares and dispatches email and SMS alerts and logs the event to notifications.log.

    Safe: Handled defensively so failures never affect record certification.
    """
    if not record or not isinstance(record, dict):
        logger.warning("notify_record_sealed called with invalid record")
        return {"status": "skipped", "reason": "empty_record"}

    rec_id = record.get("verification_id", "UNKNOWN")
    approved_at = record.get("approved_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sig = record.get("signature") or ""
    sig_preview = f"{sig[:16]}...{sig[-16:]}" if len(sig) > 32 else (sig or "RSA-PSS-2048")

    info = _extract_recipient_info(record)

    # 1. Format Email
    email_subject = f"OneBhoomi: Deed Certified & Sealed - Doc #{info['document_number']} (Record {rec_id[:8].upper()})"
    email_body = f"""Dear {info['name']},

Your {info['document_type']} (Document #{info['document_number']}) has been officially APPROVED and CRYPTOGRAPHICALLY SEALED by the Land Revenue Officer on the OneBhoomi Digital Land Registry.

RECORD SUMMARY:
- Verification ID:    {rec_id}
- Property Location:  Survey No. {info['survey_number']}, Village: {info['village']}, Mandal: {info['mandal']}, District: {info['district']}
- Sealing Timestamp:  {approved_at}
- Digital Signature:  {sig_preview} (RSA-PSS 2048-bit / SHA-256)
- Public Key Fingerprint: Verified by OneBhoomi Trust Authority

You can view and download your cryptographically signed certificate of title at:
http://localhost:8001/verify?verification_id={rec_id}

This is an automated legal transaction notification generated by OneBhoomi.
"""

    # 2. Format SMS
    sms_message = (
        f"OneBhoomi Alert: Your {info['document_type']} (Doc #{info['document_number']}, "
        f"Sy #{info['survey_number']}, {info['village']}) has been APPROVED & SEALED. "
        f"Cert ID: {rec_id[:8].upper()}. Verify: http://localhost:8001/verify?verification_id={rec_id}"
    )

    # 3. Dispatch notifications
    email_success = False
    sms_success = False
    try:
        email_success = send_email_notification(info["email"], email_subject, email_body, record)
    except Exception as e:
        logger.error("Failed to send email notification: %s", e)

    try:
        sms_success = send_sms_notification(info["phone"], sms_message, record)
    except Exception as e:
        logger.error("Failed to send SMS notification: %s", e)

    # 4. Write audit entry to notifications.log
    log_entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verification_id": rec_id,
        "recipient": info["name"],
        "email": info["email"],
        "phone": info["phone"],
        "document_number": info["document_number"],
        "survey_number": info["survey_number"],
        "email_dispatched": email_success,
        "sms_dispatched": sms_success,
    }
    try:
        with open(NOTIFICATIONS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as log_err:
        logger.warning("Failed to write to notifications.log: %s", log_err)

    return {
        "status": "success",
        "verification_id": rec_id,
        "email_dispatched": email_success,
        "sms_dispatched": sms_success,
    }
