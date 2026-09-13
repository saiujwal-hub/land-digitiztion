"""
notification_service.py - OneBhoomi Post-Seal Notification Hook Service

Dispatches email and SMS notifications upon successful cryptographic sealing of a land record.
Designed with pluggable provider hooks (e.g. Twilio, MSG91, SendGrid/SMTP) behind a unified interface.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional

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
    Pluggable for SendGrid, SES, or SMTP. Formats and prints to console for demo.
    """
    formatted_email = (
        f"\n{'='*72}\n"
        f"[EMAIL NOTIFICATION DISPATCHED]\n"
        f"{'-'*72}\n"
        f"To:      {to_email}\n"
        f"From:    OneBhoomi Land Registry <noreply@onebhoomi.gov.in>\n"
        f"Subject: {subject}\n"
        f"{'-'*72}\n"
        f"{body.strip()}\n"
        f"{'='*72}"
    )
    _safe_print(formatted_email)
    logger.info("Email notification dispatched to %s", to_email)
    return True


def send_sms_notification(phone_number: str, message: str, record: Dict[str, Any]) -> bool:
    """
    SMS dispatch provider hook.
    Pluggable for Twilio or MSG91 gateway. Formats and prints to console for demo.
    """
    formatted_sms = (
        f"\n{'='*72}\n"
        f"[SMS NOTIFICATION DISPATCHED]\n"
        f"{'-'*72}\n"
        f"To:      {phone_number}\n"
        f"Sender:  GOV-ONEBHOOMI\n"
        f"Message: {message.strip()}\n"
        f"{'='*72}\n"
    )
    _safe_print(formatted_sms)
    logger.info("SMS notification dispatched to %s", phone_number)
    return True


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
