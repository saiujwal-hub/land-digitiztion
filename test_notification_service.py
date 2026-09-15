"""
test_notification_service.py - Comprehensive Unit Tests for Notification Service

Verifies:
1. Real dispatch when environment variables are configured (SMTP and Twilio REST).
2. Fail-safe console fallback when environment variables are missing.
3. Resilient error handling (failures return False without raising exceptions).
4. No secret leakage (passwords and auth tokens are not exposed in logs or errors).
5. notify_record_sealed pipeline safety.
"""

from email.message import EmailMessage
import logging
import os
import smtplib
from unittest.mock import MagicMock, patch

import pytest
import requests

from notification_service import (
    notify_record_sealed,
    send_email_notification,
    send_sms_notification,
)


@pytest.fixture
def clean_env(monkeypatch):
    """Ensure notification environment variables are cleanly managed."""
    env_vars = [
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USER",
        "SMTP_PASSWORD",
        "SMTP_FROM_ADDRESS",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_FROM_NUMBER",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def mock_record():
    return {
        "verification_id": "test-rec-1234567890",
        "approved_at": "2026-09-15T12:00:00Z",
        "signature": "mock_signature_data_long_enough_to_preview_securely_12345678",
        "document_payload": {
            "document_number": "DOC-9988",
            "document_type": "Sale Deed",
            "property": {
                "survey_number": "42/A",
                "village": "Ramanagara",
                "mandal": "Bidadi",
                "district": "Ramanagara",
            },
            "parties": [{"name": "Ramesh Kumar"}],
        },
    }


# ============================================================================
# EMAIL TESTS
# ============================================================================


def test_send_email_real_dispatch(clean_env):
    """Verify real SMTP dispatch is executed when all required env vars are present."""
    clean_env.setenv("SMTP_HOST", "smtp.example.com")
    clean_env.setenv("SMTP_PORT", "587")
    clean_env.setenv("SMTP_USER", "mailer@example.com")
    clean_env.setenv("SMTP_PASSWORD", "secret_pass_123")
    clean_env.setenv("SMTP_FROM_ADDRESS", "noreply@onebhoomi.gov.in")

    mock_smtp_instance = MagicMock()
    # Support context manager usage: with smtplib.SMTP(...) as server:
    mock_smtp_instance.__enter__.return_value = mock_smtp_instance

    with patch("smtplib.SMTP", return_value=mock_smtp_instance) as mock_smtp_cls:
        success = send_email_notification(
            to_email="citizen@example.com",
            subject="Test Sealing Subject",
            body="Hello, your record is sealed.",
            record={},
        )

        assert success is True
        mock_smtp_cls.assert_called_once_with("smtp.example.com", 587, timeout=10)
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with("mailer@example.com", "secret_pass_123")
        mock_smtp_instance.send_message.assert_called_once()

        sent_msg = mock_smtp_instance.send_message.call_args[0][0]
        assert isinstance(sent_msg, EmailMessage)
        assert sent_msg["To"] == "citizen@example.com"
        assert sent_msg["From"] == "noreply@onebhoomi.gov.in"
        assert sent_msg["Subject"] == "Test Sealing Subject"
        assert "Hello, your record is sealed." in sent_msg.get_content()


def test_send_email_fallback_when_env_missing(clean_env, caplog):
    """Verify console fallback mode triggers when required env vars are missing."""
    # Only partial config
    clean_env.setenv("SMTP_HOST", "smtp.example.com")

    with patch("smtplib.SMTP") as mock_smtp:
        with caplog.at_level(logging.WARNING):
            success = send_email_notification(
                to_email="citizen@example.com",
                subject="Test Fallback",
                body="Fallback body text",
                record={},
            )

        assert success is True
        mock_smtp.assert_not_called()
        assert "console fallback mode" in caplog.text


def test_send_email_failure_caught_and_returns_false(clean_env, caplog):
    """Verify SMTP exceptions are caught, logged without password exposure, and return False."""
    clean_env.setenv("SMTP_HOST", "smtp.example.com")
    clean_env.setenv("SMTP_PORT", "587")
    clean_env.setenv("SMTP_USER", "mailer@example.com")
    clean_env.setenv("SMTP_PASSWORD", "super_secret_password_do_not_leak")
    clean_env.setenv("SMTP_FROM_ADDRESS", "noreply@onebhoomi.gov.in")

    mock_smtp_instance = MagicMock()
    mock_smtp_instance.__enter__.return_value = mock_smtp_instance
    mock_smtp_instance.login.side_effect = smtplib.SMTPAuthenticationError(
        535, b"Authentication failed for user mailer@example.com"
    )

    with patch("smtplib.SMTP", return_value=mock_smtp_instance):
        with caplog.at_level(logging.ERROR):
            success = send_email_notification(
                to_email="citizen@example.com",
                subject="Test Error Handling",
                body="Body text",
                record={},
            )

        assert success is False
        assert "Failed to send email notification" in caplog.text
        # Ensure password was NOT leaked in logs
        assert "super_secret_password_do_not_leak" not in caplog.text


# ============================================================================
# SMS TESTS
# ============================================================================


def test_send_sms_real_dispatch(clean_env):
    """Verify Twilio REST API dispatch is executed when required env vars are present."""
    clean_env.setenv("TWILIO_ACCOUNT_SID", "mock_account_sid_12345")
    clean_env.setenv("TWILIO_AUTH_TOKEN", "mock_auth_token_xyz")
    clean_env.setenv("TWILIO_FROM_NUMBER", "+15005550006")

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.text = '{"sid": "SMmock123", "status": "queued"}'

    with patch("requests.post", return_value=mock_response) as mock_post:
        success = send_sms_notification(
            phone_number="+919876543210",
            message="Your land record has been sealed.",
            record={},
        )

        assert success is True
        mock_post.assert_called_once_with(
            "https://api.twilio.com/2010-04-01/Accounts/mock_account_sid_12345/Messages.json",
            data={
                "From": "+15005550006",
                "To": "+919876543210",
                "Body": "Your land record has been sealed.",
            },
            auth=("mock_account_sid_12345", "mock_auth_token_xyz"),
            timeout=10,
        )


def test_send_sms_fallback_when_env_missing(clean_env, caplog):
    """Verify console fallback mode triggers when Twilio env vars are not configured."""
    with patch("requests.post") as mock_post:
        with caplog.at_level(logging.WARNING):
            success = send_sms_notification(
                phone_number="+919876543210",
                message="Your land record has been sealed.",
                record={},
            )

        assert success is True
        mock_post.assert_not_called()
        assert "console fallback mode" in caplog.text


def test_send_sms_http_error_returns_false(clean_env, caplog):
    """Verify HTTP error status codes from Twilio return False and do not raise."""
    clean_env.setenv("TWILIO_ACCOUNT_SID", "mock_account_sid_12345")
    clean_env.setenv("TWILIO_AUTH_TOKEN", "sensitive_auth_token_never_log")
    clean_env.setenv("TWILIO_FROM_NUMBER", "+15005550006")

    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = '{"code": 21211, "message": "The \'To\' number is not a valid phone number."}'

    with patch("requests.post", return_value=mock_response):
        with caplog.at_level(logging.ERROR):
            success = send_sms_notification(
                phone_number="invalid-num",
                message="Hello",
                record={},
            )

        assert success is False
        assert "Twilio SMS dispatch failed to invalid-num: HTTP 400" in caplog.text
        assert "sensitive_auth_token_never_log" not in caplog.text


def test_send_sms_network_exception_returns_false(clean_env, caplog):
    """Verify network exceptions (e.g. timeout) return False and do not crash."""
    clean_env.setenv("TWILIO_ACCOUNT_SID", "mock_account_sid_12345")
    clean_env.setenv("TWILIO_AUTH_TOKEN", "sensitive_auth_token_never_log")
    clean_env.setenv("TWILIO_FROM_NUMBER", "+15005550006")

    with patch("requests.post", side_effect=requests.exceptions.ConnectTimeout("Connection timed out")):
        with caplog.at_level(logging.ERROR):
            success = send_sms_notification(
                phone_number="+919876543210",
                message="Hello",
                record={},
            )

        assert success is False
        assert "Failed to send SMS notification" in caplog.text
        assert "ConnectTimeout" in caplog.text
        assert "sensitive_auth_token_never_log" not in caplog.text


# ============================================================================
# PIPELINE INTEGRATION TESTS (notify_record_sealed)
# ============================================================================


def test_notify_record_sealed_end_to_end(mock_record, clean_env, tmp_path, monkeypatch):
    """Verify notify_record_sealed orchestrates email and SMS without blocking pipeline."""
    monkeypatch.setattr(
        "notification_service.NOTIFICATIONS_LOG_PATH",
        tmp_path / "test_notifications.log",
    )

    # In fallback mode
    result = notify_record_sealed(mock_record)

    assert result["status"] == "success"
    assert result["verification_id"] == "test-rec-1234567890"
    assert result["email_dispatched"] is True
    assert result["sms_dispatched"] is True
    assert (tmp_path / "test_notifications.log").exists()


def test_notify_record_sealed_resilient_to_dispatch_failure(mock_record, clean_env, tmp_path, monkeypatch):
    """Verify that dispatch failure records failure status but does not raise or fail the pipeline."""
    monkeypatch.setattr(
        "notification_service.NOTIFICATIONS_LOG_PATH",
        tmp_path / "test_notifications.log",
    )

    clean_env.setenv("SMTP_HOST", "smtp.example.com")
    clean_env.setenv("SMTP_USER", "u")
    clean_env.setenv("SMTP_PASSWORD", "p")
    clean_env.setenv("SMTP_FROM_ADDRESS", "f@ex.com")

    clean_env.setenv("TWILIO_ACCOUNT_SID", "AC123")
    clean_env.setenv("TWILIO_AUTH_TOKEN", "tok")
    clean_env.setenv("TWILIO_FROM_NUMBER", "+1234")

    # Simulate both failing
    with patch("smtplib.SMTP", side_effect=Exception("SMTP down")), \
         patch("requests.post", side_effect=Exception("Twilio down")):
        result = notify_record_sealed(mock_record)

        assert result["status"] == "success"
        assert result["email_dispatched"] is False
        assert result["sms_dispatched"] is False
