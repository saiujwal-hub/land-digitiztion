import os
import requests

# Default URL for the external verification microservice
VERIFICATION_SERVICE_URL = os.environ.get("VERIFICATION_SERVICE_URL", "http://127.0.0.1:8004")


class VerificationError(Exception):
    """Raised when the external verification API request fails or returns an error."""
    pass


def create_verification(fields: dict, verify_url: str) -> dict:
    """
    Sends finalized land document fields and the explicit verify URL to the
    external verification microservice endpoint (/sign) to generate a secure
    signature, public key, and QR code.

    Args:
        fields (dict): A dictionary containing the finalized land document fields.
        verify_url (str): The URL to embed in the generated QR code.

    Returns:
        dict: The raw parsed JSON response containing:
              - "success" (bool)
              - "signature" (str)
              - "public_key" (str)
              - "qr_code" (str)

    Raises:
        VerificationError: If the external service is unavailable, returns a non-200
                           status code, or returns invalid JSON.
    """
    payload = {
        "fields": fields,
        "verify_url": verify_url
    }

    url = f"{VERIFICATION_SERVICE_URL.rstrip('/')}/sign"

    try:
        response = requests.post(url, json=payload, timeout=5)
    except requests.exceptions.RequestException as exc:
        raise VerificationError(
            f"External verification service is unreachable at {url}. Details: {exc}"
        )

    if response.status_code != 200:
        error_detail = "No error message provided."
        try:
            resp_data = response.json()
            error_detail = resp_data.get("detail", resp_data)
        except Exception:
            if response.text:
                error_detail = response.text
        raise VerificationError(
            f"Verification service error (HTTP {response.status_code}): {error_detail}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise VerificationError(
            f"Received invalid JSON response from verification service: {exc}"
        )

    return data


def verify_verification(fields: dict, signature: str) -> dict:
    """
    Sends document fields and signature to the external verification microservice
    endpoint (/verify) to cryptographically verify if the data matches the signature.

    Args:
        fields (dict): A dictionary containing the land document fields to verify.
        signature (str): The Base64 encoded signature to verify against.

    Returns:
        dict: The raw parsed JSON response containing:
              - "success" (bool)
              - "verified" (bool)

    Raises:
        VerificationError: If the external service is unavailable, returns a non-200
                           status code, or returns invalid JSON.
    """
    payload = {
        "fields": fields,
        "signature": signature
    }

    url = f"{VERIFICATION_SERVICE_URL.rstrip('/')}/verify"

    try:
        response = requests.post(url, json=payload, timeout=5)
    except requests.exceptions.RequestException as exc:
        raise VerificationError(
            f"External verification service is unreachable at {url}. Details: {exc}"
        )

    if response.status_code != 200:
        error_detail = "No error message provided."
        try:
            resp_data = response.json()
            error_detail = resp_data.get("detail", resp_data)
        except Exception:
            if response.text:
                error_detail = response.text
        raise VerificationError(
            f"Verification service error (HTTP {response.status_code}): {error_detail}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise VerificationError(
            f"Received invalid JSON response from verification service: {exc}"
        )

    return data
