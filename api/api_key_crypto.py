"""Encryption helpers for JobSelect API keys.

API keys are encrypted for authenticated display/retrieval and separately
hashed for lookup/verification. The encryption key must live only in the
server environment, never in Supabase or the frontend.
"""

import base64
import hashlib
import hmac
import os
import warnings

from cryptography.fernet import Fernet, InvalidToken

ENV_NAME = "API_KEY_ENCRYPTION_SECRET"


def _fernet() -> Fernet:
    secret = os.getenv(ENV_NAME, "").strip()

    # Backwards-compatible emergency fallback. This keeps an existing
    # deployment working if the new environment variable has not been added
    # yet. SUPA_KEY is already server-side only. Production deployments should
    # still set API_KEY_ENCRYPTION_SECRET to a dedicated Fernet key.
    if not secret:
        fallback = os.getenv("SUPA_KEY", "").strip()
        if not fallback:
            raise RuntimeError(
                f"{ENV_NAME} must be configured, or SUPA_KEY must be available"
            )
        secret = base64.urlsafe_b64encode(
            hashlib.sha256(fallback.encode("utf-8")).digest()
        ).decode("ascii")
        warnings.warn(
            f"{ENV_NAME} is not configured; using a compatibility key derived "
            "from SUPA_KEY. Configure a dedicated encryption secret in production.",
            RuntimeWarning,
            stacklevel=2,
        )

    try:
        return Fernet(secret.encode("ascii"))
    except Exception as exc:
        raise RuntimeError(f"{ENV_NAME} is not a valid Fernet key") from exc


def hash_api_key(api_key: str) -> str:
    """Return the deterministic SHA-256 fingerprint used for API lookup."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key for authenticated, server-side retrieval."""
    return _fernet().encrypt(api_key.encode("utf-8")).decode("ascii")


def decrypt_api_key(ciphertext: str) -> str:
    """Decrypt a stored API key."""
    try:
        return _fernet().decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except (InvalidToken, UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("Stored API key could not be decrypted") from exc


def keys_match(candidate: str, stored_hash: str) -> bool:
    """Constant-time comparison of a candidate API key against its hash."""
    if not stored_hash:
        return False
    return hmac.compare_digest(hash_api_key(candidate), stored_hash.lower())


def is_encrypted(value: str | None) -> bool:
    """Best-effort check for a Fernet ciphertext."""
    if not value:
        return False
    try:
        base64.urlsafe_b64decode(value.encode("ascii"))
        return value.startswith("gAAAAA")
    except Exception:
        return False
