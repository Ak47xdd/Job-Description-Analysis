"""API-key hashing and authenticated encryption helpers."""

import base64
import hashlib
import hmac
import os

from cryptography.fernet import Fernet, InvalidToken

ENV_NAME = "API_KEY_ENCRYPTION_SECRET"


def _fernet() -> Fernet:
    """Build a stable Fernet key from the server-only encryption secret.

    The environment variable may be any sufficiently random secret. We derive
    a valid 32-byte Fernet key instead of requiring operators to paste a
    pre-encoded Fernet key, which avoids configuration-format failures.
    """
    secret = os.getenv(ENV_NAME, "").strip()
    if not secret:
        # Temporary compatibility path for deployments that have not added the
        # dedicated secret yet. SUPA_KEY remains server-side only.
        secret = os.getenv("SUPA_KEY", "").strip()
    if not secret:
        raise RuntimeError(f"{ENV_NAME} or SUPA_KEY must be configured")

    derived = hashlib.sha256(secret.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(derived))


def hash_api_key(api_key: str) -> str:
    """Return the deterministic SHA-256 fingerprint used for API lookup."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key for authenticated server-side retrieval."""
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
