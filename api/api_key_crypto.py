"""API-key hashing and authenticated encryption helpers."""

import base64
import hashlib
import hmac
import os

from cryptography.fernet import Fernet, InvalidToken

ENV_NAME = "API_KEY_ENCRYPTION_SECRET"


def _fernet() -> Fernet:
    """Build a stable Fernet key from a server-only secret.

    Prefer the dedicated encryption secret. The Supabase service-role key is
    accepted only as a backwards-compatible fallback so existing deployments
    do not stop provisioning API keys during the migration.
    """
    secret = (
        os.getenv(ENV_NAME)
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPA_KEY")
        or ""
    ).strip()

    if not secret:
        raise RuntimeError(
            f"{ENV_NAME} or SUPABASE_SERVICE_ROLE_KEY (or SUPA_KEY) must be configured"
        )

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
