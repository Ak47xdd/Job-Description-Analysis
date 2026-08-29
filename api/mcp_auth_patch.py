"""Compatibility helper for MCP Bearer authentication.

The FastAPI MCP transport commonly forwards API credentials as:
    Authorization: Bearer <JobAnalyze API key>

The existing JobAnalyze API uses the custom JobAnalyze_6k_Key header.
This module exposes a small helper so the authentication layer can accept
both forms without changing the existing API-key validation/database logic.
"""

from fastapi import Header


def resolve_jobanalyze_api_key(
    custom_api_key: str | None,
    authorization: str | None,
) -> str | None:
    """Return the JobAnalyze key from either supported HTTP auth form."""
    if custom_api_key:
        return custom_api_key

    if not authorization:
        return None

    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None

    credentials = credentials.strip()
    return credentials or None
