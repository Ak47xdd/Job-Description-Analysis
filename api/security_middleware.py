"""Production security middleware for the FastAPI service.

This layer provides defense-in-depth around the existing application without
changing the public /web_analyze contract. Admin operations require explicit
admin API-key hashes configured in ADMIN_API_KEY_HASHES. The legacy public
API-key generation endpoint is disabled because it accepted an arbitrary email
without proving account ownership.
"""

import hashlib
import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


ADMIN_PATHS = {
    ("POST", "/news"),
    ("PATCH", "/news/"),
    ("GET", "/news"),
    ("POST", "/careers/openings"),
    ("PATCH", "/careers/openings/"),
    ("GET", "/careers/applications"),
}


def _is_admin_path(method: str, path: str) -> bool:
    if (method, path) in ADMIN_PATHS:
        return True
    return any(
        method == m and path.startswith(prefix)
        for m, prefix in ADMIN_PATHS
        if prefix.endswith("/")
    )


def _admin_hashes() -> set[str]:
    return {
        value.strip().lower()
        for value in os.getenv("ADMIN_API_KEY_HASHES", "").split(",")
        if value.strip()
    }


def _presented_key_hash(request: Request) -> str | None:
    key = request.headers.get("JobAnalyze_6k_Key")
    if not key:
        return None
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # /API/Generate previously accepted an arbitrary email and returned a
        # new API key. It is not a safe authentication mechanism, so disable it
        # until a real authenticated provisioning flow is used.
        if request.method == "POST" and request.url.path.rstrip("/") == "/API/Generate":
            return JSONResponse(
                {"detail": "API key generation requires an authenticated account."},
                status_code=410,
            )

        # A normal user API key must never grant access to administrative data
        # such as all career applications or unpublished news.
        if _is_admin_path(request.method, request.url.path):
            presented = _presented_key_hash(request)
            if not presented or presented not in _admin_hashes():
                return JSONResponse(
                    {"detail": "Administrator authorization required."},
                    status_code=403,
                )

        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=(), usb=(), browsing-topics=()",
        )
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains",
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("Vary", "Origin")
        response.headers.pop("Server", None)
        return response
