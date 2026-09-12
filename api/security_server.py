"""Production entrypoint that wraps the existing FastAPI app with security headers."""

import os

import uvicorn

from JobAnalyze_API import app
from security_middleware import SecurityHeadersMiddleware

app.add_middleware(SecurityHeadersMiddleware)


if __name__ == "__main__":
    # Render supplies PORT dynamically. Hard-coding 5000 can make the service
    # fail health/port detection even when the application starts correctly.
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(
        "security_server:app",
        host="0.0.0.0",
        port=port,
        server_header=False,
        workers=1,
    )
