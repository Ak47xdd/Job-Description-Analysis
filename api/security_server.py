"""Production entrypoint that wraps the existing FastAPI app with security headers."""

import uvicorn

from JobAnalyze_API import app
from security_middleware import SecurityHeadersMiddleware

# Apply after CORS so security headers are present on normal and error responses.
app.add_middleware(SecurityHeadersMiddleware)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, server_header=False)
