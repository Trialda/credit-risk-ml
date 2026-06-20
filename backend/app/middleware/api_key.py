import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

logger = logging.getLogger(__name__)

PROTECTED_PATHS = {"/predict", "/explain"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Enforce API key authentication on protected endpoints.

    Checks the X-API-Key header on /predict and /explain.
    All other endpoints are unauthenticated.

    If API_KEY is not configured, authentication is skipped, 
    this allows the system to run in development without a key.
    """

    async def dispatch(self, request: Request, call_next):
        """Check API key on protected paths.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            403 response if key is missing or invalid,
            otherwise the normal response.
        """
        if not settings.api_key:
            return await call_next(request)

        if request.url.path in PROTECTED_PATHS:
            api_key = request.headers.get("X-API-Key")

            if not api_key:
                logger.warning(
                    "Missing API key on %s from %s",
                    request.url.path,
                    request.client.host if request.client else "unknown",
                )
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Missing X-API-Key header"},
                )

            if api_key != settings.api_key:
                logger.warning(
                    "Invalid API key on %s from %s",
                    request.url.path,
                    request.client.host if request.client else "unknown",
                )
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Invalid API key"},
                )

        return await call_next(request)