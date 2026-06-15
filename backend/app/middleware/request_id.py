import logging
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into every incoming request.

    The request ID is:
    - Added to the request state for use in route handlers
    - Added to the response headers for client-side correlation
    - Logged at the start of each request for log correlation

    If the client sends an X-Request-ID header, that value is used.
    Otherwise a new UUID4 is generated. This allows external systems
    to trace requests across service boundaries.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request, injecting request ID throughout lifecycle.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            Response with X-Request-ID header attached.
        """
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        logger.info(
            "Request started: %s %s request_id=%s",
            request.method,
            request.url.path,
            request_id,
        )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response