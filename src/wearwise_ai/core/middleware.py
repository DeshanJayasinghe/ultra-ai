from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    header_name = "X-Request-ID"

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        correlation_id = self._resolve_correlation_id(request)
        request.state.correlation_id = correlation_id

        response = await call_next(request)
        response.headers[self.header_name] = correlation_id

        return response

    def _resolve_correlation_id(self, request: Request) -> str:
        value = request.headers.get(self.header_name, "").strip()

        if not value or len(value) > 128:
            return str(uuid4())

        return value
