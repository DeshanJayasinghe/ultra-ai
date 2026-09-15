from __future__ import annotations

import json
from dataclasses import dataclass
from time import sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class InternalApiError(RuntimeError):
    """Raised when an internal API call fails."""


@dataclass(frozen=True, slots=True)
class InternalApiConfig:
    base_url: str
    bearer_token: str
    timeout_seconds: float = 10.0
    max_attempts: int = 2
    retry_backoff_seconds: float = 0.25


class LaravelInternalApiClient:
    def __init__(self, config: InternalApiConfig) -> None:
        self._config = config

    def get_ai_job(self, job_id: str, *, request_id: str) -> dict[str, Any]:
        return self._json_request("GET", f"/internal/v1/ai/jobs/{job_id}", request_id=request_id)

    def claim_next_job(
        self,
        *,
        request_id: str,
        worker_id: str,
        job_types: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "worker_id": worker_id,
        }
        if job_types:
            payload["job_types"] = job_types

        return self._json_request(
            "POST",
            "/internal/v1/ai/jobs/claim-next",
            request_id=request_id,
            payload=payload,
        )

    def get_media_asset_access(
        self, asset_id: str, *, request_id: str, variant: str | None = None
    ) -> dict[str, Any]:
        path = f"/internal/v1/media/assets/{asset_id}/access"
        if variant:
            path = f"{path}?variant={variant}"

        return self._json_request(
            "GET",
            path,
            request_id=request_id,
        )

    def report_progress(
        self,
        job_id: str,
        *,
        request_id: str,
        worker_id: str,
        progress_percent: int,
        status_message: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "worker_id": worker_id,
            "progress_percent": progress_percent,
        }
        if status_message is not None:
            payload["status_message"] = status_message

        try:
            return self._json_request(
                "POST",
                f"/internal/v1/ai/jobs/{job_id}/progress",
                request_id=request_id,
                payload=payload,
            )
        except InternalApiError as exc:
            print(
                "progress report failed; continuing: "
                f"job_id={job_id} progress_percent={progress_percent} error={exc}"
            )
            return {}

    def complete_job(
        self,
        job_id: str,
        *,
        request_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self._json_request(
            "POST",
            f"/internal/v1/ai/jobs/{job_id}/complete",
            request_id=request_id,
            payload=payload,
        )

    def fail_job(
        self,
        job_id: str,
        *,
        request_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self._json_request(
            "POST",
            f"/internal/v1/ai/jobs/{job_id}/fail",
            request_id=request_id,
            payload=payload,
        )

    def _json_request(
        self,
        method: str,
        path: str,
        *,
        request_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            url=f"{self._config.base_url.rstrip('/')}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self._config.bearer_token}",
                "Content-Type": "application/json",
                "X-Request-ID": request_id,
            },
        )

        raw = self._send_with_retries(request, method=method, path=path)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InternalApiError("Internal API response was not valid JSON.") from exc

    def _send_with_retries(self, request: Request, *, method: str, path: str) -> str:
        attempts = max(1, self._config.max_attempts)
        last_error: InternalApiError | None = None
        for attempt in range(1, attempts + 1):
            try:
                with urlopen(request, timeout=self._config.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                message = f"Internal API {method} {path} failed with HTTP {exc.code}."
                if detail:
                    message = f"{message} {detail}"
                raise InternalApiError(message) from exc
            except URLError as exc:
                last_error = InternalApiError(
                    f"Internal API {method} {path} could not be completed."
                )
                if attempt >= attempts:
                    raise last_error from exc
            except TimeoutError as exc:
                last_error = InternalApiError(f"Internal API {method} {path} timed out.")
                if attempt >= attempts:
                    raise last_error from exc

            sleep(self._config.retry_backoff_seconds * attempt)

        raise last_error or InternalApiError(f"Internal API {method} {path} failed.")
