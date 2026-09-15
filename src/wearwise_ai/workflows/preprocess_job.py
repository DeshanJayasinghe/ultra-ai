from __future__ import annotations

from dataclasses import dataclass
from os import getenv
from socket import timeout as SocketTimeout
from typing import Protocol
from urllib.error import URLError
from urllib.request import urlopen

from wearwise_ai.application.preprocessing_service import (
    PreprocessingRequest,
    PreprocessingResult,
    PreprocessingService,
)
from wearwise_ai.infrastructure.internal_api_client import LaravelInternalApiClient


class BinaryFetcher(Protocol):
    def fetch(self, url: str) -> bytes: ...


class UrlBinaryFetcher:
    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        self._timeout_seconds = timeout_seconds or _media_download_timeout_seconds()

    def fetch(self, url: str) -> bytes:
        try:
            with urlopen(url, timeout=self._timeout_seconds) as response:
                return response.read()
        except (TimeoutError, SocketTimeout) as exc:
            raise RuntimeError(
                f"Could not download media asset bytes within {self._timeout_seconds:g}s."
            ) from exc
        except URLError as exc:
            raise RuntimeError("Could not download media asset bytes.") from exc


def _media_download_timeout_seconds() -> float:
    value = getenv("WEARWISE_AI_MEDIA_DOWNLOAD_TIMEOUT_SECONDS")
    if value is None or value == "":
        return 60.0

    try:
        return max(1.0, float(value))
    except ValueError:
        return 60.0


@dataclass(frozen=True, slots=True)
class PreprocessJobOutcome:
    request_id: str
    job_id: str
    asset_id: str
    result: PreprocessingResult


class PreprocessJobWorkflow:
    def __init__(
        self,
        *,
        internal_api: LaravelInternalApiClient,
        preprocessing_service: PreprocessingService,
        binary_fetcher: BinaryFetcher | None = None,
    ) -> None:
        self._internal_api = internal_api
        self._preprocessing_service = preprocessing_service
        self._binary_fetcher = binary_fetcher or UrlBinaryFetcher()

    def run(self, *, job_id: str, worker_id: str, request_id: str) -> PreprocessJobOutcome:
        job_response = self._internal_api.get_ai_job(job_id, request_id=request_id)
        job_data = job_response["data"]
        asset_id = job_data["media_asset_id"]

        self._internal_api.report_progress(
            job_id,
            request_id=request_id,
            worker_id=worker_id,
            progress_percent=10,
            status_message="fetching media asset",
        )

        access_response = self._internal_api.get_media_asset_access(asset_id, request_id=request_id)
        access = access_response["data"]["access"]
        media_asset = access_response["data"]["media_asset"]

        image_bytes = self._binary_fetcher.fetch(access["url"])

        self._internal_api.report_progress(
            job_id,
            request_id=request_id,
            worker_id=worker_id,
            progress_percent=40,
            status_message="validating and planning normalization",
        )

        result = self._preprocessing_service.process(
            PreprocessingRequest(
                asset_id=asset_id,
                content_type=media_asset["content_type"],
                image_bytes=image_bytes,
                checksum_sha256=media_asset.get("checksum_sha256"),
            )
        )

        self._internal_api.report_progress(
            job_id,
            request_id=request_id,
            worker_id=worker_id,
            progress_percent=90 if result.status == "ready" else 70,
            status_message="preprocessing ready" if result.status == "ready" else "preprocessing failed validation",
        )

        return PreprocessJobOutcome(
            request_id=request_id,
            job_id=job_id,
            asset_id=asset_id,
            result=result,
        )
