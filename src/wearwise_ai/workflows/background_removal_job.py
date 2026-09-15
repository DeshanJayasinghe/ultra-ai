from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from wearwise_ai.application.background_removal_service import (
    BackgroundRemovalRequest,
    BackgroundRemovalResult,
    BackgroundRemovalService,
)
from wearwise_ai.application.preprocessing_service import (
    PreprocessingRequest,
    PreprocessingService,
)
from wearwise_ai.infrastructure.internal_api_client import LaravelInternalApiClient
from wearwise_ai.workflows.preprocess_job import BinaryFetcher, UrlBinaryFetcher


@dataclass(frozen=True, slots=True)
class BackgroundRemovalJobOutcome:
    request_id: str
    job_id: str
    asset_id: str
    preprocessing_status: str
    result: BackgroundRemovalResult | None


class BackgroundRemovalJobWorkflow:
    def __init__(
        self,
        *,
        internal_api: LaravelInternalApiClient,
        preprocessing_service: PreprocessingService,
        background_removal_service: BackgroundRemovalService,
        binary_fetcher: BinaryFetcher | None = None,
    ) -> None:
        self._internal_api = internal_api
        self._preprocessing_service = preprocessing_service
        self._background_removal_service = background_removal_service
        self._binary_fetcher = binary_fetcher or UrlBinaryFetcher()

    def run(self, *, job_id: str, worker_id: str, request_id: str) -> BackgroundRemovalJobOutcome:
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
            progress_percent=35,
            status_message="validating image input",
        )

        preprocessing = self._preprocessing_service.process(
            PreprocessingRequest(
                asset_id=asset_id,
                content_type=media_asset["content_type"],
                image_bytes=image_bytes,
                checksum_sha256=media_asset.get("checksum_sha256"),
            )
        )

        if preprocessing.status != "ready" or preprocessing.normalization_plan is None:
            self._internal_api.fail_job(
                job_id,
                request_id=request_id,
                payload={
                    "worker_id": worker_id,
                    "error_code": "BACKGROUND_REMOVAL_INPUT_INVALID",
                    "error_message": "Image validation failed before background removal.",
                    "status_message": "background removal input invalid",
                    "retryable": False,
                },
            )

            return BackgroundRemovalJobOutcome(
                request_id=request_id,
                job_id=job_id,
                asset_id=asset_id,
                preprocessing_status=preprocessing.status,
                result=None,
            )

        self._internal_api.report_progress(
            job_id,
            request_id=request_id,
            worker_id=worker_id,
            progress_percent=65,
            status_message="running background removal",
        )

        result = self._background_removal_service.process(
            BackgroundRemovalRequest(
                asset_id=asset_id,
                image_bytes=image_bytes,
                normalization_plan=preprocessing.normalization_plan,
            )
        )

        self._internal_api.complete_job(
            job_id,
            request_id=request_id,
            payload=result.to_complete_payload(worker_id=worker_id),
        )

        return BackgroundRemovalJobOutcome(
            request_id=request_id,
            job_id=job_id,
            asset_id=asset_id,
            preprocessing_status=preprocessing.status,
            result=result,
        )
