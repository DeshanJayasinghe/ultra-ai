from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wearwise_ai.application.clothing_classification_service import (
    CLOTHING_CLASSIFICATION_MODEL_FAMILY,
    CLOTHING_CLASSIFICATION_MODEL_NAME,
    CLOTHING_CLASSIFICATION_MODEL_VERSION,
    CLOTHING_CLASSIFICATION_PROCESSING_VERSION,
    ClothingClassificationRequest,
    ClothingClassificationResult,
    ClothingClassificationService,
)
from wearwise_ai.core.config import get_settings
from wearwise_ai.infrastructure.internal_api_client import LaravelInternalApiClient
from wearwise_ai.storage import ArtifactReader, LocalArtifactStore
from wearwise_ai.workflows.artifact_validation import validate_cleaned_artifact_or_fail
from wearwise_ai.workflows.colour_extraction_job import (
    ArtifactLocation,
    _artifact_location,
    _background_removed_options,
)
from wearwise_ai.workflows.preprocess_job import BinaryFetcher, UrlBinaryFetcher


@dataclass(frozen=True, slots=True)
class ClothingClassificationJobOutcome:
    request_id: str
    job_id: str
    asset_id: str
    status: str
    result: ClothingClassificationResult | None


class ClothingClassificationJobWorkflow:
    def __init__(
        self,
        *,
        internal_api: LaravelInternalApiClient,
        classification_service: ClothingClassificationService,
        binary_fetcher: BinaryFetcher | None = None,
        artifact_reader: ArtifactReader | None = None,
    ) -> None:
        self._internal_api = internal_api
        self._classification_service = classification_service
        self._binary_fetcher = binary_fetcher or UrlBinaryFetcher()
        self._artifact_reader = artifact_reader or LocalArtifactStore(
            root=Path(get_settings().artifact_root)
        )

    def run(self, *, job_id: str, worker_id: str, request_id: str) -> ClothingClassificationJobOutcome:
        job_response = self._internal_api.get_ai_job(job_id, request_id=request_id)
        job_data = job_response["data"]
        asset_id = job_data["media_asset_id"]
        options = job_data.get("options") if isinstance(job_data, dict) else {}
        background_removed = _background_removed_options(options)

        self._internal_api.report_progress(
            job_id,
            request_id=request_id,
            worker_id=worker_id,
            progress_percent=15,
            status_message="checking background removal quality",
        )

        if background_removed.get("review_required") is True:
            payload = _review_required_payload(
                worker_id=worker_id,
                quality_status=background_removed.get("quality_status"),
                quality_reasons=background_removed.get("quality_reasons"),
            )
            self._internal_api.complete_job(job_id, request_id=request_id, payload=payload)
            return ClothingClassificationJobOutcome(
                request_id=request_id,
                job_id=job_id,
                asset_id=asset_id,
                status="needs_review",
                result=None,
            )

        artifact_location = _artifact_location(background_removed)
        if artifact_location is None:
            self._internal_api.fail_job(
                job_id,
                request_id=request_id,
                payload={
                    "worker_id": worker_id,
                    "error_code": "CLOTHING_CLASSIFICATION_INPUT_MISSING",
                    "error_message": "Cleaned garment image access URL is missing.",
                    "status_message": "clothing classification input missing",
                    "retryable": False,
                },
            )
            return ClothingClassificationJobOutcome(
                request_id=request_id,
                job_id=job_id,
                asset_id=asset_id,
                status="failed",
                result=None,
            )

        self._internal_api.report_progress(
            job_id,
            request_id=request_id,
            worker_id=worker_id,
            progress_percent=45,
            status_message="fetching cleaned garment image",
        )
        image_bytes = self._read_artifact(artifact_location)
        content_type = str(background_removed.get("content_type") or "image/webp")

        self._internal_api.report_progress(
            job_id,
            request_id=request_id,
            worker_id=worker_id,
            progress_percent=60,
            status_message="validating cleaned garment image",
        )
        if not validate_cleaned_artifact_or_fail(
            internal_api=self._internal_api,
            job_id=job_id,
            worker_id=worker_id,
            request_id=request_id,
            image_bytes=image_bytes,
            content_type=content_type,
            error_code="CLOTHING_CLASSIFICATION_INPUT_INVALID",
            status_message="clothing classification input invalid",
        ):
            return ClothingClassificationJobOutcome(
                request_id=request_id,
                job_id=job_id,
                asset_id=asset_id,
                status="failed",
                result=None,
            )

        self._internal_api.report_progress(
            job_id,
            request_id=request_id,
            worker_id=worker_id,
            progress_percent=75,
            status_message="classifying garment category",
        )
        result = self._classification_service.classify(
            ClothingClassificationRequest(
                asset_id=asset_id,
                image_bytes=image_bytes,
                content_type=content_type,
            )
        )
        self._internal_api.complete_job(
            job_id,
            request_id=request_id,
            payload=result.to_complete_payload(worker_id=worker_id),
        )

        return ClothingClassificationJobOutcome(
            request_id=request_id,
            job_id=job_id,
            asset_id=asset_id,
            status=result.status,
            result=result,
        )

    def _read_artifact(self, location: ArtifactLocation) -> bytes:
        if location.kind == "url":
            return self._binary_fetcher.fetch(location.value)

        return self._artifact_reader.read_bytes(location.value)


def _review_required_payload(
    *,
    worker_id: str,
    quality_status: object,
    quality_reasons: object,
) -> dict[str, object]:
    reasons = quality_reasons if isinstance(quality_reasons, list) else []
    return {
        "worker_id": worker_id,
        "model_family": CLOTHING_CLASSIFICATION_MODEL_FAMILY,
        "model_name": CLOTHING_CLASSIFICATION_MODEL_NAME,
        "model_version": CLOTHING_CLASSIFICATION_MODEL_VERSION,
        "processing_version": CLOTHING_CLASSIFICATION_PROCESSING_VERSION,
        "status_message": "clothing classification skipped; background removal needs review",
        "result": {
            "category": None,
            "subcategory": None,
            "review_required": True,
            "review_reasons": reasons,
        },
        "confidence": None,
        "resource_metrics": {
            "classification_status_reason": "background_removal_needs_review",
            "background_removal_quality_status": (
                quality_status if isinstance(quality_status, str) else "needs_review"
            ),
            "background_removal_quality_reasons": ",".join(
                reason for reason in reasons if isinstance(reason, str)
            ),
        },
    }
