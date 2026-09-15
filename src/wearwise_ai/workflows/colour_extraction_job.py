from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wearwise_ai.application.colour_extraction_service import (
    COLOUR_EXTRACTION_MODEL_FAMILY,
    COLOUR_EXTRACTION_MODEL_NAME,
    COLOUR_EXTRACTION_MODEL_VERSION,
    COLOUR_EXTRACTION_PROCESSING_VERSION,
    ColourExtractionRequest,
    ColourExtractionResult,
    ColourExtractionService,
)
from wearwise_ai.core.config import get_settings
from wearwise_ai.infrastructure.internal_api_client import LaravelInternalApiClient
from wearwise_ai.storage import ArtifactReader, LocalArtifactStore
from wearwise_ai.workflows.artifact_validation import validate_cleaned_artifact_or_fail
from wearwise_ai.workflows.preprocess_job import BinaryFetcher, UrlBinaryFetcher


@dataclass(frozen=True, slots=True)
class ColourExtractionJobOutcome:
    request_id: str
    job_id: str
    asset_id: str
    status: str
    result: ColourExtractionResult | None


class ColourExtractionJobWorkflow:
    def __init__(
        self,
        *,
        internal_api: LaravelInternalApiClient,
        colour_extraction_service: ColourExtractionService,
        binary_fetcher: BinaryFetcher | None = None,
        artifact_reader: ArtifactReader | None = None,
    ) -> None:
        self._internal_api = internal_api
        self._colour_extraction_service = colour_extraction_service
        self._binary_fetcher = binary_fetcher or UrlBinaryFetcher()
        self._artifact_reader = artifact_reader or LocalArtifactStore(
            root=Path(get_settings().artifact_root)
        )

    def run(self, *, job_id: str, worker_id: str, request_id: str) -> ColourExtractionJobOutcome:
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
            return ColourExtractionJobOutcome(
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
                    "error_code": "COLOUR_EXTRACTION_INPUT_MISSING",
                    "error_message": "Cleaned garment image access URL is missing.",
                    "status_message": "colour extraction input missing",
                    "retryable": False,
                },
            )
            return ColourExtractionJobOutcome(
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
            error_code="COLOUR_EXTRACTION_INPUT_INVALID",
            status_message="colour extraction input invalid",
        ):
            return ColourExtractionJobOutcome(
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
            status_message="extracting garment colours",
        )
        result = self._colour_extraction_service.extract(
            ColourExtractionRequest(
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

        return ColourExtractionJobOutcome(
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


def _background_removed_options(options: object) -> dict[str, object]:
    if not isinstance(options, dict):
        return {}

    value = options.get("background_removed")
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True, slots=True)
class ArtifactLocation:
    kind: str
    value: str


def _artifact_location(background_removed: dict[str, object]) -> ArtifactLocation | None:
    access_url = background_removed.get("access_url")
    if isinstance(access_url, str) and access_url != "":
        return ArtifactLocation(kind="url", value=access_url)

    storage_key = background_removed.get("storage_key")
    if isinstance(storage_key, str) and storage_key != "":
        return ArtifactLocation(kind="storage_key", value=storage_key)

    return None


def _review_required_payload(
    *,
    worker_id: str,
    quality_status: object,
    quality_reasons: object,
) -> dict[str, object]:
    reasons = quality_reasons if isinstance(quality_reasons, list) else []
    return {
        "worker_id": worker_id,
        "model_family": COLOUR_EXTRACTION_MODEL_FAMILY,
        "model_name": COLOUR_EXTRACTION_MODEL_NAME,
        "model_version": COLOUR_EXTRACTION_MODEL_VERSION,
        "processing_version": COLOUR_EXTRACTION_PROCESSING_VERSION,
        "status_message": "colour extraction skipped; background removal needs review",
        "result": {
            "primary_colour": None,
            "secondary_colour": None,
            "review_required": True,
            "review_reasons": reasons,
        },
        "confidence": None,
        "resource_metrics": {
            "colour_extraction_status_reason": "background_removal_needs_review",
            "background_removal_quality_status": (
                quality_status if isinstance(quality_status, str) else "needs_review"
            ),
            "background_removal_quality_reasons": ",".join(
                reason for reason in reasons if isinstance(reason, str)
            ),
        },
    }
