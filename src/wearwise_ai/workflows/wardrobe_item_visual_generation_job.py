from __future__ import annotations

from dataclasses import dataclass

from wearwise_ai.application.wardrobe_item_visual_generation_service import (
    WardrobeItemVisualGenerationRequest,
    WardrobeItemVisualGenerationResult,
    WardrobeItemVisualGenerationService,
)
from wearwise_ai.infrastructure.internal_api_client import LaravelInternalApiClient
from wearwise_ai.storage import ArtifactReader
from wearwise_ai.workflows.artifact_validation import validate_cleaned_artifact_or_fail
from wearwise_ai.workflows.preprocess_job import BinaryFetcher, UrlBinaryFetcher


@dataclass(frozen=True, slots=True)
class WardrobeItemVisualGenerationJobOutcome:
    request_id: str
    job_id: str
    media_asset_id: str
    result: WardrobeItemVisualGenerationResult | None
    status: str


class WardrobeItemVisualGenerationJobWorkflow:
    def __init__(
        self,
        *,
        internal_api: LaravelInternalApiClient,
        generation_service: WardrobeItemVisualGenerationService,
        binary_fetcher: BinaryFetcher | None = None,
        artifact_reader: ArtifactReader | None = None,
    ) -> None:
        self._internal_api = internal_api
        self._generation_service = generation_service
        self._binary_fetcher = binary_fetcher or UrlBinaryFetcher()
        self._artifact_reader = artifact_reader or _default_artifact_reader()

    def run(
        self,
        *,
        job_id: str,
        worker_id: str,
        request_id: str,
    ) -> WardrobeItemVisualGenerationJobOutcome:
        job_response = self._internal_api.get_ai_job(job_id, request_id=request_id)
        job_data = job_response["data"]
        media_asset_id = str(job_data["media_asset_id"])
        options = job_data.get("options") if isinstance(job_data.get("options"), dict) else {}
        background_removed = _background_removed_options(options)

        self._internal_api.report_progress(
            job_id,
            request_id=request_id,
            worker_id=worker_id,
            progress_percent=20,
            status_message="fetching cleaned garment image",
        )

        artifact_location = _artifact_location(background_removed)
        if artifact_location is None:
            access_response = self._internal_api.get_media_asset_access(
                media_asset_id,
                request_id=request_id,
                variant="background_removed_v4",
            )
            access = access_response["data"]["access"]
            media_asset = access_response["data"]["media_asset"]
            image_bytes = self._binary_fetcher.fetch(str(access["url"]))
            content_type = _access_content_type(
                access=access,
                media_asset=media_asset,
                fallback="image/png",
            )
        else:
            image_bytes = self._read_artifact(artifact_location)
            content_type = str(background_removed.get("content_type") or "image/webp")

        self._internal_api.report_progress(
            job_id,
            request_id=request_id,
            worker_id=worker_id,
            progress_percent=45,
            status_message="validating cleaned garment image",
        )
        if not validate_cleaned_artifact_or_fail(
            internal_api=self._internal_api,
            job_id=job_id,
            worker_id=worker_id,
            request_id=request_id,
            image_bytes=image_bytes,
            content_type=content_type,
            error_code="WARDROBE_PRESENTATION_INPUT_INVALID",
            status_message="wardrobe presentation input invalid",
        ):
            return WardrobeItemVisualGenerationJobOutcome(
                request_id=request_id,
                job_id=job_id,
                media_asset_id=media_asset_id,
                result=None,
                status="failed",
            )

        self._internal_api.report_progress(
            job_id,
            request_id=request_id,
            worker_id=worker_id,
            progress_percent=70,
            status_message="generating wardrobe presentation image",
        )
        result = self._generation_service.generate(
            WardrobeItemVisualGenerationRequest(
                job_id=job_id,
                media_asset_id=media_asset_id,
                name=_optional_string(options.get("name")),
                category_code=_optional_string(options.get("category_code")),
                subcategory_code=_optional_string(options.get("subcategory_code")),
                image_bytes=image_bytes,
                content_type=content_type,
            )
        )
        self._internal_api.complete_job(
            job_id,
            request_id=request_id,
            payload=result.to_complete_payload(worker_id=worker_id),
        )

        return WardrobeItemVisualGenerationJobOutcome(
            request_id=request_id,
            job_id=job_id,
            media_asset_id=media_asset_id,
            result=result,
            status="completed",
        )

    def _read_artifact(self, location: ArtifactLocation) -> bytes:
        if location.kind == "url":
            return self._binary_fetcher.fetch(location.value)

        return self._artifact_reader.read_bytes(location.value)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value != "" else None


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


def _access_content_type(
    *,
    access: object,
    media_asset: object,
    fallback: str,
) -> str:
    if not isinstance(media_asset, dict):
        return fallback

    variant_name = access.get("variant") if isinstance(access, dict) else None
    variants = media_asset.get("variants")
    if isinstance(variant_name, str) and isinstance(variants, list):
        for variant in variants:
            if not isinstance(variant, dict):
                continue

            if variant.get("variant") == variant_name:
                content_type = variant.get("content_type")
                if isinstance(content_type, str) and content_type != "":
                    return content_type

    content_type = media_asset.get("content_type")
    return content_type if isinstance(content_type, str) and content_type != "" else fallback


def _default_artifact_reader() -> ArtifactReader:
    from pathlib import Path

    from wearwise_ai.core.config import get_settings
    from wearwise_ai.storage import LocalArtifactStore

    return LocalArtifactStore(root=Path(get_settings().artifact_root))
