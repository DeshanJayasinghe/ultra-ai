from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wearwise_ai.application.outfit_visual_generation_service import (
    OutfitVisualGenerationRequest,
    OutfitVisualGenerationResult,
    OutfitVisualGenerationService,
    OutfitVisualItem,
)
from wearwise_ai.infrastructure.internal_api_client import LaravelInternalApiClient
from wearwise_ai.workflows.preprocess_job import BinaryFetcher, UrlBinaryFetcher


@dataclass(frozen=True, slots=True)
class OutfitVisualGenerationJobOutcome:
    request_id: str
    job_id: str
    anchor_media_asset_id: str
    item_count: int
    result: OutfitVisualGenerationResult


class OutfitVisualGenerationJobWorkflow:
    def __init__(
        self,
        *,
        internal_api: LaravelInternalApiClient,
        generation_service: OutfitVisualGenerationService,
        binary_fetcher: BinaryFetcher | None = None,
    ) -> None:
        self._internal_api = internal_api
        self._generation_service = generation_service
        self._binary_fetcher = binary_fetcher or UrlBinaryFetcher()

    def run(
        self,
        *,
        job_id: str,
        worker_id: str,
        request_id: str,
    ) -> OutfitVisualGenerationJobOutcome:
        job_response = self._internal_api.get_ai_job(job_id, request_id=request_id)
        job_data = job_response["data"]
        anchor_media_asset_id = str(job_data["media_asset_id"])
        options = job_data.get("options") if isinstance(job_data.get("options"), dict) else {}
        item_specs = _item_specs(options)

        self._internal_api.report_progress(
            job_id,
            request_id=request_id,
            worker_id=worker_id,
            progress_percent=15,
            status_message="fetching outfit items",
        )

        items = tuple(self._fetch_item(spec, request_id=request_id) for spec in item_specs)

        self._internal_api.report_progress(
            job_id,
            request_id=request_id,
            worker_id=worker_id,
            progress_percent=60,
            status_message="generating outfit visual",
        )

        result = self._generation_service.generate(
            OutfitVisualGenerationRequest(
                job_id=job_id,
                anchor_media_asset_id=anchor_media_asset_id,
                title=_optional_string(options.get("title")),
                style=_optional_string(options.get("style")),
                items=items,
            )
        )
        payload = result.to_complete_payload(worker_id=worker_id)
        payload["resource_metrics"] = {"input_item_count": len(items)}
        self._internal_api.complete_job(job_id, request_id=request_id, payload=payload)

        return OutfitVisualGenerationJobOutcome(
            request_id=request_id,
            job_id=job_id,
            anchor_media_asset_id=anchor_media_asset_id,
            item_count=len(items),
            result=result,
        )

    def _fetch_item(self, spec: dict[str, object], *, request_id: str) -> OutfitVisualItem:
        media_asset_id = str(spec["media_asset_id"])
        access_response = self._internal_api.get_media_asset_access(
            media_asset_id,
            request_id=request_id,
            variant="presentation_v1",
        )
        access = access_response["data"]["access"]
        media_asset = access_response["data"]["media_asset"]

        return OutfitVisualItem(
            item_id=str(spec.get("item_id") or ""),
            media_asset_id=media_asset_id,
            category_code=str(spec.get("category_code") or "unknown"),
            name=str(spec.get("name") or "wardrobe item"),
            content_type=str(media_asset.get("content_type") or "image/png"),
            image_bytes=self._binary_fetcher.fetch(str(access["url"])),
        )


def _item_specs(options: dict[str, Any]) -> list[dict[str, object]]:
    value = options.get("items")
    if not isinstance(value, list) or value == []:
        raise ValueError("outfit visual generation requires options.items")

    specs = [item for item in value if isinstance(item, dict) and item.get("media_asset_id")]
    if specs == []:
        raise ValueError("outfit visual generation requires item media_asset_id values")

    return specs[:8]


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value != "" else None
