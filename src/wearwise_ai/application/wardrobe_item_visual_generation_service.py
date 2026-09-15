from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol


WARDROBE_ITEM_VISUAL_MODEL_FAMILY = "openai"
WARDROBE_ITEM_VISUAL_MODEL_NAME = "gpt-image-1-mini"
WARDROBE_ITEM_VISUAL_MODEL_VERSION = "images-v1"
WARDROBE_ITEM_VISUAL_PROCESSING_VERSION = "wardrobe-item-visual-generation-v1"
WARDROBE_ITEM_VISUAL_VARIANT = "presentation_v1"


@dataclass(frozen=True, slots=True)
class WardrobeItemVisualGenerationRequest:
    job_id: str
    media_asset_id: str
    name: str | None
    category_code: str | None
    subcategory_code: str | None
    image_bytes: bytes
    content_type: str


@dataclass(frozen=True, slots=True)
class WardrobeItemVisualOutput:
    image_bytes: bytes
    content_type: str
    width: int
    height: int
    model_family: str
    model_name: str
    model_version: str
    prompt_version: str


@dataclass(frozen=True, slots=True)
class WardrobeItemVisualGenerationResult:
    storage_key: str
    output: WardrobeItemVisualOutput

    def to_complete_payload(self, *, worker_id: str) -> dict[str, object]:
        return {
            "worker_id": worker_id,
            "model_family": self.output.model_family,
            "model_name": self.output.model_name,
            "model_version": self.output.model_version,
            "processing_version": WARDROBE_ITEM_VISUAL_PROCESSING_VERSION,
            "status_message": "wardrobe presentation image generated",
            "result": {
                "presentation_image": {
                    "variant": WARDROBE_ITEM_VISUAL_VARIANT,
                    "storage_key": self.storage_key,
                    "content_type": self.output.content_type,
                    "byte_size": len(self.output.image_bytes),
                    "checksum_sha256": hashlib.sha256(self.output.image_bytes).hexdigest(),
                    "width": self.output.width,
                    "height": self.output.height,
                    "prompt_version": self.output.prompt_version,
                }
            },
            "confidence": {
                "visual_quality": 0.8,
            },
        }


class WardrobeItemVisualGenerator(Protocol):
    def generate(self, request: WardrobeItemVisualGenerationRequest) -> WardrobeItemVisualOutput: ...


class WardrobeItemVisualGenerationService:
    def __init__(self, *, generator: WardrobeItemVisualGenerator, artifact_store) -> None:
        self._generator = generator
        self._artifact_store = artifact_store

    def generate(
        self, request: WardrobeItemVisualGenerationRequest
    ) -> WardrobeItemVisualGenerationResult:
        output = self._generator.generate(request)
        storage_key = (
            f"ai-outputs/{request.media_asset_id}/{WARDROBE_ITEM_VISUAL_VARIANT}.webp"
        )
        self._artifact_store.write_bytes(storage_key, output.image_bytes, output.content_type)

        return WardrobeItemVisualGenerationResult(storage_key=storage_key, output=output)
