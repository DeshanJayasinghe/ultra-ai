from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol


OUTFIT_VISUAL_MODEL_FAMILY = "openai"
OUTFIT_VISUAL_MODEL_NAME = "gpt-image-1-mini"
OUTFIT_VISUAL_MODEL_VERSION = "images-v1"
OUTFIT_VISUAL_PROCESSING_VERSION = "outfit-visual-generation-v1"
OUTFIT_VISUAL_VARIANT = "outfit_visual_v1"


@dataclass(frozen=True, slots=True)
class OutfitVisualItem:
    item_id: str
    media_asset_id: str
    category_code: str
    name: str
    content_type: str
    image_bytes: bytes


@dataclass(frozen=True, slots=True)
class OutfitVisualGenerationRequest:
    job_id: str
    anchor_media_asset_id: str
    title: str | None
    style: str | None
    items: tuple[OutfitVisualItem, ...]


@dataclass(frozen=True, slots=True)
class OutfitVisualOutput:
    image_bytes: bytes
    content_type: str
    width: int
    height: int
    model_family: str
    model_name: str
    model_version: str
    prompt_version: str


@dataclass(frozen=True, slots=True)
class OutfitVisualGenerationResult:
    storage_key: str
    output: OutfitVisualOutput

    def to_complete_payload(self, *, worker_id: str) -> dict[str, object]:
        return {
            "worker_id": worker_id,
            "model_family": self.output.model_family,
            "model_name": self.output.model_name,
            "model_version": self.output.model_version,
            "processing_version": OUTFIT_VISUAL_PROCESSING_VERSION,
            "status_message": "outfit visual generated",
            "result": {
                "outfit_visual": {
                    "variant": OUTFIT_VISUAL_VARIANT,
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


class OutfitVisualGenerator(Protocol):
    def generate(self, request: OutfitVisualGenerationRequest) -> OutfitVisualOutput: ...


class OutfitVisualGenerationService:
    def __init__(self, *, generator: OutfitVisualGenerator, artifact_store) -> None:
        self._generator = generator
        self._artifact_store = artifact_store

    def generate(self, request: OutfitVisualGenerationRequest) -> OutfitVisualGenerationResult:
        output = self._generator.generate(request)
        storage_key = f"ai-outputs/{request.anchor_media_asset_id}/{OUTFIT_VISUAL_VARIANT}.webp"
        self._artifact_store.write_bytes(storage_key, output.image_bytes, output.content_type)

        return OutfitVisualGenerationResult(storage_key=storage_key, output=output)
