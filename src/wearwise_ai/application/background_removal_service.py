from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol

from wearwise_ai.application.background_removal_quality import (
    BackgroundRemovalQualityAssessor,
)
from wearwise_ai.preprocessing import NormalizationPlan


BACKGROUND_REMOVAL_VERSION = 4
BACKGROUND_REMOVAL_PROCESSING_VERSION = f"background-removal-v{BACKGROUND_REMOVAL_VERSION}"


@dataclass(frozen=True, slots=True)
class BackgroundRemovalModelInfo:
    family: str
    name: str
    version: str


@dataclass(frozen=True, slots=True)
class BackgroundRemovalArtifact:
    variant: str
    storage_key: str
    content_type: str
    byte_size: int
    checksum_sha256: str
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True, slots=True)
class BackgroundRemovalOutput:
    image_bytes: bytes
    content_type: str
    confidence: float
    width: int | None = None
    height: int | None = None
    metrics: dict[str, int | float | str] | None = None


@dataclass(frozen=True, slots=True)
class BackgroundRemovalRequest:
    asset_id: str
    image_bytes: bytes
    normalization_plan: NormalizationPlan


@dataclass(frozen=True, slots=True)
class BackgroundRemovalResult:
    asset_id: str
    status: str
    model: BackgroundRemovalModelInfo
    artifact: BackgroundRemovalArtifact
    confidence: float
    quality_status: str
    quality_reasons: tuple[str, ...]
    processing_version: str = BACKGROUND_REMOVAL_PROCESSING_VERSION
    metrics: dict[str, int | float | str] | None = None

    def to_complete_payload(self, *, worker_id: str) -> dict[str, object]:
        review_required = self.quality_status == "needs_review"
        return {
            "worker_id": worker_id,
            "model_family": self.model.family,
            "model_name": self.model.name,
            "model_version": self.model.version,
            "processing_version": self.processing_version,
            "status_message": (
                "background removal completed; review recommended"
                if review_required
                else "background removal completed"
            ),
            "result": {
                "background_removed": {
                    "variant": self.artifact.variant,
                    "storage_key": self.artifact.storage_key,
                    "content_type": self.artifact.content_type,
                    "byte_size": self.artifact.byte_size,
                    "checksum_sha256": self.artifact.checksum_sha256,
                    "width": self.artifact.width,
                    "height": self.artifact.height,
                    "quality_status": self.quality_status,
                    "quality_reasons": list(self.quality_reasons),
                    "review_required": review_required,
                }
            },
            "confidence": {
                "background_removal": self.confidence,
            },
            "resource_metrics": self.metrics or {},
        }


class BackgroundRemover(Protocol):
    @property
    def model_info(self) -> BackgroundRemovalModelInfo: ...

    def remove_background(self, request: BackgroundRemovalRequest) -> BackgroundRemovalOutput: ...


class BackgroundRemovalArtifactStore(Protocol):
    def write_bytes(self, key: str, payload: bytes, content_type: str) -> None: ...


class BackgroundRemovalService:
    def __init__(
        self,
        *,
        remover: BackgroundRemover,
        artifact_store: BackgroundRemovalArtifactStore,
        quality_assessor: BackgroundRemovalQualityAssessor | None = None,
        output_prefix: str = "ai-outputs",
    ) -> None:
        self._remover = remover
        self._artifact_store = artifact_store
        self._quality_assessor = quality_assessor or BackgroundRemovalQualityAssessor()
        self._output_prefix = output_prefix.strip("/")

    def process(self, request: BackgroundRemovalRequest) -> BackgroundRemovalResult:
        output = self._remover.remove_background(request)
        variant = f"background_removed_v{BACKGROUND_REMOVAL_VERSION}"
        storage_key = f"{self._output_prefix}/{request.asset_id}/{variant}.webp"

        self._artifact_store.write_bytes(storage_key, output.image_bytes, output.content_type)
        quality = self._quality_assessor.assess(
            confidence=output.confidence,
            metrics=output.metrics,
        )
        metrics = {
            **(output.metrics or {}),
            "background_removal_quality_status": quality.status,
            "background_removal_quality_reasons": ",".join(quality.reasons),
        }

        artifact = BackgroundRemovalArtifact(
            variant=variant,
            storage_key=storage_key,
            content_type=output.content_type,
            byte_size=len(output.image_bytes),
            checksum_sha256=hashlib.sha256(output.image_bytes).hexdigest(),
            width=output.width,
            height=output.height,
        )

        return BackgroundRemovalResult(
            asset_id=request.asset_id,
            status="completed",
            model=self._remover.model_info,
            artifact=artifact,
            confidence=output.confidence,
            quality_status=quality.status,
            quality_reasons=quality.reasons,
            metrics=metrics,
        )
