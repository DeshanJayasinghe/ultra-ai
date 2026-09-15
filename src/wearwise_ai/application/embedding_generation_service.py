from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from time import perf_counter
from typing import Protocol


EMBEDDING_GENERATION_VERSION = 1
EMBEDDING_GENERATION_PROCESSING_VERSION = f"embedding-generation-v{EMBEDDING_GENERATION_VERSION}"
EMBEDDING_MODEL_FAMILY = "siglip"
EMBEDDING_MODEL_NAME = "google/siglip-base-patch16-224"
EMBEDDING_MODEL_VERSION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
EMBEDDING_NORMALIZATION = "l2"


class ImageEmbedder(Protocol):
    def embed_image(self, request: "EmbeddingGenerationRequest") -> "RawEmbedding": ...


@dataclass(frozen=True, slots=True)
class EmbeddingGenerationRequest:
    asset_id: str
    image_bytes: bytes
    content_type: str


@dataclass(frozen=True, slots=True)
class RawEmbedding:
    values: tuple[float, ...]
    model_family: str
    model_name: str
    model_version: str


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    values: tuple[float, ...]
    dimensions: int
    norm: float
    normalization: str


@dataclass(frozen=True, slots=True)
class EmbeddingGenerationResult:
    asset_id: str
    status: str
    embedding: EmbeddingVector | None
    model_family: str
    model_name: str
    model_version: str
    processing_version: str
    metrics: dict[str, int | float | str]

    def to_ai_result(self) -> dict[str, object]:
        if self.embedding is None:
            return {
                "embedding": None,
                "embedding_metadata": {
                    "model_family": self.model_family,
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "dimensions": 0,
                    "normalization": EMBEDDING_NORMALIZATION,
                },
            }

        return {
            "embedding": {
                "values": list(self.embedding.values),
                "dimensions": self.embedding.dimensions,
                "normalization": self.embedding.normalization,
            },
            "embedding_metadata": {
                "model_family": self.model_family,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "dimensions": self.embedding.dimensions,
                "normalization": self.embedding.normalization,
            },
        }

    def to_complete_payload(self, *, worker_id: str) -> dict[str, object]:
        confidence = None
        if self.embedding is not None:
            confidence = {
                "embedding_generation": 1.0,
            }

        return {
            "worker_id": worker_id,
            "model_family": self.model_family,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "processing_version": self.processing_version,
            "status_message": (
                "embedding generation completed"
                if self.status == "ready"
                else "embedding generation needs review"
            ),
            "result": {
                **self.to_ai_result(),
                "review_required": self.status != "ready",
            },
            "confidence": confidence,
            "resource_metrics": self.metrics,
        }


class EmbeddingGenerationService:
    def __init__(self, *, embedder: ImageEmbedder) -> None:
        self._embedder = embedder

    def generate(self, request: EmbeddingGenerationRequest) -> EmbeddingGenerationResult:
        started_at = perf_counter()
        raw = self._embedder.embed_image(request)
        normalized = normalize_embedding(raw.values)

        return EmbeddingGenerationResult(
            asset_id=request.asset_id,
            status="ready",
            embedding=normalized,
            model_family=raw.model_family,
            model_name=raw.model_name,
            model_version=raw.model_version,
            processing_version=EMBEDDING_GENERATION_PROCESSING_VERSION,
            metrics={
                "embedding_dimensions": normalized.dimensions,
                "embedding_norm_before_normalization": round(normalized.norm, 8),
                "embedding_generation_duration_ms": _duration_ms(started_at),
            },
        )


def normalize_embedding(values: tuple[float, ...]) -> EmbeddingVector:
    if len(values) == 0:
        raise ValueError("Embedding vector must not be empty.")

    norm = sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ValueError("Embedding vector norm must be greater than zero.")

    return EmbeddingVector(
        values=tuple(round(value / norm, 8) for value in values),
        dimensions=len(values),
        norm=norm,
        normalization=EMBEDDING_NORMALIZATION,
    )


def _duration_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)
