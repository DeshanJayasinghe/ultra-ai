from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Protocol

from wearwise_ai.application.embedding_generation_service import (
    EMBEDDING_MODEL_FAMILY,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_VERSION,
    EmbeddingGenerationRequest,
    RawEmbedding,
)
from wearwise_ai.models.background_removal.birefnet_adapter import ModelUnavailableError


DEFAULT_SIGLIP_MODULES = (
    "PIL",
    "numpy",
    "torch",
    "transformers",
)


@dataclass(frozen=True, slots=True)
class SigLIPConfig:
    model_name: str = EMBEDDING_MODEL_NAME
    model_version: str = EMBEDDING_MODEL_VERSION
    model_family: str = EMBEDDING_MODEL_FAMILY
    required_modules: tuple[str, ...] = DEFAULT_SIGLIP_MODULES


@dataclass(frozen=True, slots=True)
class SigLIPDependencyStatus:
    required_modules: tuple[str, ...]
    missing_modules: tuple[str, ...]

    @property
    def is_ready(self) -> bool:
        return len(self.missing_modules) == 0


class SigLIPRunner(Protocol):
    def embed_image(
        self,
        request: EmbeddingGenerationRequest,
        config: SigLIPConfig,
    ) -> tuple[float, ...]: ...


def check_siglip_dependencies(
    required_modules: tuple[str, ...] = DEFAULT_SIGLIP_MODULES,
) -> SigLIPDependencyStatus:
    missing = tuple(module_name for module_name in required_modules if find_spec(module_name) is None)
    return SigLIPDependencyStatus(
        required_modules=required_modules,
        missing_modules=missing,
    )


class SigLIPEmbeddingGenerator:
    """ImageEmbedder adapter for SigLIP-style garment embeddings."""

    def __init__(
        self,
        *,
        config: SigLIPConfig | None = None,
        runner: SigLIPRunner | None = None,
    ) -> None:
        self._config = config or SigLIPConfig()
        self._runner = runner

    def embed_image(self, request: EmbeddingGenerationRequest) -> RawEmbedding:
        if self._runner is None:
            status = check_siglip_dependencies(self._config.required_modules)
            if not status.is_ready:
                missing = ", ".join(status.missing_modules)
                raise ModelUnavailableError(
                    "SigLIP embedding dependencies are not installed: "
                    f"{missing}. Install the ML runtime extra and configure a model runner."
                )

            raise ModelUnavailableError(
                "SigLIP dependencies are installed, but no model runner is configured."
            )

        return RawEmbedding(
            values=self._runner.embed_image(request, self._config),
            model_family=self._config.model_family,
            model_name=self._config.model_name,
            model_version=self._config.model_version,
        )
