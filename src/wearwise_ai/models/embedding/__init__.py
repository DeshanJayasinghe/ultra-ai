"""Embedding model adapters."""

from wearwise_ai.models.embedding.siglip_adapter import (
    DEFAULT_SIGLIP_MODULES,
    SigLIPConfig,
    SigLIPDependencyStatus,
    SigLIPEmbeddingGenerator,
    SigLIPRunner,
    check_siglip_dependencies,
)
from wearwise_ai.models.embedding.huggingface_runner import (
    HuggingFaceSigLIPRunnerConfig,
    TransformersSigLIPImageEmbedder,
)

__all__ = [
    "DEFAULT_SIGLIP_MODULES",
    "SigLIPConfig",
    "SigLIPDependencyStatus",
    "SigLIPEmbeddingGenerator",
    "SigLIPRunner",
    "HuggingFaceSigLIPRunnerConfig",
    "TransformersSigLIPImageEmbedder",
    "check_siglip_dependencies",
]
