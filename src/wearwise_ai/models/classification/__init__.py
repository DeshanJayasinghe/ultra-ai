"""Clothing classification model adapters."""

from wearwise_ai.models.classification.huggingface_runner import (
    DEFAULT_SIGLIP_CLASSIFICATION_MODULES,
    TransformersSigLIPZeroShotCategoryRunner,
    ZeroShotPromptSet,
)
from wearwise_ai.models.classification.openai_adapter import (
    OPENAI_CLASSIFICATION_RESPONSE_SCHEMA,
    OpenAICategoryClassifier,
    OpenAIClassificationConfig,
)
from wearwise_ai.models.classification.siglip_adapter import (
    SigLIPCategoryClassifier,
    SigLIPCategoryRunner,
)

__all__ = [
    "DEFAULT_SIGLIP_CLASSIFICATION_MODULES",
    "OPENAI_CLASSIFICATION_RESPONSE_SCHEMA",
    "OpenAICategoryClassifier",
    "OpenAIClassificationConfig",
    "SigLIPCategoryClassifier",
    "SigLIPCategoryRunner",
    "TransformersSigLIPZeroShotCategoryRunner",
    "ZeroShotPromptSet",
]
