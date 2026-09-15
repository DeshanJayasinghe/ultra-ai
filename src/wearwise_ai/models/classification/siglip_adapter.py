from __future__ import annotations

from typing import Protocol

from wearwise_ai.application.clothing_classification_service import (
    SIGLIP_CLASSIFICATION_MODEL_FAMILY,
    SIGLIP_CLASSIFICATION_MODEL_NAME,
    SIGLIP_CLASSIFICATION_MODEL_VERSION,
    ClothingClassificationRequest,
    RawCategoryPrediction,
)
from wearwise_ai.models.background_removal.birefnet_adapter import ModelUnavailableError


class SigLIPCategoryRunner(Protocol):
    def classify_category(
        self, request: ClothingClassificationRequest
    ) -> RawCategoryPrediction: ...


class SigLIPCategoryClassifier:
    """CategoryClassifier adapter for SigLIP zero-shot or embedding similarity classifiers."""

    def __init__(self, *, runner: SigLIPCategoryRunner | None = None) -> None:
        self._runner = runner

    def classify(self, request: ClothingClassificationRequest) -> RawCategoryPrediction:
        if self._runner is None:
            raise ModelUnavailableError("SigLIP category classifier runner is not configured.")

        prediction = self._runner.classify_category(request)
        return RawCategoryPrediction(
            category_code=prediction.category_code,
            confidence=prediction.confidence,
            model_family=prediction.model_family or SIGLIP_CLASSIFICATION_MODEL_FAMILY,
            model_name=prediction.model_name or SIGLIP_CLASSIFICATION_MODEL_NAME,
            model_version=prediction.model_version or SIGLIP_CLASSIFICATION_MODEL_VERSION,
            alternatives=prediction.alternatives,
        )
