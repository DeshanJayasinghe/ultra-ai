from __future__ import annotations

import unittest

from wearwise_ai.application.clothing_classification_service import (
    ClothingClassificationRequest,
    RawCategoryPrediction,
)
from wearwise_ai.models.background_removal.birefnet_adapter import ModelUnavailableError
from wearwise_ai.models.classification import SigLIPCategoryClassifier


class FakeCategoryRunner:
    def classify_category(self, request: ClothingClassificationRequest) -> RawCategoryPrediction:
        return RawCategoryPrediction(
            category_code="tops",
            confidence=0.88,
            model_family="siglip",
            model_name="test-siglip",
            model_version="rev-1",
        )


class SigLIPCategoryAdapterTests(unittest.TestCase):
    def test_raises_controlled_error_when_runner_is_missing(self) -> None:
        classifier = SigLIPCategoryClassifier()

        with self.assertRaisesRegex(ModelUnavailableError, "runner is not configured"):
            classifier.classify(_request())

    def test_delegates_to_configured_runner(self) -> None:
        classifier = SigLIPCategoryClassifier(runner=FakeCategoryRunner())

        prediction = classifier.classify(_request())

        self.assertEqual(prediction.category_code, "tops")
        self.assertEqual(prediction.confidence, 0.88)
        self.assertEqual(prediction.model_version, "rev-1")


def _request() -> ClothingClassificationRequest:
    return ClothingClassificationRequest(
        asset_id="asset-1",
        image_bytes=b"image",
        content_type="image/webp",
    )


if __name__ == "__main__":
    unittest.main()
