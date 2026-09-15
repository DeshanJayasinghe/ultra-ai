from __future__ import annotations

import unittest

from wearwise_ai.application.clothing_classification_service import (
    ClothingClassificationRequest,
    ClothingClassificationRule,
    ClothingClassificationService,
    ConfidenceRoutedClassifier,
    RawCategoryPrediction,
)


class FakeClassifier:
    def __init__(self, prediction: RawCategoryPrediction) -> None:
        self.prediction = prediction

    def classify(self, request: ClothingClassificationRequest) -> RawCategoryPrediction:
        return self.prediction


class ClothingClassificationServiceTests(unittest.TestCase):
    def test_returns_category_suggestion_for_supported_taxonomy_code(self) -> None:
        service = ClothingClassificationService(
            classifier=FakeClassifier(
                RawCategoryPrediction(
                    category_code="tops",
                    confidence=0.91,
                    model_family="siglip",
                    model_name="test-siglip",
                    model_version="rev-1",
                    alternatives=(("jackets", 0.22),),
                )
            )
        )

        result = service.classify(_request())

        self.assertEqual(result.status, "ready")
        self.assertEqual(result.category.code, "tops")
        self.assertEqual(result.category.confidence, 0.91)
        payload = result.to_complete_payload(worker_id="worker-a")
        self.assertEqual(payload["status_message"], "clothing classification completed")
        self.assertEqual(payload["result"]["category"]["value"], "tops")
        self.assertEqual(payload["result"]["category"]["alternatives"][0]["value"], "jackets")
        self.assertFalse(payload["result"]["review_required"])

    def test_marks_low_confidence_category_for_review(self) -> None:
        service = ClothingClassificationService(
            classifier=FakeClassifier(
                RawCategoryPrediction(
                    category_code="tops",
                    confidence=0.42,
                    model_family="siglip",
                    model_name="test-siglip",
                    model_version="rev-1",
                )
            ),
            rule=ClothingClassificationRule(
                min_category_confidence=0.55,
                min_category_margin=0.15,
            ),
        )

        result = service.classify(_request())

        self.assertEqual(result.status, "needs_review")
        payload = result.to_complete_payload(worker_id="worker-a")
        self.assertEqual(payload["status_message"], "clothing classification needs review")
        self.assertTrue(payload["result"]["review_required"])
        self.assertEqual(payload["result"]["category"]["value"], "tops")
        self.assertEqual(
            result.metrics["classification_status_reason"],
            "category_confidence_below_threshold",
        )

    def test_marks_close_alternative_for_review(self) -> None:
        service = ClothingClassificationService(
            classifier=FakeClassifier(
                RawCategoryPrediction(
                    category_code="tops",
                    confidence=0.88,
                    model_family="openai",
                    model_name="gpt-5",
                    model_version="responses-v1",
                    alternatives=(("jackets", 0.81),),
                )
            )
        )

        result = service.classify(_request())

        self.assertEqual(result.status, "needs_review")
        self.assertEqual(
            result.metrics["classification_status_reason"],
            "category_margin_below_threshold",
        )

    def test_rejects_category_outside_taxonomy(self) -> None:
        service = ClothingClassificationService(
            classifier=FakeClassifier(
                RawCategoryPrediction(
                    category_code="unknown",
                    confidence=0.95,
                    model_family="openai",
                    model_name="gpt-5",
                    model_version="responses-v1",
                )
            )
        )

        result = service.classify(_request())

        self.assertEqual(result.status, "needs_review")
        self.assertIsNone(result.category)
        self.assertEqual(result.metrics["classification_status_reason"], "category_not_in_taxonomy")

    def test_confidence_routed_classifier_uses_primary_when_confident(self) -> None:
        primary = CountingClassifier(
            RawCategoryPrediction(
                category_code="tops",
                confidence=0.91,
                model_family="siglip",
                model_name="test-siglip",
                model_version="rev-1",
                alternatives=(("jackets", 0.2),),
            )
        )
        fallback = CountingClassifier(
            RawCategoryPrediction(
                category_code="jackets",
                confidence=0.93,
                model_family="openai",
                model_name="gpt-test",
                model_version="responses-v1",
            )
        )

        prediction = ConfidenceRoutedClassifier(primary=primary, fallback=fallback).classify(
            _request()
        )

        self.assertEqual(prediction.category_code, "tops")
        self.assertEqual(primary.call_count, 1)
        self.assertEqual(fallback.call_count, 0)

    def test_confidence_routed_classifier_falls_back_when_uncertain(self) -> None:
        primary = CountingClassifier(
            RawCategoryPrediction(
                category_code="tops",
                confidence=0.61,
                model_family="siglip",
                model_name="test-siglip",
                model_version="rev-1",
            )
        )
        fallback = CountingClassifier(
            RawCategoryPrediction(
                category_code="jackets",
                confidence=0.93,
                model_family="openai",
                model_name="gpt-test",
                model_version="responses-v1",
            )
        )

        prediction = ConfidenceRoutedClassifier(primary=primary, fallback=fallback).classify(
            _request()
        )

        self.assertEqual(prediction.category_code, "jackets")
        self.assertEqual(primary.call_count, 1)
        self.assertEqual(fallback.call_count, 1)


class CountingClassifier:
    def __init__(self, prediction: RawCategoryPrediction) -> None:
        self.prediction = prediction
        self.call_count = 0

    def classify(self, request: ClothingClassificationRequest) -> RawCategoryPrediction:
        self.call_count += 1
        return self.prediction


def _request() -> ClothingClassificationRequest:
    return ClothingClassificationRequest(
        asset_id="asset-1",
        image_bytes=b"image",
        content_type="image/webp",
    )


if __name__ == "__main__":
    unittest.main()
