from __future__ import annotations

import json
import unittest

from wearwise_ai.application.clothing_classification_service import (
    ClothingClassificationRequest,
)
from wearwise_ai.models.classification import OpenAICategoryClassifier, OpenAIClassificationConfig


class FakeOpenAIResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.output_text = json.dumps(payload)


class FakeResponsesClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeOpenAIResponse:
        self.calls.append(kwargs)
        return FakeOpenAIResponse(self.payload)


class OpenAICategoryAdapterTests(unittest.TestCase):
    def test_classifies_with_structured_response_and_image_input(self) -> None:
        client = FakeResponsesClient(
            {
                "category": "jackets",
                "confidence": 0.93,
                "alternatives": [
                    {"category": "tops", "confidence": 0.18},
                    {"category": "unknown", "confidence": 0.04},
                ],
                "review_required": False,
                "review_reason": None,
                "visible_evidence": ["outerwear shape", "front opening"],
            }
        )
        classifier = OpenAICategoryClassifier(
            config=OpenAIClassificationConfig(
                model_name="gpt-test",
                model_version="test-version",
                image_detail="high",
            ),
            responses_client=client,
        )

        prediction = classifier.classify(_request())

        self.assertEqual(prediction.category_code, "jackets")
        self.assertEqual(prediction.confidence, 0.93)
        self.assertEqual(prediction.model_family, "openai")
        self.assertEqual(prediction.model_name, "gpt-test")
        self.assertEqual(prediction.model_version, "test-version")
        self.assertFalse(prediction.review_required)
        self.assertEqual(prediction.alternatives, (("tops", 0.18),))

        call = client.calls[0]
        self.assertEqual(call["model"], "gpt-test")
        self.assertEqual(call["reasoning"], {"effort": "minimal"})
        self.assertEqual(call["max_output_tokens"], 600)
        self.assertEqual(call["text"]["verbosity"], "low")
        self.assertEqual(call["text"]["format"]["type"], "json_schema")
        image_block = call["input"][1]["content"][1]
        self.assertEqual(image_block["type"], "input_image")
        self.assertEqual(image_block["detail"], "high")
        self.assertTrue(image_block["image_url"].startswith("data:image/png;base64,"))

    def test_unknown_category_requests_review(self) -> None:
        classifier = OpenAICategoryClassifier(
            responses_client=FakeResponsesClient(
                {
                    "category": "unknown",
                    "confidence": 0.71,
                    "alternatives": [],
                    "review_required": False,
                    "review_reason": "item is unclear",
                    "visible_evidence": [],
                }
            )
        )

        prediction = classifier.classify(_request())

        self.assertEqual(prediction.category_code, "unknown")
        self.assertTrue(prediction.review_required)
        self.assertEqual(prediction.review_reason, "item is unclear")


def _request() -> ClothingClassificationRequest:
    return ClothingClassificationRequest(
        asset_id="asset-1",
        image_bytes=b"fake-image",
        content_type="image/png",
    )


if __name__ == "__main__":
    unittest.main()
