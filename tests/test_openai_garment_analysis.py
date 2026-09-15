from __future__ import annotations

import json
import unittest

from wearwise_ai.application.attribute_extraction_service import AttributeExtractionRequest
from wearwise_ai.application.clothing_classification_service import (
    ClothingClassificationRequest,
)
from wearwise_ai.models.openai_garment_analysis import (
    GarmentAnalysisAttributeExtractor,
    GarmentAnalysisCategoryClassifier,
    InMemoryGarmentAnalysisCache,
    OpenAIGarmentAnalyser,
    OpenAIGarmentAnalysisConfig,
)


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


class OpenAIGarmentAnalysisTests(unittest.TestCase):
    def test_projects_category_and_attributes_from_one_cached_call(self) -> None:
        client = FakeResponsesClient(_payload())
        analyser = OpenAIGarmentAnalyser(
            config=OpenAIGarmentAnalysisConfig(
                model_name="gpt-test",
                model_version="test-version",
            ),
            responses_client=client,
            cache=InMemoryGarmentAnalysisCache(),
        )

        classifier = GarmentAnalysisCategoryClassifier(analyser=analyser)
        extractor = GarmentAnalysisAttributeExtractor(analyser=analyser)

        category = classifier.classify(
            ClothingClassificationRequest(
                asset_id="asset-1",
                image_bytes=b"same-image",
                content_type="image/png",
            )
        )
        attributes = extractor.extract(
            AttributeExtractionRequest(
                asset_id="asset-1",
                image_bytes=b"same-image",
                content_type="image/png",
                category_code="tops",
            )
        )

        self.assertEqual(category.category_code, "tops")
        self.assertEqual(category.confidence, 0.94)
        self.assertEqual(attributes.subcategory.value, "t-shirt")
        self.assertEqual(attributes.brand.value, "Nike")
        self.assertEqual(client.calls[0]["reasoning"], {"effort": "minimal"})
        self.assertEqual(client.calls[0]["max_output_tokens"], 900)
        self.assertEqual(client.calls[0]["text"]["verbosity"], "low")
        self.assertEqual(len(client.calls), 1)


def _payload() -> dict[str, object]:
    return {
        "category": "tops",
        "confidence": 0.94,
        "alternatives": [{"category": "jackets", "confidence": 0.12}],
        "visible_evidence": ["shirt shape"],
        "subcategory": {"value": "t-shirt", "confidence": 0.9, "reason": "crew neck"},
        "brand": {"value": "Nike", "confidence": 0.82, "reason": "visible chest logo"},
        "material": {"value": "cotton", "confidence": 0.72, "reason": "matte knit"},
        "pattern": {"value": "solid", "confidence": 0.88, "reason": "single colour"},
        "seasons": [{"value": "all-season", "confidence": 0.7, "reason": "basic tee"}],
        "occasions": [{"value": "daily", "confidence": 0.86, "reason": "casual basic"}],
        "formality": {"value": "casual", "confidence": 0.9, "reason": "relaxed cut"},
        "style_tags": [{"value": "minimal", "confidence": 0.78, "reason": "plain item"}],
        "review_required": False,
        "review_reason": None,
    }


if __name__ == "__main__":
    unittest.main()
