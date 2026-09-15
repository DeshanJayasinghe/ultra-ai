from __future__ import annotations

import json
import unittest

from wearwise_ai.application.attribute_extraction_service import AttributeExtractionRequest
from wearwise_ai.application.colour_extraction_service import ColourExtractionRequest
from wearwise_ai.models.openai_vision import (
    OpenAIAttributeExtractor,
    OpenAIColourExtractor,
    OpenAIVisionExtractionConfig,
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


class OpenAIVisionAdapterTests(unittest.TestCase):
    def test_colour_extractor_maps_openai_response(self) -> None:
        client = FakeResponsesClient(
            {
                "primary_colour": {
                    "value": "white",
                    "hex": "#f4f1ec",
                    "confidence": 0.91,
                    "coverage": 0.76,
                    "source_rgb": [244, 241, 236],
                },
                "secondary_colour": {
                    "value": "black",
                    "hex": "#181818",
                    "confidence": 0.64,
                    "coverage": 0.18,
                    "source_rgb": [24, 24, 24],
                },
                "review_required": False,
                "review_reason": None,
            }
        )
        extractor = OpenAIColourExtractor(
            config=OpenAIVisionExtractionConfig(model_name="gpt-test"),
            responses_client=client,
        )

        result = extractor.extract(
            ColourExtractionRequest(
                asset_id="asset-1",
                image_bytes=b"image",
                content_type="image/png",
            )
        )

        self.assertEqual(result.model_family, "openai")
        self.assertEqual(result.model_name, "gpt-test")
        self.assertEqual(result.primary_colour.code, "white")
        self.assertEqual(result.primary_colour.source_rgb, (244, 241, 236))
        self.assertEqual(result.secondary_colour.code, "black")
        self.assertFalse(result.review_required)
        self.assertEqual(client.calls[0]["model"], "gpt-test")
        self.assertEqual(client.calls[0]["reasoning"], {"effort": "minimal"})
        self.assertEqual(client.calls[0]["max_output_tokens"], 600)
        self.assertEqual(client.calls[0]["text"]["verbosity"], "low")
        image_block = client.calls[0]["input"][1]["content"][1]
        self.assertEqual(image_block["detail"], "low")

    def test_attribute_extractor_maps_openai_response(self) -> None:
        client = FakeResponsesClient(
            {
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
        )
        extractor = OpenAIAttributeExtractor(
            config=OpenAIVisionExtractionConfig(model_name="gpt-test"),
            responses_client=client,
        )

        result = extractor.extract(
            AttributeExtractionRequest(
                asset_id="asset-1",
                image_bytes=b"image",
                content_type="image/png",
                category_code="tops",
            )
        )

        self.assertEqual(result.model_family, "openai")
        self.assertEqual(result.subcategory.value, "t-shirt")
        self.assertEqual(result.brand.value, "Nike")
        self.assertEqual(result.material.value, "cotton")
        self.assertEqual(result.seasons[0].value, "all-season")
        self.assertFalse(result.review_required)
        self.assertEqual(client.calls[0]["model"], "gpt-test")
        self.assertEqual(client.calls[0]["reasoning"], {"effort": "minimal"})
        self.assertEqual(client.calls[0]["max_output_tokens"], 600)
        self.assertEqual(client.calls[0]["text"]["verbosity"], "low")
        developer_text = client.calls[0]["input"][0]["content"][0]["text"]
        user_text = client.calls[0]["input"][1]["content"][0]["text"]
        self.assertNotIn("tops", developer_text)
        self.assertIn("Current category hint: tops.", user_text)


if __name__ == "__main__":
    unittest.main()
