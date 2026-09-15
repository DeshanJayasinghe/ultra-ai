from __future__ import annotations

import base64
import unittest

from wearwise_ai.application.outfit_visual_generation_service import (
    OutfitVisualGenerationRequest,
    OutfitVisualItem,
)
from wearwise_ai.application.wardrobe_item_visual_generation_service import (
    WardrobeItemVisualGenerationRequest,
)
from wearwise_ai.models.openai_image import (
    OpenAIOutfitVisualConfig,
    OpenAIOutfitVisualGenerator,
    OpenAIWardrobeItemVisualConfig,
    OpenAIWardrobeItemVisualGenerator,
)


class FakeImageData:
    def __init__(self, payload: bytes) -> None:
        self.b64_json = base64.b64encode(payload).decode("ascii")


class FakeImageResponse:
    def __init__(self, payload: bytes) -> None:
        self.data = [FakeImageData(payload)]


class FakeImagesClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> FakeImageResponse:
        self.calls.append(kwargs)
        return FakeImageResponse(b"generated-webp")


class OpenAIImageAdapterTests(unittest.TestCase):
    def test_outfit_visual_generator_maps_image_response(self) -> None:
        client = FakeImagesClient()
        generator = OpenAIOutfitVisualGenerator(
            config=OpenAIOutfitVisualConfig(model_name="gpt-image-test", size="1024x1536"),
            images_client=client,
        )

        result = generator.generate(
            OutfitVisualGenerationRequest(
                job_id="job-1",
                anchor_media_asset_id="asset-1",
                title="Classic Chic",
                style="premium flat-lay",
                items=(
                    OutfitVisualItem(
                        item_id="item-1",
                        media_asset_id="asset-1",
                        category_code="tops",
                        name="red t shirt",
                        content_type="image/webp",
                        image_bytes=b"item-image",
                    ),
                ),
            )
        )

        self.assertEqual(result.image_bytes, b"generated-webp")
        self.assertEqual(result.content_type, "image/webp")
        self.assertEqual(result.width, 1024)
        self.assertEqual(result.height, 1536)
        self.assertEqual(result.model_name, "gpt-image-test")
        self.assertEqual(client.calls[0]["model"], "gpt-image-test")
        self.assertIn("red t shirt", client.calls[0]["prompt"])

    def test_wardrobe_item_visual_generator_maps_image_response(self) -> None:
        client = FakeImagesClient()
        generator = OpenAIWardrobeItemVisualGenerator(
            config=OpenAIWardrobeItemVisualConfig(model_name="gpt-image-test", size="1024x1024"),
            images_client=client,
        )

        result = generator.generate(
            WardrobeItemVisualGenerationRequest(
                job_id="job-1",
                media_asset_id="asset-1",
                name="light blue denim",
                category_code="bottoms",
                subcategory_code="jeans",
                content_type="image/webp",
                image_bytes=b"item-image",
            )
        )

        self.assertEqual(result.image_bytes, b"generated-webp")
        self.assertEqual(result.content_type, "image/webp")
        self.assertEqual(result.width, 1024)
        self.assertEqual(result.height, 1024)
        self.assertEqual(result.model_name, "gpt-image-test")
        self.assertEqual(client.calls[0]["model"], "gpt-image-test")
        self.assertIn("light blue denim", client.calls[0]["prompt"])
        self.assertIn("bottoms", client.calls[0]["prompt"])


if __name__ == "__main__":
    unittest.main()
