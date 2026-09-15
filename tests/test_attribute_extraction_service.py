from __future__ import annotations

from io import BytesIO
import unittest

from wearwise_ai.application.attribute_extraction_service import (
    AttributeExtractionRequest,
    AttributeExtractionRule,
    AttributeExtractionService,
)


class AttributeExtractionServiceTests(unittest.TestCase):
    def test_extracts_visible_attributes_for_top_cutout(self) -> None:
        image_bytes = _rgba_cutout(width=20, height=40, body=(7, 6, 12, 34), sleeves=True)
        service = AttributeExtractionService(
            rule=AttributeExtractionRule(min_visible_pixels=10, min_confidence=0.5)
        )

        result = service.extract(
            AttributeExtractionRequest(
                asset_id="asset-1",
                image_bytes=image_bytes,
                content_type="image/png",
                category_code="tops",
            )
        )

        self.assertEqual(result.status, "ready")
        payload = result.to_complete_payload(worker_id="worker-a")
        self.assertFalse(payload["result"]["review_required"])
        self.assertNotIn("attributes", payload["result"])
        self.assertIsNone(payload["result"]["brand"])
        self.assertEqual(payload["result"]["subcategory"]["value"], "shirt")
        self.assertEqual(payload["result"]["material"]["value"], "cotton")
        self.assertIn(payload["result"]["pattern"]["value"], {"solid", "graphic"})
        self.assertEqual(payload["result"]["seasons"][0]["value"], "all-season")
        self.assertEqual(payload["result"]["occasions"][0]["value"], "daily")
        self.assertEqual(payload["result"]["formality"]["value"], "casual")
        self.assertEqual(payload["result"]["style_tags"][0]["value"], "casual")

    def test_skips_unsupported_category_attributes_for_review(self) -> None:
        image_bytes = _rgba_cutout(width=20, height=20, body=(6, 6, 14, 14), sleeves=False)

        result = AttributeExtractionService(rule=AttributeExtractionRule(min_visible_pixels=10)).extract(
            AttributeExtractionRequest(
                asset_id="asset-2",
                image_bytes=image_bytes,
                content_type="image/png",
                category_code="shoes",
            )
        )

        self.assertEqual(result.status, "needs_review")
        self.assertEqual(result.subcategory.value, "trainers")
        self.assertIsNone(result.brand)

    def test_marks_low_visible_pixel_cutout_for_review(self) -> None:
        image_bytes = _rgba_cutout(width=4, height=4, body=(1, 1, 1, 1), sleeves=False)

        result = AttributeExtractionService().extract(
            AttributeExtractionRequest(
                asset_id="asset-3",
                image_bytes=image_bytes,
                content_type="image/png",
                category_code="tops",
            )
        )

        self.assertEqual(result.status, "needs_review")
        self.assertEqual(
            result.metrics["attribute_extraction_status_reason"],
            "not_enough_visible_pixels",
        )


def _rgba_cutout(
    *,
    width: int,
    height: int,
    body: tuple[int, int, int, int],
    sleeves: bool,
) -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle(body, fill=(30, 90, 170, 255))
    if sleeves:
        draw.rectangle((1, body[1] + 2, body[0], body[3] - 8), fill=(30, 90, 170, 255))
        draw.rectangle((body[2], body[1] + 2, width - 2, body[3] - 8), fill=(30, 90, 170, 255))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
