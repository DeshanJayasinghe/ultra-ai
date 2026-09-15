from __future__ import annotations

from io import BytesIO
import unittest

from wearwise_ai.application.colour_extraction_service import (
    ColourExtractionRequest,
    ColourExtractionRule,
    ColourExtractionService,
)


class ColourExtractionServiceTests(unittest.TestCase):
    def test_extracts_primary_and_secondary_from_visible_pixels(self) -> None:
        image_bytes = _rgba_image_bytes(
            [
                [(25, 70, 150, 255), (25, 70, 150, 255), (40, 118, 75, 255), (0, 0, 0, 0)],
                [(25, 70, 150, 255), (25, 70, 150, 255), (40, 118, 75, 255), (0, 0, 0, 0)],
                [(25, 70, 150, 255), (25, 70, 150, 255), (40, 118, 75, 255), (0, 0, 0, 0)],
                [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            ]
        )
        service = ColourExtractionService(
            rule=ColourExtractionRule(min_visible_pixels=4, min_secondary_coverage=0.2)
        )

        result = service.extract(
            ColourExtractionRequest(
                asset_id="asset-1",
                image_bytes=image_bytes,
                content_type="image/png",
            )
        )

        self.assertEqual(result.status, "ready")
        self.assertEqual(result.primary_colour.code, "blue")
        self.assertEqual(result.secondary_colour.code, "green")
        self.assertAlmostEqual(result.primary_colour.coverage, 0.666667)
        self.assertAlmostEqual(result.secondary_colour.coverage, 0.333333)
        self.assertEqual(result.metrics["visible_pixel_count"], 9)

        payload = result.to_ai_result()
        self.assertEqual(payload["primary_colour"]["value"], "blue")
        self.assertEqual(payload["secondary_colour"]["value"], "green")

    def test_ignores_transparent_background_pixels(self) -> None:
        image_bytes = _rgba_image_bytes(
            [
                [(255, 255, 255, 0), (255, 255, 255, 0)],
                [(17, 19, 22, 255), (17, 19, 22, 255)],
            ]
        )
        service = ColourExtractionService(rule=ColourExtractionRule(min_visible_pixels=2))

        result = service.extract(
            ColourExtractionRequest(
                asset_id="asset-2",
                image_bytes=image_bytes,
                content_type="image/png",
            )
        )

        self.assertEqual(result.primary_colour.code, "black")
        self.assertIsNone(result.secondary_colour)

    def test_marks_result_for_review_when_not_enough_visible_pixels(self) -> None:
        image_bytes = _rgba_image_bytes(
            [
                [(0, 0, 0, 0), (0, 0, 0, 0)],
                [(0, 0, 0, 0), (200, 60, 60, 255)],
            ]
        )

        result = ColourExtractionService().extract(
            ColourExtractionRequest(
                asset_id="asset-3",
                image_bytes=image_bytes,
                content_type="image/png",
            )
        )

        self.assertEqual(result.status, "needs_review")
        self.assertIsNone(result.primary_colour)
        self.assertEqual(
            result.metrics["colour_extraction_status_reason"],
            "not_enough_visible_pixels",
        )


def _rgba_image_bytes(rows: list[list[tuple[int, int, int, int]]]) -> bytes:
    from PIL import Image

    height = len(rows)
    width = len(rows[0])
    image = Image.new("RGBA", (width, height))
    image.putdata([pixel for row in rows for pixel in row])

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
