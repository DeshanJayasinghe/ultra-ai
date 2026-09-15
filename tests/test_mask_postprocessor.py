from __future__ import annotations

from io import BytesIO
import unittest

from wearwise_ai.application.background_removal_service import BackgroundRemovalRequest
from wearwise_ai.models.background_removal import PillowAlphaMaskPostprocessor
from wearwise_ai.preprocessing import ImageProbe, plan_normalization


class MaskPostprocessorTests(unittest.TestCase):
    def test_build_cutout_adds_alpha_metrics(self) -> None:
        source_bytes = _source_image_bytes()
        mask_bytes = _mask_image_bytes()
        output = PillowAlphaMaskPostprocessor().build_cutout(
            request=BackgroundRemovalRequest(
                asset_id="asset-1",
                image_bytes=source_bytes,
                normalization_plan=plan_normalization(
                    ImageProbe(width=4, height=4, format_name="png", mime_type="image/png")
                ),
            ),
            mask_bytes=mask_bytes,
            confidence=0.8,
        )

        self.assertEqual(output.content_type, "image/webp")
        self.assertEqual(output.metrics["foreground_coverage_ratio"], 0.5)
        self.assertEqual(output.metrics["transparent_coverage_ratio"], 0.5)
        self.assertEqual(output.metrics["alpha_min"], 0)
        self.assertEqual(output.metrics["alpha_max"], 255)


def _source_image_bytes() -> bytes:
    from PIL import Image

    image = Image.new("RGB", (4, 4), "#336699")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _mask_image_bytes() -> bytes:
    from PIL import Image

    image = Image.new("L", (4, 4), 0)
    for x in range(2):
        for y in range(4):
            image.putpixel((x, y), 255)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
