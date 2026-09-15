from __future__ import annotations

from io import BytesIO
import os
import unittest

from wearwise_ai.application.background_removal_service import BackgroundRemovalRequest
from wearwise_ai.models.background_removal import (
    HuggingFaceBiRefNetRunnerConfig,
    PillowAlphaMaskPostprocessor,
    TransformersBiRefNetPredictor,
)
from wearwise_ai.preprocessing import ImageProbe, plan_normalization


@unittest.skipUnless(
    os.getenv("WEARWISE_RUN_ML_TESTS") == "1",
    "set WEARWISE_RUN_ML_TESTS=1 to run real BiRefNet integration tests",
)
class BiRefNetMlIntegrationTests(unittest.TestCase):
    def test_real_birefnet_predictor_generates_cutout(self) -> None:
        image_bytes = _synthetic_garment_image_bytes()
        runner_config = HuggingFaceBiRefNetRunnerConfig(
            model_id=os.getenv("WEARWISE_BIREFNET_MODEL_ID", "ZhengPeng7/BiRefNet"),
            revision=os.getenv("WEARWISE_BIREFNET_MODEL_REVISION") or None,
            device=os.getenv("WEARWISE_BIREFNET_DEVICE", "cpu"),
            local_files_only=os.getenv("WEARWISE_BIREFNET_LOCAL_FILES_ONLY") == "1",
        )

        mask_bytes, confidence, metrics = TransformersBiRefNetPredictor().predict_mask(
            image_bytes=image_bytes,
            runner_config=runner_config,
        )
        cutout = PillowAlphaMaskPostprocessor().build_cutout(
            request=BackgroundRemovalRequest(
                asset_id="synthetic-top",
                image_bytes=image_bytes,
                normalization_plan=plan_normalization(
                    ImageProbe(width=384, height=512, format_name="png", mime_type="image/png")
                ),
            ),
            mask_bytes=mask_bytes,
            confidence=confidence,
            metrics=metrics,
        )

        self.assertGreater(len(mask_bytes), 0)
        self.assertGreater(len(cutout.image_bytes), 0)
        self.assertEqual(cutout.content_type, "image/webp")
        self.assertEqual(cutout.width, 384)
        self.assertEqual(cutout.height, 512)


def _synthetic_garment_image_bytes() -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (384, 512), "#f3f3f3")
    draw = ImageDraw.Draw(image)
    draw.polygon(
        [
            (150, 86),
            (234, 86),
            (312, 180),
            (270, 226),
            (242, 186),
            (242, 430),
            (142, 430),
            (142, 186),
            (114, 226),
            (72, 180),
        ],
        fill="#234c8c",
    )
    draw.rectangle((166, 86, 218, 126), fill="#f3f3f3")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
