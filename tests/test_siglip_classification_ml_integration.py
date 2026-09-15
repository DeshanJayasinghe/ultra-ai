from __future__ import annotations

import os
import unittest
from io import BytesIO

from wearwise_ai.application.clothing_classification_service import (
    SIGLIP_CLASSIFICATION_MODEL_VERSION,
    ClothingClassificationRequest,
)
from wearwise_ai.models.classification import TransformersSigLIPZeroShotCategoryRunner
from wearwise_ai.models.embedding import HuggingFaceSigLIPRunnerConfig


@unittest.skipUnless(
    os.getenv("WEARWISE_RUN_ML_TESTS") == "1",
    "set WEARWISE_RUN_ML_TESTS=1 to run real SigLIP classification integration tests",
)
class SigLIPClassificationMlIntegrationTests(unittest.TestCase):
    def test_real_siglip_runner_generates_category_prediction(self) -> None:
        model_id = os.getenv("WEARWISE_SIGLIP_MODEL_ID", "google/siglip-base-patch16-224")
        revision = (
            os.getenv("WEARWISE_SIGLIP_MODEL_REVISION") or SIGLIP_CLASSIFICATION_MODEL_VERSION
        )
        runner = TransformersSigLIPZeroShotCategoryRunner(
            runner_config=HuggingFaceSigLIPRunnerConfig(
                model_id=model_id,
                revision=revision,
                device=os.getenv("WEARWISE_SIGLIP_DEVICE", "cpu"),
                local_files_only=os.getenv("WEARWISE_SIGLIP_LOCAL_FILES_ONLY") == "1",
            )
        )

        prediction = runner.classify_category(
            ClothingClassificationRequest(
                asset_id="synthetic-top",
                image_bytes=_synthetic_garment_image_bytes(),
                content_type="image/png",
            )
        )

        self.assertIn(
            prediction.category_code,
            {"tops", "bottoms", "shoes", "jackets", "accessories"},
        )
        self.assertGreaterEqual(prediction.confidence, 0)
        self.assertLessEqual(prediction.confidence, 1)


def _synthetic_garment_image_bytes() -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (224, 224), "#f4f4f4")
    draw = ImageDraw.Draw(image)
    draw.polygon(
        [
            (88, 34),
            (136, 34),
            (184, 82),
            (164, 108),
            (142, 84),
            (142, 194),
            (82, 194),
            (82, 84),
            (60, 108),
            (40, 82),
        ],
        fill="#245aa2",
    )
    draw.rectangle((99, 34, 125, 56), fill="#f4f4f4")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
