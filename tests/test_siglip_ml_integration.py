from __future__ import annotations

from io import BytesIO
import os
import unittest

from wearwise_ai.application.embedding_generation_service import EmbeddingGenerationRequest
from wearwise_ai.application.embedding_generation_service import EMBEDDING_MODEL_VERSION
from wearwise_ai.models.embedding import (
    HuggingFaceSigLIPRunnerConfig,
    SigLIPConfig,
    SigLIPEmbeddingGenerator,
    TransformersSigLIPImageEmbedder,
)


@unittest.skipUnless(
    os.getenv("WEARWISE_RUN_ML_TESTS") == "1",
    "set WEARWISE_RUN_ML_TESTS=1 to run real SigLIP integration tests",
)
class SigLIPMlIntegrationTests(unittest.TestCase):
    def test_real_siglip_runner_generates_image_embedding(self) -> None:
        model_id = os.getenv("WEARWISE_SIGLIP_MODEL_ID", "google/siglip-base-patch16-224")
        revision = os.getenv("WEARWISE_SIGLIP_MODEL_REVISION") or EMBEDDING_MODEL_VERSION
        generator = SigLIPEmbeddingGenerator(
            config=SigLIPConfig(
                model_name=model_id,
                model_version=revision,
            ),
            runner=TransformersSigLIPImageEmbedder(
                runner_config=HuggingFaceSigLIPRunnerConfig(
                    model_id=model_id,
                    revision=revision,
                    device=os.getenv("WEARWISE_SIGLIP_DEVICE", "cpu"),
                    local_files_only=os.getenv("WEARWISE_SIGLIP_LOCAL_FILES_ONLY") == "1",
                )
            ),
        )

        embedding = generator.embed_image(
            EmbeddingGenerationRequest(
                asset_id="synthetic-top",
                image_bytes=_synthetic_garment_image_bytes(),
                content_type="image/png",
            )
        )

        self.assertEqual(embedding.model_family, "siglip")
        self.assertEqual(embedding.model_name, model_id)
        self.assertGreater(len(embedding.values), 0)


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
