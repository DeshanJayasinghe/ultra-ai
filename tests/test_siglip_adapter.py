from __future__ import annotations

import unittest

from wearwise_ai.application.embedding_generation_service import EmbeddingGenerationRequest
from wearwise_ai.models.background_removal.birefnet_adapter import ModelUnavailableError
from wearwise_ai.models.embedding.siglip_adapter import (
    SigLIPConfig,
    SigLIPEmbeddingGenerator,
    check_siglip_dependencies,
)


class FakeSigLIPRunner:
    def embed_image(
        self,
        request: EmbeddingGenerationRequest,
        config: SigLIPConfig,
    ) -> tuple[float, ...]:
        return (1.0, 2.0, 2.0)


class SigLIPAdapterTests(unittest.TestCase):
    def test_dependency_check_reports_missing_modules(self) -> None:
        status = check_siglip_dependencies(("module_that_should_not_exist_for_wearwise",))

        self.assertFalse(status.is_ready)
        self.assertEqual(status.missing_modules, ("module_that_should_not_exist_for_wearwise",))

    def test_raises_controlled_error_when_no_runner_is_configured(self) -> None:
        generator = SigLIPEmbeddingGenerator(
            config=SigLIPConfig(required_modules=("math",)),
        )

        with self.assertRaisesRegex(ModelUnavailableError, "no model runner is configured"):
            generator.embed_image(
                EmbeddingGenerationRequest(
                    asset_id="asset-1",
                    image_bytes=b"image",
                    content_type="image/webp",
                )
            )

    def test_delegates_to_configured_runner(self) -> None:
        generator = SigLIPEmbeddingGenerator(
            config=SigLIPConfig(model_version="rev-1"),
            runner=FakeSigLIPRunner(),
        )

        embedding = generator.embed_image(
            EmbeddingGenerationRequest(
                asset_id="asset-1",
                image_bytes=b"image",
                content_type="image/webp",
            )
        )

        self.assertEqual(embedding.values, (1.0, 2.0, 2.0))
        self.assertEqual(embedding.model_family, "siglip")
        self.assertEqual(embedding.model_version, "rev-1")


if __name__ == "__main__":
    unittest.main()
