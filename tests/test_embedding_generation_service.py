from __future__ import annotations

import unittest

from wearwise_ai.application.embedding_generation_service import (
    EMBEDDING_GENERATION_PROCESSING_VERSION,
    EmbeddingGenerationRequest,
    EmbeddingGenerationService,
    RawEmbedding,
    normalize_embedding,
)


class FakeEmbedder:
    def embed_image(self, request: EmbeddingGenerationRequest) -> RawEmbedding:
        return RawEmbedding(
            values=(3.0, 4.0),
            model_family="siglip",
            model_name="test-siglip",
            model_version="abc123",
        )


class EmbeddingGenerationServiceTests(unittest.TestCase):
    def test_normalizes_embedding_vector(self) -> None:
        embedding = normalize_embedding((3.0, 4.0))

        self.assertEqual(embedding.values, (0.6, 0.8))
        self.assertEqual(embedding.dimensions, 2)
        self.assertEqual(embedding.norm, 5.0)
        self.assertEqual(embedding.normalization, "l2")

    def test_rejects_empty_embedding_vector(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            normalize_embedding(())

    def test_rejects_zero_norm_embedding_vector(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            normalize_embedding((0.0, 0.0))

    def test_generates_ai_complete_payload(self) -> None:
        service = EmbeddingGenerationService(embedder=FakeEmbedder())

        result = service.generate(
            EmbeddingGenerationRequest(
                asset_id="asset-1",
                image_bytes=b"image",
                content_type="image/webp",
            )
        )

        self.assertEqual(result.status, "ready")
        self.assertEqual(result.model_name, "test-siglip")
        self.assertEqual(result.processing_version, EMBEDDING_GENERATION_PROCESSING_VERSION)
        payload = result.to_complete_payload(worker_id="worker-a")
        self.assertEqual(payload["status_message"], "embedding generation completed")
        self.assertFalse(payload["result"]["review_required"])
        self.assertEqual(payload["result"]["embedding"]["values"], [0.6, 0.8])
        self.assertEqual(payload["result"]["embedding_metadata"]["model_version"], "abc123")
        self.assertEqual(payload["resource_metrics"]["embedding_dimensions"], 2)


if __name__ == "__main__":
    unittest.main()
