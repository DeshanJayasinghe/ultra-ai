from __future__ import annotations

from contextlib import nullcontext
from io import BytesIO
import unittest

from wearwise_ai.application.embedding_generation_service import EmbeddingGenerationRequest
from wearwise_ai.models.background_removal.birefnet_adapter import ModelUnavailableError
from wearwise_ai.models.embedding import (
    HuggingFaceSigLIPRunnerConfig,
    SigLIPConfig,
    TransformersSigLIPImageEmbedder,
)


class FakeTensor:
    def __init__(self, values: list[float] | None = None) -> None:
        self.values = values or [1.0, 2.0, 2.0]
        self.devices: list[str] = []

    def to(self, device: str) -> "FakeTensor":
        self.devices.append(device)
        return self

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def flatten(self) -> "FakeTensor":
        return self

    def tolist(self) -> list[float]:
        return self.values


class FakeImageProcessor:
    def __init__(self) -> None:
        self.called = False
        self.pixel_values = FakeTensor()

    def __call__(self, *, images: object, return_tensors: str) -> dict[str, FakeTensor]:
        self.called = True
        return {"pixel_values": self.pixel_values}


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[FakeTensor] = []

    def get_image_features(self, *, pixel_values: FakeTensor) -> FakeTensor:
        self.calls.append(pixel_values)
        return FakeTensor([3.0, 4.0])


class FakeTorch:
    def inference_mode(self) -> object:
        return nullcontext()


class HuggingFaceSigLIPRunnerTests(unittest.TestCase):
    def test_runner_loads_processor_and_model_to_return_image_features(self) -> None:
        model = FakeModel()
        image_processor = FakeImageProcessor()
        embedder = TransformersSigLIPImageEmbedder(
            runner_config=HuggingFaceSigLIPRunnerConfig(
                model_id="google/siglip-base-patch16-224",
                device="cuda",
            ),
            required_modules=("PIL",),
            model_loader=lambda config: model,
            image_processor_loader=lambda config: image_processor,
            torch_loader=lambda: FakeTorch(),
        )

        values = embedder.embed_image(
            _request(),
            SigLIPConfig(model_name="google/siglip-base-patch16-224", model_version="rev-1"),
        )

        self.assertEqual(values, (3.0, 4.0))
        self.assertTrue(image_processor.called)
        self.assertEqual(model.calls, [image_processor.pixel_values])
        self.assertEqual(image_processor.pixel_values.devices, ["cuda"])
        self.assertEqual(embedder.last_metrics["model_id"], "google/siglip-base-patch16-224")
        self.assertEqual(embedder.last_metrics["embedding_dimensions"], 2)

    def test_default_runner_can_be_forced_to_fail_without_download_attempt(self) -> None:
        embedder = TransformersSigLIPImageEmbedder(
            runner_config=HuggingFaceSigLIPRunnerConfig(model_id="google/siglip-base-patch16-224"),
            required_modules=("wearwise_missing_test_module",),
        )

        with self.assertRaises(ModelUnavailableError):
            embedder.embed_image(_request(), SigLIPConfig())


def _request() -> EmbeddingGenerationRequest:
    from PIL import Image

    image = Image.new("RGB", (2, 2), color=(30, 90, 170))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return EmbeddingGenerationRequest(
        asset_id="asset-1",
        image_bytes=buffer.getvalue(),
        content_type="image/png",
    )


if __name__ == "__main__":
    unittest.main()
