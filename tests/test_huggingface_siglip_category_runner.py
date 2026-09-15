from __future__ import annotations

from contextlib import nullcontext
from io import BytesIO
import unittest

from wearwise_ai.application.clothing_classification_service import (
    CategoryDefinition,
    ClothingClassificationRequest,
)
from wearwise_ai.models.background_removal.birefnet_adapter import ModelUnavailableError
from wearwise_ai.models.classification import (
    TransformersSigLIPZeroShotCategoryRunner,
    ZeroShotPromptSet,
)
from wearwise_ai.models.embedding import HuggingFaceSigLIPRunnerConfig


class FakeTensor:
    def __init__(self, values: list[float] | list[list[float]]) -> None:
        self.values = values
        self.devices: list[str] = []

    def to(self, device: str) -> "FakeTensor":
        self.devices.append(device)
        return self

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list[float] | list[list[float]]:
        return self.values


class FakeProcessor:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.pixel_values = FakeTensor([[1.0, 0.0]])
        self.input_ids = FakeTensor([[1], [2]])
        self.attention_mask = FakeTensor([[1], [1]])

    def __call__(
        self,
        *,
        images: object,
        text: list[str],
        padding: bool,
        return_tensors: str,
    ) -> dict[str, FakeTensor]:
        self.prompts = text
        return {
            "pixel_values": self.pixel_values,
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }


class FakeModel:
    def __init__(self) -> None:
        self.image_calls: list[FakeTensor] = []
        self.text_calls: list[tuple[FakeTensor, FakeTensor]] = []

    def get_image_features(self, *, pixel_values: FakeTensor) -> FakeTensor:
        self.image_calls.append(pixel_values)
        return FakeTensor([[1.0, 0.0]])

    def get_text_features(self, *, input_ids: FakeTensor, attention_mask: FakeTensor) -> FakeTensor:
        self.text_calls.append((input_ids, attention_mask))
        return FakeTensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )


class FakeTorch:
    def inference_mode(self) -> object:
        return nullcontext()


class HuggingFaceSigLIPCategoryRunnerTests(unittest.TestCase):
    def test_runner_scores_category_prompts_and_returns_best_category(self) -> None:
        model = FakeModel()
        processor = FakeProcessor()
        runner = TransformersSigLIPZeroShotCategoryRunner(
            runner_config=HuggingFaceSigLIPRunnerConfig(
                model_id="google/siglip-base-patch16-224",
                revision="rev-1",
                device="cuda",
            ),
            categories=(
                CategoryDefinition("tops", "tops"),
                CategoryDefinition("shoes", "shoes"),
            ),
            prompt_set=ZeroShotPromptSet(templates=("a photo of {label}",)),
            required_modules=("PIL",),
            model_loader=lambda config: model,
            processor_loader=lambda config: processor,
            torch_loader=lambda: FakeTorch(),
        )

        prediction = runner.classify_category(_request())

        self.assertEqual(prediction.category_code, "shoes")
        self.assertGreater(prediction.confidence, 0.9)
        self.assertEqual(prediction.alternatives[0][0], "tops")
        self.assertEqual(processor.prompts, ["a photo of tops", "a photo of shoes"])
        self.assertEqual(processor.pixel_values.devices, ["cuda"])
        self.assertEqual(model.image_calls, [processor.pixel_values])
        self.assertEqual(model.text_calls, [(processor.input_ids, processor.attention_mask)])
        self.assertEqual(runner.last_metrics["prompt_count"], 2)

    def test_default_runner_can_be_forced_to_fail_without_download_attempt(self) -> None:
        runner = TransformersSigLIPZeroShotCategoryRunner(
            runner_config=HuggingFaceSigLIPRunnerConfig(model_id="google/siglip-base-patch16-224"),
            required_modules=("wearwise_missing_test_module",),
        )

        with self.assertRaises(ModelUnavailableError):
            runner.classify_category(_request())


def _request() -> ClothingClassificationRequest:
    from PIL import Image

    image = Image.new("RGB", (2, 2), color=(30, 90, 170))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return ClothingClassificationRequest(
        asset_id="asset-1",
        image_bytes=buffer.getvalue(),
        content_type="image/png",
    )


if __name__ == "__main__":
    unittest.main()
