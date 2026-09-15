from __future__ import annotations

import unittest
from unittest.mock import patch

from wearwise_ai.application.background_removal_service import (
    BackgroundRemovalOutput,
    BackgroundRemovalRequest,
)
from wearwise_ai.models.background_removal import (
    BiRefNetConfig,
    HuggingFaceBiRefNetRunner,
    HuggingFaceBiRefNetRunnerConfig,
    ModelUnavailableError,
    TransformersBiRefNetPredictor,
)
from wearwise_ai.preprocessing import ImageProbe, plan_normalization


class FakePredictor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict_mask(
        self,
        *,
        image_bytes: bytes,
        runner_config: HuggingFaceBiRefNetRunnerConfig,
    ) -> tuple[bytes, float, dict[str, int | float | str]]:
        self.calls.append(runner_config.model_id)
        return b"mask", 0.93, {"predictor_duration_ms": 81}


class FakePostprocessor:
    def __init__(self) -> None:
        self.mask_bytes: bytes | None = None
        self.metrics: dict[str, int | float | str] | None = None

    def build_cutout(
        self,
        *,
        request: BackgroundRemovalRequest,
        mask_bytes: bytes,
        confidence: float,
        metrics: dict[str, int | float | str] | None = None,
    ) -> BackgroundRemovalOutput:
        self.mask_bytes = mask_bytes
        self.metrics = metrics
        return BackgroundRemovalOutput(
            image_bytes=b"cutout",
            content_type="image/webp",
            confidence=confidence,
            width=request.normalization_plan.target_canvas_width,
            height=request.normalization_plan.target_canvas_height,
            metrics=metrics,
        )


def _request() -> BackgroundRemovalRequest:
    return BackgroundRemovalRequest(
        asset_id="asset-1",
        image_bytes=b"image",
        normalization_plan=plan_normalization(
            ImageProbe(width=600, height=800, format_name="jpeg", mime_type="image/jpeg")
        ),
    )


class HuggingFaceBiRefNetRunnerTests(unittest.TestCase):
    def test_runner_delegates_prediction_and_postprocessing(self) -> None:
        predictor = FakePredictor()
        postprocessor = FakePostprocessor()
        runner = HuggingFaceBiRefNetRunner(
            runner_config=HuggingFaceBiRefNetRunnerConfig(
                model_id="ZhengPeng7/BiRefNet",
                device="cuda",
            ),
            predictor=predictor,
            postprocessor=postprocessor,
        )

        output = runner.remove_background(_request(), BiRefNetConfig(model_name="BiRefNet"))

        self.assertEqual(output.image_bytes, b"cutout")
        self.assertEqual(output.confidence, 0.93)
        self.assertEqual(predictor.calls, ["ZhengPeng7/BiRefNet"])
        self.assertEqual(postprocessor.mask_bytes, b"mask")
        self.assertEqual(postprocessor.metrics["model_id"], "ZhengPeng7/BiRefNet")
        self.assertEqual(postprocessor.metrics["device"], "cuda")
        self.assertEqual(postprocessor.metrics["adapter_model_name"], "BiRefNet")

    def test_default_predictor_can_be_forced_to_fail_without_download_attempt(self) -> None:
        predictor = TransformersBiRefNetPredictor(
            required_modules=("wearwise_missing_test_module",)
        )

        with self.assertRaises(ModelUnavailableError):
            predictor.predict_mask(
                image_bytes=b"image",
                runner_config=HuggingFaceBiRefNetRunnerConfig(model_id="ZhengPeng7/BiRefNet"),
            )

    def test_runner_uses_default_predictor_when_no_predictor_is_injected(self) -> None:
        with patch(
            "wearwise_ai.models.background_removal.huggingface_runner."
            "TransformersBiRefNetPredictor"
        ) as predictor_class:
            predictor_class.return_value.predict_mask.return_value = (
                b"mask",
                0.77,
                {"predictor_duration_ms": 12},
            )
            postprocessor = FakePostprocessor()
            runner = HuggingFaceBiRefNetRunner(
                runner_config=HuggingFaceBiRefNetRunnerConfig(model_id="ZhengPeng7/BiRefNet"),
                postprocessor=postprocessor,
            )

            output = runner.remove_background(_request(), BiRefNetConfig())

        self.assertEqual(output.confidence, 0.77)
        self.assertEqual(postprocessor.mask_bytes, b"mask")


if __name__ == "__main__":
    unittest.main()
