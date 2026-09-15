from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from time import perf_counter
from typing import Any, Callable, Protocol

from wearwise_ai.application.background_removal_service import (
    BackgroundRemovalOutput,
    BackgroundRemovalRequest,
)
from wearwise_ai.models.background_removal.birefnet_adapter import (
    BiRefNetConfig,
    ModelUnavailableError,
    check_birefnet_dependencies,
)
from wearwise_ai.models.background_removal.mask_postprocessor import (
    MaskPostprocessor,
    PillowAlphaMaskPostprocessor,
)


@dataclass(frozen=True, slots=True)
class HuggingFaceBiRefNetRunnerConfig:
    model_id: str
    revision: str | None = None
    device: str = "cpu"
    trust_remote_code: bool = True
    input_width: int = 1024
    input_height: int = 1024
    use_half_precision: bool = False
    local_files_only: bool = False


class BiRefNetPredictor(Protocol):
    def predict_mask(
        self,
        *,
        image_bytes: bytes,
        runner_config: HuggingFaceBiRefNetRunnerConfig,
    ) -> tuple[bytes, float, dict[str, int | float | str]]: ...


class HuggingFaceBiRefNetRunner:
    """BiRefNet runner that delegates inference to a Hugging Face-backed predictor.

    The predictor boundary keeps framework-specific model code isolated while the
    runner owns the WearWise output contract and mask post-processing step.
    """

    def __init__(
        self,
        *,
        runner_config: HuggingFaceBiRefNetRunnerConfig,
        predictor: BiRefNetPredictor | None = None,
        postprocessor: MaskPostprocessor | None = None,
    ) -> None:
        self._runner_config = runner_config
        self._predictor = predictor
        self._postprocessor = postprocessor or PillowAlphaMaskPostprocessor()

    def remove_background(
        self,
        request: BackgroundRemovalRequest,
        config: BiRefNetConfig,
    ) -> BackgroundRemovalOutput:
        started_at = perf_counter()
        predictor = self._predictor or TransformersBiRefNetPredictor()
        mask_bytes, confidence, metrics = predictor.predict_mask(
            image_bytes=request.image_bytes,
            runner_config=self._runner_config,
        )
        metrics = {
            **metrics,
            "runner_duration_ms": round((perf_counter() - started_at) * 1000, 3),
            "model_id": self._runner_config.model_id,
            "device": self._runner_config.device,
            "adapter_model_name": config.model_name,
        }

        return self._postprocessor.build_cutout(
            request=request,
            mask_bytes=mask_bytes,
            confidence=confidence,
            metrics=metrics,
        )


class TransformersBiRefNetPredictor:
    """Lazy-loading predictor for BiRefNet Hugging Face inference.

    The predictor follows the official model-card flow without requiring
    torchvision: resize to 1024x1024, normalize with ImageNet statistics, run
    the model, take the final prediction, apply sigmoid, and return a grayscale
    mask resized back to the original image size.
    """

    def __init__(
        self,
        *,
        required_modules: tuple[str, ...] | None = None,
        model_loader: Callable[[HuggingFaceBiRefNetRunnerConfig], Any] | None = None,
    ) -> None:
        self._required_modules = required_modules
        self._model_loader = model_loader or _load_birefnet_model
        self._loaded_model: Any | None = None

    def predict_mask(
        self,
        *,
        image_bytes: bytes,
        runner_config: HuggingFaceBiRefNetRunnerConfig,
    ) -> tuple[bytes, float, dict[str, int | float | str]]:
        started_at = perf_counter()
        status = check_birefnet_dependencies(self._required_modules or ())
        if self._required_modules is None:
            status = check_birefnet_dependencies()

        if not status.is_ready:
            missing = ", ".join(status.missing_modules)
            raise ModelUnavailableError(
                "BiRefNet Hugging Face runtime dependencies are not installed: "
                f"{missing}. Install wearwise-ai[ml] in the worker image."
            )

        # Keep these imports lazy so the service can boot in API-only/test modes.
        import numpy as np
        from PIL import Image
        import torch

        with Image.open(BytesIO(image_bytes)) as input_image:
            original_image = input_image.convert("RGB")

        original_size = original_image.size
        resized_image = original_image.resize(
            (runner_config.input_width, runner_config.input_height),
            _pil_bilinear_resample(Image),
        )

        image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
        image_array = (image_array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )
        image_array = np.transpose(image_array, (2, 0, 1))

        input_tensor = torch.from_numpy(image_array).unsqueeze(0).to(runner_config.device)
        if runner_config.use_half_precision:
            input_tensor = input_tensor.half()

        model = self._get_model(runner_config)

        inference_started_at = perf_counter()
        with torch.inference_mode():
            output = model(input_tensor)
        inference_duration_ms = round((perf_counter() - inference_started_at) * 1000, 3)

        prediction = _select_final_prediction(output)
        prediction = prediction.sigmoid().detach().cpu()
        if prediction.ndim == 4:
            prediction = prediction[0, 0]
        elif prediction.ndim == 3:
            prediction = prediction[0]

        mask_array = prediction.numpy()
        mask_array = np.clip(mask_array * 255.0, 0, 255).astype(np.uint8)

        mask_image = Image.fromarray(mask_array).convert("L").resize(
            original_size,
            _pil_bilinear_resample(Image),
        )
        mask_buffer = BytesIO()
        mask_image.save(mask_buffer, format="PNG")

        confidence = float(np.mean(mask_array) / 255.0)

        return (
            mask_buffer.getvalue(),
            confidence,
            {
                "predictor_duration_ms": round((perf_counter() - started_at) * 1000, 3),
                "inference_duration_ms": inference_duration_ms,
                "mask_width": original_size[0],
                "mask_height": original_size[1],
                "input_width": runner_config.input_width,
                "input_height": runner_config.input_height,
            },
        )

    def _get_model(self, runner_config: HuggingFaceBiRefNetRunnerConfig) -> Any:
        if self._loaded_model is None:
            self._loaded_model = self._model_loader(runner_config)
        return self._loaded_model


def _load_birefnet_model(runner_config: HuggingFaceBiRefNetRunnerConfig) -> Any:
    modules = _load_transformers_modules()
    model_cls = modules["AutoModelForImageSegmentation"]
    kwargs: dict[str, object] = {
        "trust_remote_code": runner_config.trust_remote_code,
        "local_files_only": runner_config.local_files_only,
    }
    if runner_config.revision:
        kwargs["revision"] = runner_config.revision

    model = model_cls.from_pretrained(runner_config.model_id, **kwargs)
    model.to(runner_config.device)
    model.eval()
    if runner_config.use_half_precision:
        model.half()
    return model


def _select_final_prediction(output: Any) -> Any:
    if isinstance(output, (list, tuple)):
        return output[-1]

    if isinstance(output, dict):
        for key in ("logits", "pred_masks", "prediction"):
            if key in output:
                return output[key]

    for attr_name in ("logits", "pred_masks", "prediction"):
        if hasattr(output, attr_name):
            return getattr(output, attr_name)

    raise ModelUnavailableError("BiRefNet model output did not contain a usable mask prediction.")


def _pil_bilinear_resample(image_module: Any) -> Any:
    return getattr(getattr(image_module, "Resampling", image_module), "BILINEAR")


def _load_transformers_modules() -> dict[str, Any]:
    try:
        from transformers import AutoImageProcessor, AutoModelForImageSegmentation
    except ImportError as exc:
        raise ModelUnavailableError(
            "transformers with AutoModelForImageSegmentation is required for BiRefNet inference."
        ) from exc

    return {
        "AutoImageProcessor": AutoImageProcessor,
        "AutoModelForImageSegmentation": AutoModelForImageSegmentation,
    }
