from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from io import BytesIO
from time import perf_counter
from typing import Any, Callable

from wearwise_ai.application.embedding_generation_service import EmbeddingGenerationRequest
from wearwise_ai.models.background_removal.birefnet_adapter import ModelUnavailableError
from wearwise_ai.models.embedding.siglip_adapter import (
    SigLIPConfig,
    check_siglip_dependencies,
)


@dataclass(frozen=True, slots=True)
class HuggingFaceSigLIPRunnerConfig:
    model_id: str
    revision: str | None = None
    device: str = "cpu"
    trust_remote_code: bool = False
    use_half_precision: bool = False
    local_files_only: bool = False


class TransformersSigLIPImageEmbedder:
    """Lazy-loading SigLIP image embedder backed by Hugging Face Transformers."""

    def __init__(
        self,
        *,
        runner_config: HuggingFaceSigLIPRunnerConfig,
        required_modules: tuple[str, ...] | None = None,
        model_loader: Callable[[HuggingFaceSigLIPRunnerConfig], Any] | None = None,
        image_processor_loader: Callable[[HuggingFaceSigLIPRunnerConfig], Any] | None = None,
        torch_loader: Callable[[], Any] | None = None,
    ) -> None:
        self._runner_config = runner_config
        self._required_modules = required_modules
        self._model_loader = model_loader or _load_siglip_model
        self._image_processor_loader = image_processor_loader or _load_siglip_image_processor
        self._torch_loader = torch_loader or _load_torch_module
        self._loaded_model: Any | None = None
        self._loaded_processor: Any | None = None

    def embed_image(
        self,
        request: EmbeddingGenerationRequest,
        config: SigLIPConfig,
    ) -> tuple[float, ...]:
        started_at = perf_counter()
        status = check_siglip_dependencies(self._required_modules or ())
        if self._required_modules is None:
            status = check_siglip_dependencies()

        if not status.is_ready:
            missing = ", ".join(status.missing_modules)
            raise ModelUnavailableError(
                "SigLIP Hugging Face runtime dependencies are not installed: "
                f"{missing}. Install wearwise-ai[ml] in the worker image."
            )

        torch = self._torch_loader()

        from PIL import Image

        with Image.open(BytesIO(request.image_bytes)) as input_image:
            image = input_image.convert("RGB")

        image_processor = self._get_image_processor()
        model = self._get_model()
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {
            key: _move_tensor_to_device(value, self._runner_config.device)
            for key, value in inputs.items()
        }

        inference_started_at = perf_counter()
        with _inference_context(torch):
            features = model.get_image_features(**inputs)
        inference_duration_ms = round((perf_counter() - inference_started_at) * 1000, 3)

        if self._runner_config.use_half_precision and hasattr(features, "float"):
            features = features.float()

        values = _features_to_tuple(features)
        if len(values) == 0:
            raise ModelUnavailableError("SigLIP image feature output was empty.")

        self.last_metrics = {
            "predictor_duration_ms": round((perf_counter() - started_at) * 1000, 3),
            "inference_duration_ms": inference_duration_ms,
            "model_id": self._runner_config.model_id,
            "device": self._runner_config.device,
            "adapter_model_name": config.model_name,
            "embedding_dimensions": len(values),
        }
        return values

    def _get_model(self) -> Any:
        if self._loaded_model is None:
            self._loaded_model = self._model_loader(self._runner_config)
        return self._loaded_model

    def _get_image_processor(self) -> Any:
        if self._loaded_processor is None:
            self._loaded_processor = self._image_processor_loader(self._runner_config)
        return self._loaded_processor


def _load_siglip_model(runner_config: HuggingFaceSigLIPRunnerConfig) -> Any:
    modules = _load_transformers_modules()
    model_cls = modules["SiglipModel"]
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


def _load_siglip_image_processor(runner_config: HuggingFaceSigLIPRunnerConfig) -> Any:
    modules = _load_transformers_modules()
    processor_cls = modules["SiglipImageProcessor"]
    kwargs: dict[str, object] = {
        "local_files_only": runner_config.local_files_only,
    }
    if runner_config.revision:
        kwargs["revision"] = runner_config.revision

    return processor_cls.from_pretrained(runner_config.model_id, **kwargs)


def _load_transformers_modules() -> dict[str, Any]:
    try:
        from transformers import SiglipImageProcessor, SiglipModel
    except ImportError as exc:
        raise ModelUnavailableError(
            "transformers with SiglipModel and SiglipImageProcessor is required for SigLIP inference."
        ) from exc

    return {
        "SiglipModel": SiglipModel,
        "SiglipImageProcessor": SiglipImageProcessor,
    }


def _load_torch_module() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ModelUnavailableError("torch is required for SigLIP inference.") from exc

    return torch


def _move_tensor_to_device(value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    return value


def _inference_context(torch_module: Any) -> Any:
    context_factory = getattr(torch_module, "inference_mode", None)
    if context_factory is None:
        return nullcontext()
    return context_factory()


def _features_to_tuple(features: Any) -> tuple[float, ...]:
    value = features
    for method_name in ("detach", "cpu", "flatten"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()

    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]

    return tuple(float(item) for item in value)
