from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from math import exp, sqrt
from time import perf_counter
from typing import Any

from wearwise_ai.application.clothing_classification_service import (
    DEFAULT_CATEGORY_DEFINITIONS,
    SIGLIP_CLASSIFICATION_MODEL_FAMILY,
    SIGLIP_CLASSIFICATION_MODEL_NAME,
    SIGLIP_CLASSIFICATION_MODEL_VERSION,
    CategoryDefinition,
    ClothingClassificationRequest,
    RawCategoryPrediction,
)
from wearwise_ai.models.background_removal.birefnet_adapter import ModelUnavailableError
from wearwise_ai.models.embedding.huggingface_runner import (
    HuggingFaceSigLIPRunnerConfig,
    _inference_context,
    _load_siglip_model,
    _load_torch_module,
    _move_tensor_to_device,
)
from wearwise_ai.models.embedding.siglip_adapter import check_siglip_dependencies

DEFAULT_SIGLIP_CLASSIFICATION_MODULES = (
    "PIL",
    "numpy",
    "sentencepiece",
    "torch",
    "transformers",
)


@dataclass(frozen=True, slots=True)
class ZeroShotPromptSet:
    templates: tuple[str, ...] = (
        "a photo of {label}",
        "a photo of a {label}",
        "a clean product photo of {label}",
    )

    def prompts_for(self, category: CategoryDefinition) -> tuple[str, ...]:
        return tuple(template.format(label=category.label) for template in self.templates)


class TransformersSigLIPZeroShotCategoryRunner:
    """SigLIP zero-shot category classifier for the WearWise category taxonomy."""

    def __init__(
        self,
        *,
        runner_config: HuggingFaceSigLIPRunnerConfig,
        categories: tuple[CategoryDefinition, ...] = DEFAULT_CATEGORY_DEFINITIONS,
        prompt_set: ZeroShotPromptSet | None = None,
        required_modules: tuple[str, ...] | None = None,
        model_loader: Callable[[HuggingFaceSigLIPRunnerConfig], Any] | None = None,
        processor_loader: Callable[[HuggingFaceSigLIPRunnerConfig], Any] | None = None,
        torch_loader: Callable[[], Any] | None = None,
    ) -> None:
        self._runner_config = runner_config
        self._categories = categories
        self._prompt_set = prompt_set or ZeroShotPromptSet()
        self._required_modules = required_modules
        self._model_loader = model_loader or _load_siglip_model
        self._processor_loader = processor_loader or _load_siglip_processor
        self._torch_loader = torch_loader or _load_torch_module
        self._loaded_model: Any | None = None
        self._loaded_processor: Any | None = None

    def classify_category(self, request: ClothingClassificationRequest) -> RawCategoryPrediction:
        started_at = perf_counter()
        status = check_siglip_dependencies(self._required_modules or ())
        if self._required_modules is None:
            status = check_siglip_dependencies(DEFAULT_SIGLIP_CLASSIFICATION_MODULES)

        if not status.is_ready:
            missing = ", ".join(status.missing_modules)
            raise ModelUnavailableError(
                "SigLIP zero-shot classification dependencies are not installed: "
                f"{missing}. Install wearwise-ai[ml] in the worker image."
            )

        if self._categories == ():
            raise ModelUnavailableError("SigLIP zero-shot classifier has no categories configured.")

        torch = self._torch_loader()

        from PIL import Image

        with Image.open(BytesIO(request.image_bytes)) as input_image:
            image = input_image.convert("RGB")

        prompts_by_category = [
            (category, self._prompt_set.prompts_for(category)) for category in self._categories
        ]
        prompts = [
            prompt for _, category_prompts in prompts_by_category for prompt in category_prompts
        ]

        processor = self._get_processor()
        model = self._get_model()
        inputs = processor(images=image, text=prompts, padding=True, return_tensors="pt")
        inputs = {
            key: _move_tensor_to_device(value, self._runner_config.device)
            for key, value in inputs.items()
        }

        image_inputs = {"pixel_values": inputs["pixel_values"]}
        text_inputs = {
            key: value
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }

        inference_started_at = perf_counter()
        with _inference_context(torch):
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
        inference_duration_ms = round((perf_counter() - inference_started_at) * 1000, 3)

        image_vector = _feature_rows(image_features)[0]
        text_vectors = _feature_rows(text_features)
        category_scores = _category_scores(
            image_vector=image_vector,
            text_vectors=text_vectors,
            prompts_by_category=prompts_by_category,
        )
        confidences = _softmax([score for _, score in category_scores])
        ranked = sorted(
            [
                (category.code, score, confidences[index])
                for index, (category, score) in enumerate(category_scores)
            ],
            key=lambda item: item[2],
            reverse=True,
        )

        best_code, _, confidence = ranked[0]
        alternatives = tuple((code, round(conf, 6)) for code, _, conf in ranked[1:4])
        self.last_metrics = {
            "predictor_duration_ms": round((perf_counter() - started_at) * 1000, 3),
            "inference_duration_ms": inference_duration_ms,
            "model_id": self._runner_config.model_id,
            "device": self._runner_config.device,
            "category_count": len(self._categories),
            "prompt_count": len(prompts),
        }

        return RawCategoryPrediction(
            category_code=best_code,
            confidence=round(confidence, 6),
            model_family=SIGLIP_CLASSIFICATION_MODEL_FAMILY,
            model_name=self._runner_config.model_id or SIGLIP_CLASSIFICATION_MODEL_NAME,
            model_version=self._runner_config.revision or SIGLIP_CLASSIFICATION_MODEL_VERSION,
            alternatives=alternatives,
        )

    def _get_model(self) -> Any:
        if self._loaded_model is None:
            self._loaded_model = self._model_loader(self._runner_config)
        return self._loaded_model

    def _get_processor(self) -> Any:
        if self._loaded_processor is None:
            self._loaded_processor = self._processor_loader(self._runner_config)
        return self._loaded_processor


def _load_siglip_processor(runner_config: HuggingFaceSigLIPRunnerConfig) -> Any:
    try:
        from transformers import SiglipProcessor
    except ImportError as exc:
        raise ModelUnavailableError(
            "transformers with SiglipProcessor is required for SigLIP zero-shot classification."
        ) from exc

    kwargs: dict[str, object] = {
        "local_files_only": runner_config.local_files_only,
    }
    if runner_config.revision:
        kwargs["revision"] = runner_config.revision

    return SiglipProcessor.from_pretrained(runner_config.model_id, **kwargs)


def _category_scores(
    *,
    image_vector: tuple[float, ...],
    text_vectors: list[tuple[float, ...]],
    prompts_by_category: list[tuple[CategoryDefinition, tuple[str, ...]]],
) -> list[tuple[CategoryDefinition, float]]:
    scores: list[tuple[CategoryDefinition, float]] = []
    offset = 0
    for category, prompts in prompts_by_category:
        prompt_vectors = text_vectors[offset : offset + len(prompts)]
        offset += len(prompts)
        prompt_scores = [_cosine_similarity(image_vector, vector) for vector in prompt_vectors]
        scores.append((category, sum(prompt_scores) / len(prompt_scores)))

    return scores


def _feature_rows(features: Any) -> list[tuple[float, ...]]:
    value = features
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()

    if not isinstance(value, list):
        raise ModelUnavailableError("SigLIP feature output was not list-like.")

    if value and not isinstance(value[0], list):
        return [tuple(float(item) for item in value)]

    return [tuple(float(item) for item in row) for row in value]


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or len(left) == 0:
        raise ModelUnavailableError("SigLIP image and text feature dimensions do not match.")

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ModelUnavailableError("SigLIP feature norm must be greater than zero.")

    return dot_product / (left_norm * right_norm)


def _softmax(scores: list[float], temperature: float = 10.0) -> list[float]:
    scaled = [score * temperature for score in scores]
    max_score = max(scaled)
    exps = [exp(score - max_score) for score in scaled]
    total = sum(exps)
    return [value / total for value in exps]
