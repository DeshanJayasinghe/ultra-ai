from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol

from wearwise_ai.application.clothing_classification_service import (
    CLOTHING_CLASSIFICATION_MODEL_FAMILY,
    CLOTHING_CLASSIFICATION_MODEL_NAME,
    CLOTHING_CLASSIFICATION_MODEL_VERSION,
    DEFAULT_CATEGORY_DEFINITIONS,
    ClothingClassificationRequest,
    RawCategoryPrediction,
)
from wearwise_ai.models.background_removal.birefnet_adapter import ModelUnavailableError
from wearwise_ai.models.openai_usage import log_usage, usage_metrics

OPENAI_UNKNOWN_CATEGORY = "unknown"
OPENAI_CLASSIFICATION_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "category",
        "confidence",
        "alternatives",
        "review_required",
        "review_reason",
        "visible_evidence",
    ],
    "properties": {
        "category": {
            "type": "string",
            "enum": [
                "tops",
                "bottoms",
                "shoes",
                "jackets",
                "accessories",
                OPENAI_UNKNOWN_CATEGORY,
            ],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "alternatives": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["category", "confidence"],
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "tops",
                            "bottoms",
                            "shoes",
                            "jackets",
                            "accessories",
                            OPENAI_UNKNOWN_CATEGORY,
                        ],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "review_required": {"type": "boolean"},
        "review_reason": {"type": ["string", "null"]},
        "visible_evidence": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "string"},
        },
    },
}


class OpenAIResponsesClient(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class OpenAIClassificationConfig:
    model_name: str = CLOTHING_CLASSIFICATION_MODEL_NAME
    model_version: str = CLOTHING_CLASSIFICATION_MODEL_VERSION
    image_detail: str = "low"
    reasoning_effort: str = "minimal"
    max_output_tokens: int = 600
    verbosity: str = "low"
    max_image_edge: int = 512


class OpenAICategoryClassifier:
    """CategoryClassifier adapter backed by OpenAI vision and strict JSON output."""

    def __init__(
        self,
        *,
        config: OpenAIClassificationConfig | None = None,
        responses_client: OpenAIResponsesClient | None = None,
    ) -> None:
        self._config = config or OpenAIClassificationConfig()
        self._responses_client = responses_client

    def classify(self, request: ClothingClassificationRequest) -> RawCategoryPrediction:
        response = self._client().create(
            model=self._config.model_name,
            input=[
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _developer_instruction(),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Classify this background-removed wardrobe item.",
                        },
                        {
                            "type": "input_image",
                            "image_url": _data_url(
                                content_type=request.content_type,
                                image_bytes=request.image_bytes,
                                max_edge=self._config.max_image_edge,
                            ),
                            "detail": self._config.image_detail,
                        },
                    ],
                },
            ],
            reasoning={"effort": self._config.reasoning_effort},
            max_output_tokens=self._config.max_output_tokens,
            text={
                "verbosity": self._config.verbosity,
                "format": {
                    "type": "json_schema",
                    "name": "wearwise_clothing_category",
                    "strict": True,
                    "schema": OPENAI_CLASSIFICATION_RESPONSE_SCHEMA,
                }
            },
        )
        log_usage(
            usage_metrics(response, model_name=self._config.model_name),
            job_type="clothing_classification",
        )
        payload = _response_json(response)
        category = str(payload["category"])
        confidence = _score(payload["confidence"])
        alternatives = _alternatives(payload.get("alternatives", []))
        review_reason = payload.get("review_reason")

        return RawCategoryPrediction(
            category_code=category,
            confidence=confidence,
            model_family=CLOTHING_CLASSIFICATION_MODEL_FAMILY,
            model_name=self._config.model_name,
            model_version=self._config.model_version,
            alternatives=alternatives,
            review_required=bool(payload["review_required"]) or category == OPENAI_UNKNOWN_CATEGORY,
            review_reason=review_reason if isinstance(review_reason, str) else None,
        )

    def _client(self) -> OpenAIResponsesClient:
        if self._responses_client is not None:
            return self._responses_client

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelUnavailableError(
                "OpenAI SDK is not installed. Install wearwise-ai dependencies "
                "and set OPENAI_API_KEY."
            ) from exc

        self._responses_client = OpenAI().responses
        return self._responses_client


def _developer_instruction() -> str:
    categories = ", ".join(category.code for category in DEFAULT_CATEGORY_DEFINITIONS)
    return (
        "You are WearWise's garment category classifier. Classify exactly one visible wardrobe "
        f"item into one of these categories: {categories}. Use unknown when the image does not "
        "clearly show a wardrobe item or when evidence is insufficient. Do not guess. Set "
        "review_required to true when uncertain, when multiple categories are plausible, or when "
        "the item is not in the taxonomy. Alternatives must be ranked from most to least likely "
        "and must not repeat the selected category."
    )


def _data_url(*, content_type: str, image_bytes: bytes, max_edge: int) -> str:
    image_bytes, content_type = _downscaled(
        content_type=content_type,
        image_bytes=image_bytes,
        max_edge=max_edge,
    )
    encoded = base64.b64encode(image_bytes).decode("ascii")
    safe_content_type = content_type if content_type.startswith("image/") else "image/png"
    return f"data:{safe_content_type};base64,{encoded}"


def _downscaled(*, content_type: str, image_bytes: bytes, max_edge: int) -> tuple[bytes, str]:
    if max_edge <= 0:
        return image_bytes, content_type

    try:
        from PIL import Image
    except ImportError:
        return image_bytes, content_type

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if max(image.size) <= max_edge:
                return image_bytes, content_type

            rgba = image.convert("RGBA")
            rgba.thumbnail((max_edge, max_edge), Image.LANCZOS)
            buffer = BytesIO()
            rgba.save(buffer, format="WEBP", quality=85)
            return buffer.getvalue(), "image/webp"
    except Exception:
        return image_bytes, content_type


def _response_json(response: Any) -> dict[str, Any]:
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or output_text.strip() == "":
        raise ModelUnavailableError("OpenAI clothing classifier returned an empty response.")

    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ModelUnavailableError("OpenAI clothing classifier returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise ModelUnavailableError("OpenAI clothing classifier returned a non-object payload.")

    return payload


def _alternatives(value: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, list):
        return ()

    alternatives: list[tuple[str, float]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        if not isinstance(category, str) or category in seen or category == OPENAI_UNKNOWN_CATEGORY:
            continue
        alternatives.append((category, _score(item.get("confidence", 0))))
        seen.add(category)

    return tuple(alternatives)


def _score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    return round(max(0.0, min(1.0, score)), 6)
