from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol

from wearwise_ai.application.attribute_extraction_service import (
    ATTRIBUTE_EXTRACTION_MODEL_FAMILY,
    AttributeExtractionRequest,
    AttributeSuggestion,
    RawAttributeExtraction,
)
from wearwise_ai.application.colour_extraction_service import (
    COLOUR_EXTRACTION_MODEL_FAMILY,
    COLOUR_EXTRACTION_MODEL_NAME,
    COLOUR_EXTRACTION_MODEL_VERSION,
    ColourExtractionRequest,
    ExtractedColour,
    RawColourExtraction,
)
from wearwise_ai.models.background_removal.birefnet_adapter import ModelUnavailableError
from wearwise_ai.models.openai_usage import log_usage, usage_metrics

COLOUR_CODES = (
    "black",
    "white",
    "grey",
    "navy",
    "blue",
    "green",
    "red",
    "pink",
    "yellow",
    "brown",
    "beige",
    "purple",
)
SUBCATEGORY_VALUES = (
    "t-shirt",
    "shirt",
    "blouse",
    "sweater",
    "hoodie",
    "jeans",
    "trousers",
    "shorts",
    "skirt",
    "trainers",
    "boots",
    "heels",
    "sandals",
    "jacket",
    "coat",
    "blazer",
    "bag",
    "belt",
    "hat",
    "scarf",
)
MATERIAL_VALUES = ("cotton", "denim", "linen", "wool", "leather", "synthetic")
PATTERN_VALUES = ("solid", "striped", "checked", "floral", "graphic", "plain")
SEASON_VALUES = ("spring", "summer", "autumn", "winter", "all-season")
OCCASION_VALUES = ("daily", "work", "formal", "sports", "party", "travel")
FORMALITY_VALUES = ("casual", "smart_casual", "formal", "sporty")
STYLE_TAG_VALUES = ("casual", "smart_casual", "formal", "sporty", "streetwear", "minimal")


class OpenAIResponsesClient(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class OpenAIVisionExtractionConfig:
    model_name: str = COLOUR_EXTRACTION_MODEL_NAME
    model_version: str = COLOUR_EXTRACTION_MODEL_VERSION
    image_detail: str = "low"
    reasoning_effort: str = "minimal"
    max_output_tokens: int = 600
    verbosity: str = "low"
    max_image_edge: int = 512


class OpenAIColourExtractor:
    def __init__(
        self,
        *,
        config: OpenAIVisionExtractionConfig | None = None,
        responses_client: OpenAIResponsesClient | None = None,
    ) -> None:
        self._config = config or OpenAIVisionExtractionConfig()
        self._responses_client = responses_client

    def extract(self, request: ColourExtractionRequest) -> RawColourExtraction:
        response = self._client().create(
            model=self._config.model_name,
            input=_vision_input(
                instruction=(
                    "Identify the primary and secondary visible garment colours. "
                    "Use only the allowed colour values. Prefer the garment pixels, not shadows "
                    "or transparent background."
                ),
                prompt="Extract colours from this background-removed wardrobe item.",
                request=request,
                detail=self._config.image_detail,
                max_image_edge=self._config.max_image_edge,
            ),
            reasoning={"effort": self._config.reasoning_effort},
            max_output_tokens=self._config.max_output_tokens,
            text={
                "verbosity": self._config.verbosity,
                "format": {
                    "type": "json_schema",
                    "name": "wearwise_colour_extraction",
                    "strict": True,
                    "schema": _colour_schema(),
                }
            },
        )
        log_usage(
            usage_metrics(response, model_name=self._config.model_name),
            job_type="colour_extraction",
        )
        payload = _response_json(response, "OpenAI colour extractor")

        return RawColourExtraction(
            primary_colour=_colour_from_payload(payload.get("primary_colour")),
            secondary_colour=_colour_from_payload(payload.get("secondary_colour")),
            model_family=COLOUR_EXTRACTION_MODEL_FAMILY,
            model_name=self._config.model_name,
            model_version=self._config.model_version,
            review_required=bool(payload["review_required"]),
            review_reason=_optional_string(payload.get("review_reason")),
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


class OpenAIAttributeExtractor:
    def __init__(
        self,
        *,
        config: OpenAIVisionExtractionConfig | None = None,
        responses_client: OpenAIResponsesClient | None = None,
    ) -> None:
        self._config = config or OpenAIVisionExtractionConfig()
        self._responses_client = responses_client

    def extract(self, request: AttributeExtractionRequest) -> RawAttributeExtraction:
        category_hint = request.category_code or "unknown"
        response = self._client().create(
            model=self._config.model_name,
            input=_vision_input(
                instruction=(
                    "Extract wardrobe item attributes from one background-removed garment image. "
                    "Use null or empty arrays when an attribute is not visible. "
                    "Only return a brand when a visible logo, label, "
                    "or wordmark identifies it."
                ),
                prompt=(
                    "Extract garment attributes for WearWise suggestions. "
                    f"Current category hint: {category_hint}."
                ),
                request=request,
                detail=self._config.image_detail,
                max_image_edge=self._config.max_image_edge,
            ),
            reasoning={"effort": self._config.reasoning_effort},
            max_output_tokens=self._config.max_output_tokens,
            text={
                "verbosity": self._config.verbosity,
                "format": {
                    "type": "json_schema",
                    "name": "wearwise_attribute_extraction",
                    "strict": True,
                    "schema": _attribute_schema(),
                }
            },
        )
        log_usage(
            usage_metrics(response, model_name=self._config.model_name),
            job_type="attribute_extraction",
        )
        payload = _response_json(response, "OpenAI attribute extractor")

        return RawAttributeExtraction(
            subcategory=_suggestion(payload.get("subcategory")),
            brand=_suggestion(payload.get("brand")),
            material=_suggestion(payload.get("material")),
            pattern=_suggestion(payload.get("pattern")),
            seasons=_suggestions(payload.get("seasons")),
            occasions=_suggestions(payload.get("occasions")),
            formality=_suggestion(payload.get("formality")),
            style_tags=_suggestions(payload.get("style_tags")),
            model_family=ATTRIBUTE_EXTRACTION_MODEL_FAMILY,
            model_name=self._config.model_name,
            model_version=self._config.model_version,
            review_required=bool(payload["review_required"]),
            review_reason=_optional_string(payload.get("review_reason")),
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


def _vision_input(
    *,
    instruction: str,
    prompt: str,
    request: ColourExtractionRequest | AttributeExtractionRequest,
    detail: str,
    max_image_edge: int,
) -> list[dict[str, object]]:
    return [
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": instruction}],
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": _data_url(
                        content_type=request.content_type,
                        image_bytes=request.image_bytes,
                        max_edge=max_image_edge,
                    ),
                    "detail": detail,
                },
            ],
        },
    ]


def _colour_schema() -> dict[str, object]:
    colour_value = {
        "type": ["object", "null"],
        "additionalProperties": False,
        "required": ["value", "hex", "confidence", "coverage", "source_rgb"],
        "properties": {
            "value": {"type": "string", "enum": list(COLOUR_CODES)},
            "hex": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "coverage": {"type": "number", "minimum": 0, "maximum": 1},
            "source_rgb": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "integer", "minimum": 0, "maximum": 255},
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["primary_colour", "secondary_colour", "review_required", "review_reason"],
        "properties": {
            "primary_colour": colour_value,
            "secondary_colour": colour_value,
            "review_required": {"type": "boolean"},
            "review_reason": {"type": ["string", "null"]},
        },
    }


def _attribute_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "subcategory",
            "brand",
            "material",
            "pattern",
            "seasons",
            "occasions",
            "formality",
            "style_tags",
            "review_required",
            "review_reason",
        ],
        "properties": {
            "subcategory": _suggestion_schema(SUBCATEGORY_VALUES),
            "brand": _open_brand_schema(),
            "material": _suggestion_schema(MATERIAL_VALUES),
            "pattern": _suggestion_schema(PATTERN_VALUES),
            "seasons": _suggestion_list_schema(SEASON_VALUES),
            "occasions": _suggestion_list_schema(OCCASION_VALUES),
            "formality": _suggestion_schema(FORMALITY_VALUES),
            "style_tags": _suggestion_list_schema(STYLE_TAG_VALUES),
            "review_required": {"type": "boolean"},
            "review_reason": {"type": ["string", "null"]},
        },
    }

def _open_brand_schema() -> dict[str, object]:
    schema = _suggestion_schema(())
    value = schema["properties"]["value"]
    if isinstance(value, dict):
        value.pop("enum", None)

    return schema


def _suggestion_schema(values: tuple[str, ...]) -> dict[str, object]:
    value_schema: dict[str, object] = {"type": "string"}
    if values:
        value_schema["enum"] = list(values)

    return {
        "type": ["object", "null"],
        "additionalProperties": False,
        "required": ["value", "confidence", "reason"],
        "properties": {
            "value": value_schema,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
    }


def _suggestion_list_schema(values: tuple[str, ...]) -> dict[str, object]:
    return {
        "type": "array",
        "maxItems": 4,
        "items": _suggestion_schema(values),
    }


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


def _response_json(response: Any, name: str) -> dict[str, Any]:
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or output_text.strip() == "":
        raise ModelUnavailableError(f"{name} returned an empty response.")

    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ModelUnavailableError(f"{name} returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise ModelUnavailableError(f"{name} returned a non-object payload.")

    return payload


def _colour_from_payload(value: object) -> ExtractedColour | None:
    if not isinstance(value, dict):
        return None

    source_rgb = value.get("source_rgb")
    if not isinstance(source_rgb, list) or len(source_rgb) != 3:
        source_rgb = [0, 0, 0]

    return ExtractedColour(
        code=str(value["value"]),
        hex_value=str(value["hex"]),
        coverage=_score(value["coverage"]),
        confidence=_score(value["confidence"]),
        source_rgb=tuple(_rgb_channel(channel) for channel in source_rgb),
    )


def _suggestion(value: object) -> AttributeSuggestion | None:
    if not isinstance(value, dict):
        return None

    return AttributeSuggestion(
        value=str(value["value"]),
        confidence=_score(value["confidence"]),
        reason=str(value["reason"]),
    )


def _suggestions(value: object) -> tuple[AttributeSuggestion, ...]:
    if not isinstance(value, list):
        return ()

    return tuple(suggestion for item in value if (suggestion := _suggestion(item)) is not None)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _score(value: object) -> float:
    if isinstance(value, int | float):
        return round(max(0.0, min(1.0, float(value))), 6)
    return 0.0


def _rgb_channel(value: object) -> int:
    if isinstance(value, int | float):
        return max(0, min(255, round(value)))
    return 0
