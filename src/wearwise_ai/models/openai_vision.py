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
    RatioSuggestion,
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

# --- v2 garment analysis vocabularies -------------------------------------
#
# Captured in the same call that already classifies the item, because the image
# tokens dominate the cost and the marginal cost of asking for more enum fields
# is a handful of output tokens. An attribute not captured here is permanently
# expensive: recovering it later means paying for a second vision pass over an
# image we have already analysed.

# Geometry. These describe where a garment sits on a body and how it stacks,
# which is what avatar compositing needs and classification fields cannot say.
GARMENT_LENGTH_VALUES = (
    "cropped",
    "hip",
    "mid-thigh",
    "knee",
    "midi",
    "ankle",
    "floor",
)
SLEEVE_LENGTH_VALUES = (
    "sleeveless",
    "cap",
    "short",
    "elbow",
    "three-quarter",
    "long",
)
NECKLINE_VALUES = (
    "crew",
    "v-neck",
    "scoop",
    "polo",
    "button-down",
    "turtleneck",
    "off-shoulder",
    "halter",
    "none",
)
FIT_VALUES = ("skinny", "slim", "regular", "relaxed", "oversized")
RISE_VALUES = ("low", "mid", "high", "none")
WAIST_POSITION_VALUES = ("natural", "dropped", "empire", "none")
CLOSURE_VALUES = ("none", "buttons", "zip", "wrap", "tie")
# Which compositing slot the garment occupies. With RISE_VALUES this is the
# highest-value pair in the v2 schema: together they turn a flat collection of
# garment images into a stack with defensible z-order and vertical alignment.
LAYER_ROLE_VALUES = ("base", "mid", "outer", "single")
TRANSPARENCY_VALUES = ("opaque", "semi-sheer", "sheer")
STRUCTURE_VALUES = ("structured", "soft", "drapey", "stiff")

# Visual signals. These improve scoring at no extra call.
PATTERN_SCALE_VALUES = ("none", "micro", "small", "medium", "large")
VISUAL_WEIGHT_VALUES = ("light", "medium", "heavy")
TEXTURE_VALUES = (
    "smooth",
    "ribbed",
    "knit",
    "woven",
    "denim",
    "leather",
    "fleece",
    "satin",
)
WATER_RESISTANCE_VALUES = ("none", "resistant", "proof")
CARE_DIFFICULTY_VALUES = ("easy", "normal", "delicate")

# Lifecycle. Donation listings need condition and flaws; capturing them now
# avoids a second paid pass over every wardrobe of every user who donates.
CONDITION_VALUES = ("like-new", "excellent", "good", "worn")
VISIBLE_FLAW_VALUES = ("stain", "tear", "missing-button", "fading", "pilling")
ESTIMATED_AGE_VALUES = ("new", "recent", "established", "old")


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
            garment_length=_suggestion(payload.get("garment_length")),
            sleeve_length=_suggestion(payload.get("sleeve_length")),
            neckline=_suggestion(payload.get("neckline")),
            fit=_suggestion(payload.get("fit")),
            rise=_suggestion(payload.get("rise")),
            waist_position=_suggestion(payload.get("waist_position")),
            closure=_suggestion(payload.get("closure")),
            layer_role=_suggestion(payload.get("layer_role")),
            transparency=_suggestion(payload.get("transparency")),
            structure=_suggestion(payload.get("structure")),
            pattern_scale=_suggestion(payload.get("pattern_scale")),
            visual_weight=_suggestion(payload.get("visual_weight")),
            texture=_suggestion(payload.get("texture")),
            formality_score=_ratio(payload.get("formality_score")),
            warmth_rating=_ratio(payload.get("warmth_rating")),
            water_resistance=_suggestion(payload.get("water_resistance")),
            care_difficulty=_suggestion(payload.get("care_difficulty")),
            condition=_suggestion(payload.get("condition")),
            visible_flaws=_suggestions(payload.get("visible_flaws")),
            estimated_age=_suggestion(payload.get("estimated_age")),
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
            garment_length=_suggestion(payload.get("garment_length")),
            sleeve_length=_suggestion(payload.get("sleeve_length")),
            neckline=_suggestion(payload.get("neckline")),
            fit=_suggestion(payload.get("fit")),
            rise=_suggestion(payload.get("rise")),
            waist_position=_suggestion(payload.get("waist_position")),
            closure=_suggestion(payload.get("closure")),
            layer_role=_suggestion(payload.get("layer_role")),
            transparency=_suggestion(payload.get("transparency")),
            structure=_suggestion(payload.get("structure")),
            pattern_scale=_suggestion(payload.get("pattern_scale")),
            visual_weight=_suggestion(payload.get("visual_weight")),
            texture=_suggestion(payload.get("texture")),
            formality_score=_ratio(payload.get("formality_score")),
            warmth_rating=_ratio(payload.get("warmth_rating")),
            water_resistance=_suggestion(payload.get("water_resistance")),
            care_difficulty=_suggestion(payload.get("care_difficulty")),
            condition=_suggestion(payload.get("condition")),
            visible_flaws=_suggestions(payload.get("visible_flaws")),
            estimated_age=_suggestion(payload.get("estimated_age")),
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
            # v2 geometry
            "garment_length",
            "sleeve_length",
            "neckline",
            "fit",
            "rise",
            "waist_position",
            "closure",
            "layer_role",
            "transparency",
            "structure",
            # v2 visual signals
            "pattern_scale",
            "visual_weight",
            "texture",
            "formality_score",
            "warmth_rating",
            "water_resistance",
            "care_difficulty",
            # v2 lifecycle
            "condition",
            "visible_flaws",
            "estimated_age",
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
            "garment_length": _suggestion_schema(GARMENT_LENGTH_VALUES),
            "sleeve_length": _suggestion_schema(SLEEVE_LENGTH_VALUES),
            "neckline": _suggestion_schema(NECKLINE_VALUES),
            "fit": _suggestion_schema(FIT_VALUES),
            "rise": _suggestion_schema(RISE_VALUES),
            "waist_position": _suggestion_schema(WAIST_POSITION_VALUES),
            "closure": _suggestion_schema(CLOSURE_VALUES),
            "layer_role": _suggestion_schema(LAYER_ROLE_VALUES),
            "transparency": _suggestion_schema(TRANSPARENCY_VALUES),
            "structure": _suggestion_schema(STRUCTURE_VALUES),
            "pattern_scale": _suggestion_schema(PATTERN_SCALE_VALUES),
            "visual_weight": _suggestion_schema(VISUAL_WEIGHT_VALUES),
            "texture": _suggestion_schema(TEXTURE_VALUES),
            "formality_score": _ratio_schema(),
            "warmth_rating": _ratio_schema(),
            "water_resistance": _suggestion_schema(WATER_RESISTANCE_VALUES),
            "care_difficulty": _suggestion_schema(CARE_DIFFICULTY_VALUES),
            "condition": _suggestion_schema(CONDITION_VALUES),
            "visible_flaws": _suggestion_list_schema(VISIBLE_FLAW_VALUES),
            "estimated_age": _suggestion_schema(ESTIMATED_AGE_VALUES),
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


def _ratio_schema() -> dict[str, object]:
    """A continuous 0..1 signal, for fields where a discrete enum loses too much.

    formality_score and warmth_rating feed scoring directly: a thin linen shirt
    and a heavy flannel are both "shirt" in "cotton", but only a continuous
    warmth value separates them for weather fit.
    """
    return {
        "type": ["object", "null"],
        "additionalProperties": False,
        "required": ["value", "confidence", "reason"],
        "properties": {
            "value": {"type": "number", "minimum": 0, "maximum": 1},
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


def _ratio(value: object) -> RatioSuggestion | None:
    if not isinstance(value, dict):
        return None

    return RatioSuggestion(
        value=_score(value["value"]),
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
