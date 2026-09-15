from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

from wearwise_ai.application.attribute_extraction_service import (
    ATTRIBUTE_EXTRACTION_MODEL_FAMILY,
    AttributeExtractionRequest,
    RawAttributeExtraction,
)
from wearwise_ai.application.clothing_classification_service import (
    CLOTHING_CLASSIFICATION_MODEL_FAMILY,
    CLOTHING_CLASSIFICATION_MODEL_NAME,
    CLOTHING_CLASSIFICATION_MODEL_VERSION,
    ClothingClassificationRequest,
    RawCategoryPrediction,
)
from wearwise_ai.models.background_removal.birefnet_adapter import ModelUnavailableError
from wearwise_ai.models.openai_usage import log_usage, usage_metrics
from wearwise_ai.models.classification.openai_adapter import (
    OPENAI_CLASSIFICATION_RESPONSE_SCHEMA,
)
from wearwise_ai.models.openai_vision import (
    _attribute_schema,
    _ratio,
    _data_url,
    _optional_string,
    _response_json,
    _score,
    _suggestion,
    _suggestions,
)

GARMENT_ANALYSIS_PROCESSING_VERSION = "garment-analysis-v2"


class OpenAIResponsesClient(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class GarmentAnalysisRequest:
    asset_id: str
    image_bytes: bytes
    content_type: str
    category_hint: str | None = None


@dataclass(frozen=True, slots=True)
class RawGarmentAnalysis:
    category: RawCategoryPrediction
    attributes: RawAttributeExtraction
    cache_status: str


@dataclass(frozen=True, slots=True)
class OpenAIGarmentAnalysisConfig:
    model_name: str = CLOTHING_CLASSIFICATION_MODEL_NAME
    model_version: str = CLOTHING_CLASSIFICATION_MODEL_VERSION
    image_detail: str = "low"
    reasoning_effort: str = "minimal"
    max_output_tokens: int = 1600
    verbosity: str = "low"
    max_image_edge: int = 512
    processing_version: str = GARMENT_ANALYSIS_PROCESSING_VERSION


class InMemoryGarmentAnalysisCache:
    def __init__(self) -> None:
        self._items: dict[str, RawGarmentAnalysis] = {}
        self._lock = Lock()

    def get(self, key: str) -> RawGarmentAnalysis | None:
        with self._lock:
            return self._items.get(key)

    def set(self, key: str, value: RawGarmentAnalysis) -> None:
        with self._lock:
            self._items[key] = value


class OpenAIGarmentAnalyser:
    def __init__(
        self,
        *,
        config: OpenAIGarmentAnalysisConfig | None = None,
        responses_client: OpenAIResponsesClient | None = None,
        cache: InMemoryGarmentAnalysisCache | None = None,
    ) -> None:
        self._config = config or OpenAIGarmentAnalysisConfig()
        self._responses_client = responses_client
        self._cache = cache or InMemoryGarmentAnalysisCache()

    def analyse(self, request: GarmentAnalysisRequest) -> RawGarmentAnalysis:
        cache_key = _cache_key(request=request, config=self._config)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return RawGarmentAnalysis(
                category=cached.category,
                attributes=cached.attributes,
                cache_status="hit",
            )

        category_hint = request.category_hint or "unknown"
        response = self._client().create(
            model=self._config.model_name,
            input=[
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Analyse one background-removed wardrobe item for WearWise. "
                                "Return the category and visible garment attributes using only "
                                "the allowed schema values. Use unknown for category when evidence "
                                "is insufficient. Only return a brand when a visible logo, label, "
                                "or wordmark identifies it. "
                                "Also report the garment geometry, because it drives how the item "
                                "is laid out on a body: garment_length and sleeve_length as they "
                                "would fall when worn, neckline, fit, rise for bottoms, "
                                "waist_position, closure, and layer_role saying whether this is "
                                "worn against the skin (base), over a base (mid), on top of an "
                                "outfit (outer), or alone (single). "
                                "Report warmth_rating as how warm the garment actually is to wear "
                                "from the visible fabric weight and construction, not from its "
                                "category: a thin linen shirt and a heavy flannel shirt differ. "
                                "Report formality_score on the same continuous scale. "
                                "Report condition and any visible_flaws honestly from the image. "
                                "Return null for any field the image does not show."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Analyse this garment. "
                                f"Current category hint: {category_hint}."
                            ),
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
                    "name": "wearwise_garment_analysis",
                    "strict": True,
                    "schema": _garment_schema(),
                },
            },
        )
        metrics = usage_metrics(response, model_name=self._config.model_name)
        log_usage(metrics, job_type="garment_analysis")
        payload = _response_json(response, "OpenAI garment analyser")
        analysis = _analysis_from_payload(payload, config=self._config, cache_status="miss")
        self._cache.set(cache_key, analysis)
        return analysis

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


class GarmentAnalysisCategoryClassifier:
    def __init__(self, *, analyser: OpenAIGarmentAnalyser) -> None:
        self._analyser = analyser

    def classify(self, request: ClothingClassificationRequest) -> RawCategoryPrediction:
        analysis = self._analyser.analyse(
            GarmentAnalysisRequest(
                asset_id=request.asset_id,
                image_bytes=request.image_bytes,
                content_type=request.content_type,
            )
        )
        return analysis.category


class GarmentAnalysisAttributeExtractor:
    def __init__(self, *, analyser: OpenAIGarmentAnalyser) -> None:
        self._analyser = analyser

    def extract(self, request: AttributeExtractionRequest) -> RawAttributeExtraction:
        analysis = self._analyser.analyse(
            GarmentAnalysisRequest(
                asset_id=request.asset_id,
                image_bytes=request.image_bytes,
                content_type=request.content_type,
                category_hint=request.category_code,
            )
        )
        return analysis.attributes


def _garment_schema() -> dict[str, object]:
    classification_properties = OPENAI_CLASSIFICATION_RESPONSE_SCHEMA["properties"]
    attribute_schema = _attribute_schema()
    attribute_properties = attribute_schema["properties"]
    attribute_required = [
        item
        for item in attribute_schema["required"]
        if item not in {"review_required", "review_reason"}
    ]

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "category",
            "confidence",
            "alternatives",
            "visible_evidence",
            *attribute_required,
            "review_required",
            "review_reason",
        ],
        "properties": {
            "category": classification_properties["category"],
            "confidence": classification_properties["confidence"],
            "alternatives": classification_properties["alternatives"],
            "visible_evidence": classification_properties["visible_evidence"],
            **{
                key: value
                for key, value in attribute_properties.items()
                if key not in {"review_required", "review_reason"}
            },
            "review_required": {"type": "boolean"},
            "review_reason": {"type": ["string", "null"]},
        },
    }


def _analysis_from_payload(
    payload: dict[str, Any],
    *,
    config: OpenAIGarmentAnalysisConfig,
    cache_status: str,
) -> RawGarmentAnalysis:
    category = str(payload["category"])
    category_prediction = RawCategoryPrediction(
        category_code=category,
        confidence=_score(payload["confidence"]),
        model_family=CLOTHING_CLASSIFICATION_MODEL_FAMILY,
        model_name=config.model_name,
        model_version=config.model_version,
        alternatives=_alternatives(payload.get("alternatives", [])),
        review_required=bool(payload["review_required"]) or category == "unknown",
        review_reason=_optional_string(payload.get("review_reason")),
    )
    attributes = RawAttributeExtraction(
        subcategory=_suggestion(payload.get("subcategory")),
        brand=_suggestion(payload.get("brand")),
        material=_suggestion(payload.get("material")),
        pattern=_suggestion(payload.get("pattern")),
        seasons=_suggestions(payload.get("seasons")),
        occasions=_suggestions(payload.get("occasions")),
        formality=_suggestion(payload.get("formality")),
        style_tags=_suggestions(payload.get("style_tags")),
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
        model_family=ATTRIBUTE_EXTRACTION_MODEL_FAMILY,
        model_name=config.model_name,
        model_version=config.model_version,
        review_required=bool(payload["review_required"]),
        review_reason=_optional_string(payload.get("review_reason")),
    )
    return RawGarmentAnalysis(
        category=category_prediction,
        attributes=attributes,
        cache_status=cache_status,
    )


def _alternatives(value: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, list):
        return ()

    alternatives: list[tuple[str, float]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        if not isinstance(category, str) or category in seen or category == "unknown":
            continue
        alternatives.append((category, _score(item.get("confidence", 0))))
        seen.add(category)

    return tuple(alternatives)


def _cache_key(*, request: GarmentAnalysisRequest, config: OpenAIGarmentAnalysisConfig) -> str:
    digest = hashlib.sha256(request.image_bytes).hexdigest()
    payload = {
        "digest": digest,
        "processing_version": config.processing_version,
        "model_name": config.model_name,
        "model_version": config.model_version,
        "image_detail": config.image_detail,
        "max_image_edge": config.max_image_edge,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
