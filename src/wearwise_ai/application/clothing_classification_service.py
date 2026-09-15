from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

CLOTHING_CLASSIFICATION_VERSION = 2
CLOTHING_CLASSIFICATION_PROCESSING_VERSION = (
    f"clothing-classification-v{CLOTHING_CLASSIFICATION_VERSION}"
)
CLOTHING_CLASSIFICATION_MODEL_FAMILY = "openai"
CLOTHING_CLASSIFICATION_MODEL_NAME = "gpt-5-nano"
CLOTHING_CLASSIFICATION_MODEL_VERSION = "responses-v1"
SIGLIP_CLASSIFICATION_MODEL_FAMILY = "siglip"
SIGLIP_CLASSIFICATION_MODEL_NAME = "google/siglip-base-patch16-224"
SIGLIP_CLASSIFICATION_MODEL_VERSION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"


class CategoryClassifier(Protocol):
    def classify(self, request: ClothingClassificationRequest) -> RawCategoryPrediction: ...


@dataclass(frozen=True, slots=True)
class CategoryDefinition:
    code: str
    label: str


@dataclass(frozen=True, slots=True)
class ClothingClassificationRequest:
    asset_id: str
    image_bytes: bytes
    content_type: str


@dataclass(frozen=True, slots=True)
class RawCategoryPrediction:
    category_code: str
    confidence: float
    model_family: str
    model_name: str
    model_version: str
    alternatives: tuple[tuple[str, float], ...] = ()
    review_required: bool = False
    review_reason: str | None = None


class ConfidenceRoutedClassifier:
    def __init__(
        self,
        *,
        primary: CategoryClassifier,
        fallback: CategoryClassifier,
        min_confidence: float = 0.82,
        min_margin: float = 0.15,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._min_confidence = min_confidence
        self._min_margin = min_margin

    def classify(self, request: ClothingClassificationRequest) -> RawCategoryPrediction:
        try:
            prediction = self._primary.classify(request)
        except Exception:
            return self._fallback.classify(request)

        top_alternative_confidence = prediction.alternatives[0][1] if prediction.alternatives else 0
        margin = prediction.confidence - top_alternative_confidence
        if (
            prediction.review_required
            or prediction.confidence < self._min_confidence
            or margin < self._min_margin
        ):
            return self._fallback.classify(request)

        return prediction


@dataclass(frozen=True, slots=True)
class CategorySuggestion:
    code: str
    confidence: float
    alternatives: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class ClothingClassificationResult:
    asset_id: str
    status: str
    category: CategorySuggestion | None
    model_family: str
    model_name: str
    model_version: str
    processing_version: str
    metrics: dict[str, int | float | str]

    def to_ai_result(self) -> dict[str, object]:
        if self.category is None:
            return {
                "category": None,
                "subcategory": None,
            }

        return {
            "category": {
                "value": self.category.code,
                "confidence": self.category.confidence,
                "alternatives": [
                    {"value": code, "confidence": confidence}
                    for code, confidence in self.category.alternatives
                ],
            },
            "subcategory": None,
        }

    def to_complete_payload(self, *, worker_id: str) -> dict[str, object]:
        confidence = None
        if self.category is not None:
            confidence = {
                "category": self.category.confidence,
            }

        return {
            "worker_id": worker_id,
            "model_family": self.model_family,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "processing_version": self.processing_version,
            "status_message": (
                "clothing classification completed"
                if self.status == "ready"
                else "clothing classification needs review"
            ),
            "result": {
                **self.to_ai_result(),
                "review_required": self.status != "ready",
            },
            "confidence": confidence,
            "resource_metrics": self.metrics,
        }


@dataclass(frozen=True, slots=True)
class ClothingClassificationRule:
    min_category_confidence: float = 0.82
    min_category_margin: float = 0.15


class ClothingClassificationService:
    def __init__(
        self,
        *,
        classifier: CategoryClassifier,
        categories: tuple[CategoryDefinition, ...] = (),
        rule: ClothingClassificationRule | None = None,
    ) -> None:
        self._classifier = classifier
        self._categories = categories or DEFAULT_CATEGORY_DEFINITIONS
        self._rule = rule or ClothingClassificationRule()

    def classify(self, request: ClothingClassificationRequest) -> ClothingClassificationResult:
        started_at = perf_counter()
        raw = self._classifier.classify(request)
        allowed_codes = {category.code for category in self._categories}

        if raw.category_code not in allowed_codes:
            return ClothingClassificationResult(
                asset_id=request.asset_id,
                status="needs_review",
                category=None,
                model_family=raw.model_family,
                model_name=raw.model_name,
                model_version=raw.model_version,
                processing_version=CLOTHING_CLASSIFICATION_PROCESSING_VERSION,
                metrics={
                    "classification_duration_ms": _duration_ms(started_at),
                    "classification_status_reason": "category_not_in_taxonomy",
                    "predicted_category": raw.category_code,
                },
            )

        alternatives = tuple(
            (code, round(confidence, 6))
            for code, confidence in raw.alternatives
            if code in allowed_codes and code != raw.category_code
        )
        suggestion = CategorySuggestion(
            code=raw.category_code,
            confidence=round(max(0.0, min(1.0, raw.confidence)), 6),
            alternatives=alternatives,
        )
        top_alternative_confidence = alternatives[0][1] if alternatives else 0.0
        category_margin = suggestion.confidence - top_alternative_confidence
        status = (
            "ready"
            if (
                suggestion.confidence >= self._rule.min_category_confidence
                and category_margin >= self._rule.min_category_margin
                and not raw.review_required
            )
            else "needs_review"
        )
        status_reason = _classification_status_reason(
            raw=raw,
            confidence=suggestion.confidence,
            category_margin=category_margin,
            rule=self._rule,
        )

        return ClothingClassificationResult(
            asset_id=request.asset_id,
            status=status,
            category=suggestion,
            model_family=raw.model_family,
            model_name=raw.model_name,
            model_version=raw.model_version,
            processing_version=CLOTHING_CLASSIFICATION_PROCESSING_VERSION,
            metrics={
                "classification_duration_ms": _duration_ms(started_at),
                "category_confidence": suggestion.confidence,
                "category_margin": round(category_margin, 6),
                "category_count": len(self._categories),
                "classification_status_reason": status_reason,
            },
        )


DEFAULT_CATEGORY_DEFINITIONS: tuple[CategoryDefinition, ...] = (
    CategoryDefinition("tops", "tops"),
    CategoryDefinition("bottoms", "bottoms"),
    CategoryDefinition("shoes", "shoes"),
    CategoryDefinition("jackets", "jackets"),
    CategoryDefinition("accessories", "accessories"),
)


def _duration_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def _classification_status_reason(
    *,
    raw: RawCategoryPrediction,
    confidence: float,
    category_margin: float,
    rule: ClothingClassificationRule,
) -> str:
    if raw.review_required:
        return raw.review_reason or "model_requested_review"
    if confidence < rule.min_category_confidence:
        return "category_confidence_below_threshold"
    if category_margin < rule.min_category_margin:
        return "category_margin_below_threshold"
    return "accepted"
