from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from time import perf_counter
from typing import Protocol

ATTRIBUTE_EXTRACTION_VERSION = 2
ATTRIBUTE_EXTRACTION_PROCESSING_VERSION = f"attribute-extraction-v{ATTRIBUTE_EXTRACTION_VERSION}"
ATTRIBUTE_EXTRACTION_MODEL_FAMILY = "openai"
ATTRIBUTE_EXTRACTION_MODEL_NAME = "gpt-5-nano"
ATTRIBUTE_EXTRACTION_MODEL_VERSION = "responses-v1"
DETERMINISTIC_ATTRIBUTE_MODEL_FAMILY = "deterministic_cv"
DETERMINISTIC_ATTRIBUTE_MODEL_NAME = "mask_geometry_attributes"
DETERMINISTIC_ATTRIBUTE_MODEL_VERSION = "1.0.0"


class AttributeExtractor(Protocol):
    def extract(self, request: AttributeExtractionRequest) -> RawAttributeExtraction: ...


@dataclass(frozen=True, slots=True)
class AttributeExtractionRequest:
    asset_id: str
    image_bytes: bytes
    content_type: str
    category_code: str | None = None


@dataclass(frozen=True, slots=True)
class AttributeSuggestion:
    value: str
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class RatioSuggestion:
    """A continuous 0..1 signal. formality_score and warmth_rating use this
    because a discrete enum loses information the scoring engine needs."""

    value: float
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class RawAttributeExtraction:
    subcategory: AttributeSuggestion | None
    brand: AttributeSuggestion | None
    material: AttributeSuggestion | None
    pattern: AttributeSuggestion | None
    seasons: tuple[AttributeSuggestion, ...]
    occasions: tuple[AttributeSuggestion, ...]
    formality: AttributeSuggestion | None
    style_tags: tuple[AttributeSuggestion, ...]
    model_family: str
    model_name: str
    model_version: str
    review_required: bool = False
    review_reason: str | None = None
    # --- v2 fields. All optional: items analysed under garment-analysis-v1 have
    # none of these, and every consumer must treat absence as "not recorded"
    # rather than as a zero or a default.
    garment_length: AttributeSuggestion | None = None
    sleeve_length: AttributeSuggestion | None = None
    neckline: AttributeSuggestion | None = None
    fit: AttributeSuggestion | None = None
    rise: AttributeSuggestion | None = None
    waist_position: AttributeSuggestion | None = None
    closure: AttributeSuggestion | None = None
    layer_role: AttributeSuggestion | None = None
    transparency: AttributeSuggestion | None = None
    structure: AttributeSuggestion | None = None
    pattern_scale: AttributeSuggestion | None = None
    visual_weight: AttributeSuggestion | None = None
    texture: AttributeSuggestion | None = None
    formality_score: RatioSuggestion | None = None
    warmth_rating: RatioSuggestion | None = None
    water_resistance: AttributeSuggestion | None = None
    care_difficulty: AttributeSuggestion | None = None
    condition: AttributeSuggestion | None = None
    visible_flaws: tuple[AttributeSuggestion, ...] = ()
    estimated_age: AttributeSuggestion | None = None


@dataclass(frozen=True, slots=True)
class AttributeExtractionResult:
    asset_id: str
    status: str
    subcategory: AttributeSuggestion | None
    brand: AttributeSuggestion | None
    material: AttributeSuggestion | None
    pattern: AttributeSuggestion | None
    seasons: tuple[AttributeSuggestion, ...]
    occasions: tuple[AttributeSuggestion, ...]
    formality: AttributeSuggestion | None
    style_tags: tuple[AttributeSuggestion, ...]
    model_family: str
    model_name: str
    model_version: str
    processing_version: str
    metrics: dict[str, int | float | str]
    garment_length: AttributeSuggestion | None = None
    sleeve_length: AttributeSuggestion | None = None
    neckline: AttributeSuggestion | None = None
    fit: AttributeSuggestion | None = None
    rise: AttributeSuggestion | None = None
    waist_position: AttributeSuggestion | None = None
    closure: AttributeSuggestion | None = None
    layer_role: AttributeSuggestion | None = None
    transparency: AttributeSuggestion | None = None
    structure: AttributeSuggestion | None = None
    pattern_scale: AttributeSuggestion | None = None
    visual_weight: AttributeSuggestion | None = None
    texture: AttributeSuggestion | None = None
    formality_score: RatioSuggestion | None = None
    warmth_rating: RatioSuggestion | None = None
    water_resistance: AttributeSuggestion | None = None
    care_difficulty: AttributeSuggestion | None = None
    condition: AttributeSuggestion | None = None
    visible_flaws: tuple[AttributeSuggestion, ...] = ()
    estimated_age: AttributeSuggestion | None = None

    def to_ai_result(self) -> dict[str, object]:
        return {
            "subcategory": _attribute_to_payload(self.subcategory),
            "brand": _attribute_to_payload(self.brand),
            "material": _attribute_to_payload(self.material),
            "pattern": _attribute_to_payload(self.pattern),
            "seasons": [_attribute_to_payload(item) for item in self.seasons],
            "occasions": [_attribute_to_payload(item) for item in self.occasions],
            "formality": _attribute_to_payload(self.formality),
            "style_tags": [_attribute_to_payload(item) for item in self.style_tags],
            "garment_length": _attribute_to_payload(self.garment_length),
            "sleeve_length": _attribute_to_payload(self.sleeve_length),
            "neckline": _attribute_to_payload(self.neckline),
            "fit": _attribute_to_payload(self.fit),
            "rise": _attribute_to_payload(self.rise),
            "waist_position": _attribute_to_payload(self.waist_position),
            "closure": _attribute_to_payload(self.closure),
            "layer_role": _attribute_to_payload(self.layer_role),
            "transparency": _attribute_to_payload(self.transparency),
            "structure": _attribute_to_payload(self.structure),
            "pattern_scale": _attribute_to_payload(self.pattern_scale),
            "visual_weight": _attribute_to_payload(self.visual_weight),
            "texture": _attribute_to_payload(self.texture),
            "formality_score": _attribute_to_payload(self.formality_score),
            "warmth_rating": _attribute_to_payload(self.warmth_rating),
            "water_resistance": _attribute_to_payload(self.water_resistance),
            "care_difficulty": _attribute_to_payload(self.care_difficulty),
            "condition": _attribute_to_payload(self.condition),
            "visible_flaws": [_attribute_to_payload(item) for item in self.visible_flaws],
            "estimated_age": _attribute_to_payload(self.estimated_age),
        }

    def to_complete_payload(self, *, worker_id: str) -> dict[str, object]:
        confidence = {}
        for name, suggestion in (
            ("subcategory", self.subcategory),
            ("brand", self.brand),
            ("material", self.material),
            ("pattern", self.pattern),
            ("formality", self.formality),
        ):
            if suggestion is not None:
                confidence[name] = suggestion.confidence

        return {
            "worker_id": worker_id,
            "model_family": self.model_family,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "processing_version": self.processing_version,
            "status_message": (
                "attribute extraction completed"
                if self.status == "ready"
                else "attribute extraction needs review"
            ),
            "result": {
                **self.to_ai_result(),
                "review_required": self.status != "ready",
            },
            "confidence": confidence or None,
            "resource_metrics": self.metrics,
        }


@dataclass(frozen=True, slots=True)
class AttributeExtractionRule:
    alpha_threshold: int = 128
    min_visible_pixels: int = 64
    min_confidence: float = 0.6


class AttributeExtractionService:
    def __init__(
        self,
        *,
        extractor: AttributeExtractor | None = None,
        rule: AttributeExtractionRule | None = None,
    ) -> None:
        self._extractor = extractor
        self._rule = rule or AttributeExtractionRule()

    def extract(self, request: AttributeExtractionRequest) -> AttributeExtractionResult:
        started_at = perf_counter()
        if self._extractor is not None:
            raw = self._extractor.extract(request)
            suggestions = [
                suggestion
                for suggestion in (
                    raw.subcategory,
                    raw.brand,
                    raw.material,
                    raw.pattern,
                    raw.formality,
                    *raw.seasons,
                    *raw.occasions,
                    *raw.style_tags,
                )
                if suggestion is not None
            ]
            status = (
                "ready"
                if suggestions
                and all(
                    suggestion.confidence >= self._rule.min_confidence for suggestion in suggestions
                )
                and not raw.review_required
                else "needs_review"
            )

            return AttributeExtractionResult(
                asset_id=request.asset_id,
                status=status,
                subcategory=raw.subcategory,
                brand=raw.brand,
                material=raw.material,
                pattern=raw.pattern,
                seasons=raw.seasons,
                occasions=raw.occasions,
                formality=raw.formality,
                style_tags=raw.style_tags,
                model_family=raw.model_family,
                model_name=raw.model_name,
                model_version=raw.model_version,
                processing_version=ATTRIBUTE_EXTRACTION_PROCESSING_VERSION,
                metrics={
                    "attribute_extraction_duration_ms": _duration_ms(started_at),
                    "attribute_extraction_status_reason": raw.review_reason or status,
                    "category_code": request.category_code or "unknown",
                    "suggestion_count": len(suggestions),
                },
            )

        geometry = _visible_geometry(
            request.image_bytes,
            alpha_threshold=self._rule.alpha_threshold,
        )
        if geometry.visible_pixel_count < self._rule.min_visible_pixels:
            return AttributeExtractionResult(
                asset_id=request.asset_id,
                status="needs_review",
                subcategory=None,
                brand=None,
                material=None,
                pattern=None,
                seasons=(),
                occasions=(),
                formality=None,
                style_tags=(),
                model_family=DETERMINISTIC_ATTRIBUTE_MODEL_FAMILY,
                model_name=DETERMINISTIC_ATTRIBUTE_MODEL_NAME,
                model_version=DETERMINISTIC_ATTRIBUTE_MODEL_VERSION,
                processing_version=ATTRIBUTE_EXTRACTION_PROCESSING_VERSION,
                metrics={
                    **geometry.to_metrics(),
                    "attribute_extraction_duration_ms": _duration_ms(started_at),
                    "attribute_extraction_status_reason": "not_enough_visible_pixels",
                },
            )

        category = request.category_code
        sleeve_length = _sleeve_length(geometry, category)
        subcategory = _subcategory(category, sleeve_length)
        material = _material(category)
        pattern = _pattern(geometry)
        seasons = _seasons(category, material)
        occasions = _occasions(category)
        formality = _formality(category)
        style_tags = _style_tags(category, pattern, formality)
        suggestions = [
            suggestion
            for suggestion in (
                subcategory,
                material,
                pattern,
                formality,
                *seasons,
                *occasions,
                *style_tags,
            )
            if suggestion is not None
        ]
        status = (
            "ready"
            if suggestions
            and all(
                suggestion.confidence >= self._rule.min_confidence for suggestion in suggestions
            )
            else "needs_review"
        )

        return AttributeExtractionResult(
            asset_id=request.asset_id,
            status=status,
            subcategory=subcategory,
            brand=None,
            material=material,
            pattern=pattern,
            seasons=seasons,
            occasions=occasions,
            formality=formality,
            style_tags=style_tags,
            model_family=DETERMINISTIC_ATTRIBUTE_MODEL_FAMILY,
            model_name=DETERMINISTIC_ATTRIBUTE_MODEL_NAME,
            model_version=DETERMINISTIC_ATTRIBUTE_MODEL_VERSION,
            processing_version=ATTRIBUTE_EXTRACTION_PROCESSING_VERSION,
            metrics={
                **geometry.to_metrics(),
                "attribute_extraction_duration_ms": _duration_ms(started_at),
                "category_code": category or "unknown",
            },
        )


@dataclass(frozen=True, slots=True)
class VisibleGeometry:
    image_width: int
    image_height: int
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    visible_pixel_count: int
    upper_outer_pixel_ratio: float
    lower_pixel_ratio: float
    visual_complexity_ratio: float

    @property
    def garment_width(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def garment_height(self) -> int:
        return self.max_y - self.min_y + 1

    @property
    def height_ratio(self) -> float:
        return self.garment_height / self.image_height

    def to_metrics(self) -> dict[str, int | float]:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "visible_pixel_count": self.visible_pixel_count,
            "garment_width": self.garment_width,
            "garment_height": self.garment_height,
            "garment_height_ratio": round(self.height_ratio, 6),
            "upper_outer_pixel_ratio": round(self.upper_outer_pixel_ratio, 6),
            "lower_pixel_ratio": round(self.lower_pixel_ratio, 6),
            "visual_complexity_ratio": round(self.visual_complexity_ratio, 6),
        }


def _visible_geometry(image_bytes: bytes, *, alpha_threshold: int) -> VisibleGeometry:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for attribute extraction.") from exc

    with Image.open(BytesIO(image_bytes)) as image:
        rgba = image.convert("RGBA")
        width, height = rgba.size
        pixels = list(rgba.getdata())

    visible: list[tuple[int, int]] = []
    for index, (_, _, _, alpha) in enumerate(pixels):
        if alpha >= alpha_threshold:
            visible.append((index % width, index // width))

    if not visible:
        return VisibleGeometry(width, height, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0)

    xs = [x for x, _ in visible]
    ys = [y for _, y in visible]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    garment_width = max_x - min_x + 1
    garment_height = max_y - min_y + 1
    upper_limit = min_y + round(garment_height * 0.42)
    lower_limit = min_y + round(garment_height * 0.72)
    left_outer = min_x + round(garment_width * 0.25)
    right_outer = max_x - round(garment_width * 0.25)
    upper_outer = [
        (x, y) for x, y in visible if y <= upper_limit and (x <= left_outer or x >= right_outer)
    ]
    lower = [(x, y) for x, y in visible if y >= lower_limit]
    visible_colours = [
        pixels[(y * width) + x][:3] for x, y in visible[:: max(1, len(visible) // 4000)]
    ]
    quantized_colours = {
        (red // 48, green // 48, blue // 48) for red, green, blue in visible_colours
    }

    return VisibleGeometry(
        image_width=width,
        image_height=height,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        visible_pixel_count=len(visible),
        upper_outer_pixel_ratio=len(upper_outer) / len(visible),
        lower_pixel_ratio=len(lower) / len(visible),
        visual_complexity_ratio=len(quantized_colours) / max(1, len(visible_colours)),
    )


def _sleeve_length(geometry: VisibleGeometry, category: str | None) -> AttributeSuggestion | None:
    if category not in {"tops", "jackets"}:
        return None

    ratio = geometry.upper_outer_pixel_ratio
    if ratio >= 0.34:
        return AttributeSuggestion("long", min(0.9, 0.58 + ratio), "wide_upper_outer_mask")
    if ratio >= 0.2:
        return AttributeSuggestion("short", min(0.82, 0.5 + ratio), "moderate_upper_outer_mask")
    return AttributeSuggestion(
        "sleeveless",
        min(0.78, 0.56 + (0.2 - ratio)),
        "minimal_upper_outer_mask",
    )


def _clothing_length(geometry: VisibleGeometry, category: str | None) -> AttributeSuggestion | None:
    if category in {"shoes", "accessories"}:
        return None

    ratio = geometry.height_ratio
    lower_ratio = geometry.lower_pixel_ratio
    if category == "bottoms":
        if ratio >= 0.78:
            return AttributeSuggestion("full_length", 0.82, "tall_bottom_garment_mask")
        return AttributeSuggestion("short", 0.68, "short_bottom_garment_mask")

    if ratio >= 0.78 and lower_ratio >= 0.18:
        return AttributeSuggestion("long", 0.74, "tall_upper_garment_mask")
    if ratio >= 0.54:
        return AttributeSuggestion("regular", 0.7, "regular_upper_garment_mask")
    return AttributeSuggestion("cropped", 0.66, "short_upper_garment_mask")


def _subcategory(
    category: str | None,
    sleeve_length: AttributeSuggestion | None,
) -> AttributeSuggestion | None:
    if category == "tops":
        if sleeve_length is not None and sleeve_length.value in {"long", "short"}:
            return AttributeSuggestion("shirt", 0.58, "top_shape_with_sleeves")
        return AttributeSuggestion("t-shirt", 0.56, "top_shape_default")
    if category == "bottoms":
        return AttributeSuggestion("trousers", 0.55, "bottom_shape_default")
    if category == "jackets":
        return AttributeSuggestion("jacket", 0.58, "jacket_category_default")
    if category == "shoes":
        return AttributeSuggestion("trainers", 0.54, "shoe_category_default")
    if category == "accessories":
        return AttributeSuggestion("bag", 0.5, "accessory_category_default")
    return None


def _material(category: str | None) -> AttributeSuggestion | None:
    if category == "bottoms":
        return AttributeSuggestion("denim", 0.55, "bottom_material_prior")
    if category == "shoes":
        return AttributeSuggestion("leather", 0.54, "shoe_material_prior")
    if category == "jackets":
        return AttributeSuggestion("wool", 0.5, "jacket_material_prior")
    if category in {"tops", "accessories"}:
        return AttributeSuggestion("cotton", 0.52, "category_material_prior")
    return None


def _pattern(geometry: VisibleGeometry) -> AttributeSuggestion:
    if geometry.visual_complexity_ratio >= 0.018:
        return AttributeSuggestion("graphic", 0.62, "high_visible_colour_complexity")
    return AttributeSuggestion("solid", 0.64, "low_visible_colour_complexity")


def _seasons(
    category: str | None,
    material: AttributeSuggestion | None,
) -> tuple[AttributeSuggestion, ...]:
    material_value = material.value if material is not None else None
    if category == "jackets" or material_value in {"wool", "leather"}:
        return (
            AttributeSuggestion("autumn", 0.58, "warm_layer_prior"),
            AttributeSuggestion("winter", 0.58, "warm_layer_prior"),
        )
    if material is not None and material.value == "linen":
        return (
            AttributeSuggestion("spring", 0.58, "light_material_prior"),
            AttributeSuggestion("summer", 0.58, "light_material_prior"),
        )
    return (AttributeSuggestion("all-season", 0.6, "general_wear_prior"),)


def _occasions(category: str | None) -> tuple[AttributeSuggestion, ...]:
    if category == "shoes":
        return (AttributeSuggestion("daily", 0.56, "shoe_daily_prior"),)
    return (
        AttributeSuggestion("daily", 0.6, "casual_item_prior"),
        AttributeSuggestion("travel", 0.52, "general_utility_prior"),
    )


def _formality(category: str | None) -> AttributeSuggestion:
    if category == "jackets":
        return AttributeSuggestion("smart-casual", 0.56, "jacket_formality_prior")
    return AttributeSuggestion("casual", 0.62, "default_formality_prior")


def _style_tags(
    category: str | None,
    pattern: AttributeSuggestion,
    formality: AttributeSuggestion,
) -> tuple[AttributeSuggestion, ...]:
    tags = [AttributeSuggestion(formality.value, formality.confidence, formality.reason)]
    if pattern.value == "graphic":
        tags.append(AttributeSuggestion("streetwear", 0.55, "graphic_pattern_prior"))
    if category == "shoes":
        tags.append(AttributeSuggestion("sporty", 0.52, "shoe_style_prior"))
    return tuple(tags)


def _attribute_to_payload(attribute: AttributeSuggestion | None) -> dict[str, object] | None:
    if attribute is None:
        return None

    return {
        "value": attribute.value,
        "confidence": attribute.confidence,
        "reason": attribute.reason,
    }


def _duration_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)
