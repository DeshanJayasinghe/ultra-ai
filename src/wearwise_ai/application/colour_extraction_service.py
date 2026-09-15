from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import sqrt
from time import perf_counter
from typing import Protocol

COLOUR_EXTRACTION_VERSION = 2
COLOUR_EXTRACTION_PROCESSING_VERSION = f"colour-extraction-v{COLOUR_EXTRACTION_VERSION}"
COLOUR_EXTRACTION_MODEL_FAMILY = "openai"
COLOUR_EXTRACTION_MODEL_NAME = "gpt-5-nano"
COLOUR_EXTRACTION_MODEL_VERSION = "responses-v1"
DETERMINISTIC_COLOUR_MODEL_FAMILY = "deterministic_cv"
DETERMINISTIC_COLOUR_MODEL_NAME = "alpha_dominant_colour"
DETERMINISTIC_COLOUR_MODEL_VERSION = "1.0.0"


class ColourExtractor(Protocol):
    def extract(self, request: ColourExtractionRequest) -> RawColourExtraction: ...


@dataclass(frozen=True, slots=True)
class ReferenceColour:
    code: str
    rgb: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ExtractedColour:
    code: str
    hex_value: str
    coverage: float
    confidence: float
    source_rgb: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class RawColourExtraction:
    primary_colour: ExtractedColour | None
    secondary_colour: ExtractedColour | None
    model_family: str
    model_name: str
    model_version: str
    review_required: bool = False
    review_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ColourExtractionRequest:
    asset_id: str
    image_bytes: bytes
    content_type: str


@dataclass(frozen=True, slots=True)
class ColourExtractionResult:
    asset_id: str
    status: str
    primary_colour: ExtractedColour | None
    secondary_colour: ExtractedColour | None
    model_family: str
    model_name: str
    model_version: str
    processing_version: str
    metrics: dict[str, int | float | str]

    def to_ai_result(self) -> dict[str, object]:
        return {
            "primary_colour": _colour_to_payload(self.primary_colour),
            "secondary_colour": _colour_to_payload(self.secondary_colour),
        }

    def to_complete_payload(self, *, worker_id: str) -> dict[str, object]:
        confidence = {}
        if self.primary_colour is not None:
            confidence["primary_colour"] = self.primary_colour.confidence
        if self.secondary_colour is not None:
            confidence["secondary_colour"] = self.secondary_colour.confidence

        return {
            "worker_id": worker_id,
            "model_family": self.model_family,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "processing_version": self.processing_version,
            "status_message": (
                "colour extraction completed"
                if self.status == "ready"
                else "colour extraction needs review"
            ),
            "result": {
                **self.to_ai_result(),
                "review_required": self.status != "ready",
            },
            "confidence": confidence or None,
            "resource_metrics": self.metrics,
        }


@dataclass(frozen=True, slots=True)
class ColourExtractionRule:
    alpha_threshold: int = 128
    bucket_size: int = 32
    min_visible_pixels: int = 64
    min_secondary_coverage: float = 0.12
    min_secondary_coverage_ratio_to_primary: float = 0.28


class ColourExtractionService:
    def __init__(
        self,
        *,
        extractor: ColourExtractor | None = None,
        reference_colours: tuple[ReferenceColour, ...] = (),
        rule: ColourExtractionRule | None = None,
    ) -> None:
        self._extractor = extractor
        self._reference_colours = reference_colours or DEFAULT_REFERENCE_COLOURS
        self._rule = rule or ColourExtractionRule()

    def extract(self, request: ColourExtractionRequest) -> ColourExtractionResult:
        started_at = perf_counter()
        if self._extractor is not None:
            raw = self._extractor.extract(request)
            status = (
                "needs_review" if raw.review_required or raw.primary_colour is None else "ready"
            )
            metrics: dict[str, int | float | str] = {
                "colour_extraction_duration_ms": _duration_ms(started_at),
                "colour_extraction_status_reason": raw.review_reason or status,
            }
            if raw.primary_colour is not None:
                metrics["primary_coverage"] = raw.primary_colour.coverage
            if raw.secondary_colour is not None:
                metrics["secondary_coverage"] = raw.secondary_colour.coverage

            return ColourExtractionResult(
                asset_id=request.asset_id,
                status=status,
                primary_colour=raw.primary_colour,
                secondary_colour=raw.secondary_colour,
                model_family=raw.model_family,
                model_name=raw.model_name,
                model_version=raw.model_version,
                processing_version=COLOUR_EXTRACTION_PROCESSING_VERSION,
                metrics=metrics,
            )

        visible_pixels = _visible_pixels(
            request.image_bytes,
            alpha_threshold=self._rule.alpha_threshold,
        )

        if len(visible_pixels) < self._rule.min_visible_pixels:
            return ColourExtractionResult(
                asset_id=request.asset_id,
                status="needs_review",
                primary_colour=None,
                secondary_colour=None,
                model_family=DETERMINISTIC_COLOUR_MODEL_FAMILY,
                model_name=DETERMINISTIC_COLOUR_MODEL_NAME,
                model_version=DETERMINISTIC_COLOUR_MODEL_VERSION,
                processing_version=COLOUR_EXTRACTION_PROCESSING_VERSION,
                metrics={
                    "visible_pixel_count": len(visible_pixels),
                    "colour_extraction_duration_ms": _duration_ms(started_at),
                    "colour_extraction_status_reason": "not_enough_visible_pixels",
                },
            )

        buckets = _bucket_pixels(visible_pixels, bucket_size=self._rule.bucket_size)
        ranked = sorted(buckets.items(), key=lambda item: item[1]["count"], reverse=True)
        total = len(visible_pixels)

        colours: list[ExtractedColour] = []
        for _, bucket in ranked:
            rgb = _mean_rgb(bucket)
            reference = _nearest_reference_colour(rgb, self._reference_colours)
            coverage = bucket["count"] / total
            confidence = _confidence(rgb, reference.rgb, coverage)
            candidate = ExtractedColour(
                code=reference.code,
                hex_value=_hex(rgb),
                coverage=round(coverage, 6),
                confidence=round(confidence, 6),
                source_rgb=rgb,
            )

            if all(existing.code != candidate.code for existing in colours):
                colours.append(candidate)

            if len(colours) == 2:
                break

        primary = colours[0] if colours else None
        secondary = colours[1] if len(colours) > 1 else None
        if (
            primary is not None
            and secondary is not None
            and (
                secondary.coverage < self._rule.min_secondary_coverage
                or secondary.coverage / primary.coverage
                < self._rule.min_secondary_coverage_ratio_to_primary
            )
        ):
            secondary = None

        return ColourExtractionResult(
            asset_id=request.asset_id,
            status="ready" if primary is not None else "needs_review",
            primary_colour=primary,
            secondary_colour=secondary,
            model_family=DETERMINISTIC_COLOUR_MODEL_FAMILY,
            model_name=DETERMINISTIC_COLOUR_MODEL_NAME,
            model_version=DETERMINISTIC_COLOUR_MODEL_VERSION,
            processing_version=COLOUR_EXTRACTION_PROCESSING_VERSION,
            metrics={
                "visible_pixel_count": total,
                "dominant_bucket_count": len(ranked),
                "primary_coverage": primary.coverage if primary is not None else 0,
                "secondary_coverage": secondary.coverage if secondary is not None else 0,
                "colour_extraction_duration_ms": _duration_ms(started_at),
            },
        )


DEFAULT_REFERENCE_COLOURS: tuple[ReferenceColour, ...] = (
    ReferenceColour("black", (20, 20, 20)),
    ReferenceColour("white", (242, 242, 235)),
    ReferenceColour("grey", (128, 128, 128)),
    ReferenceColour("navy", (20, 42, 84)),
    ReferenceColour("blue", (46, 103, 178)),
    ReferenceColour("green", (42, 110, 73)),
    ReferenceColour("red", (178, 47, 47)),
    ReferenceColour("pink", (215, 105, 150)),
    ReferenceColour("yellow", (222, 190, 58)),
    ReferenceColour("brown", (112, 70, 42)),
    ReferenceColour("beige", (205, 184, 145)),
    ReferenceColour("purple", (105, 72, 150)),
)


def _visible_pixels(image_bytes: bytes, *, alpha_threshold: int) -> list[tuple[int, int, int]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for colour extraction.") from exc

    with Image.open(BytesIO(image_bytes)) as image:
        rgba = image.convert("RGBA")
        return [
            (red, green, blue)
            for red, green, blue, alpha in rgba.getdata()
            if alpha >= alpha_threshold
        ]


def _bucket_pixels(
    pixels: list[tuple[int, int, int]],
    *,
    bucket_size: int,
) -> dict[tuple[int, int, int], dict[str, int]]:
    buckets: dict[tuple[int, int, int], dict[str, int]] = {}
    for red, green, blue in pixels:
        key = (
            red // bucket_size,
            green // bucket_size,
            blue // bucket_size,
        )
        bucket = buckets.setdefault(key, {"count": 0, "red": 0, "green": 0, "blue": 0})
        bucket["count"] += 1
        bucket["red"] += red
        bucket["green"] += green
        bucket["blue"] += blue

    return buckets


def _mean_rgb(bucket: dict[str, int]) -> tuple[int, int, int]:
    count = bucket["count"]
    return (
        round(bucket["red"] / count),
        round(bucket["green"] / count),
        round(bucket["blue"] / count),
    )


def _nearest_reference_colour(
    rgb: tuple[int, int, int],
    reference_colours: tuple[ReferenceColour, ...],
) -> ReferenceColour:
    return min(reference_colours, key=lambda colour: _distance(rgb, colour.rgb))


def _confidence(
    rgb: tuple[int, int, int],
    reference_rgb: tuple[int, int, int],
    coverage: float,
) -> float:
    max_distance = sqrt(3 * (255**2))
    closeness = 1 - min(1, _distance(rgb, reference_rgb) / max_distance)
    coverage_bonus = min(0.2, coverage * 0.35)
    return max(0, min(1, closeness * 0.8 + coverage_bonus))


def _distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _duration_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def _colour_to_payload(colour: ExtractedColour | None) -> dict[str, object] | None:
    if colour is None:
        return None

    return {
        "value": colour.code,
        "hex": colour.hex_value,
        "confidence": colour.confidence,
        "coverage": colour.coverage,
        "source_rgb": list(colour.source_rgb),
    }
