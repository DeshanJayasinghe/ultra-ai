from __future__ import annotations

from dataclasses import dataclass
import math

from wearwise_ai.preprocessing.image_decoder import ImageProbe
from wearwise_ai.preprocessing.orientation import OrientationTransform, resolve_orientation_transform


@dataclass(frozen=True, slots=True)
class CanvasNormalizationRule:
    normalization_version: int = 2
    target_long_edge: int = 2048
    garment_fill_ratio: float = 0.85
    output_format: str = "webp"
    preserve_alpha: bool = True
    generate_background_removed_variant: bool = True


@dataclass(frozen=True, slots=True)
class NormalizationPlan:
    normalization_version: int
    source_width: int
    source_height: int
    normalized_width: int
    normalized_height: int
    scale_factor: float
    output_format: str
    preserve_alpha: bool
    generate_background_removed_variant: bool
    orientation: OrientationTransform
    target_canvas_width: int
    target_canvas_height: int
    estimated_garment_fill_ratio: float


def plan_normalization(
    probe: ImageProbe,
    *,
    exif_orientation: int | None = None,
    rule: CanvasNormalizationRule | None = None,
) -> NormalizationPlan:
    rules = rule or CanvasNormalizationRule()
    orientation = resolve_orientation_transform(exif_orientation)

    source_width = probe.height if orientation.swap_dimensions else probe.width
    source_height = probe.width if orientation.swap_dimensions else probe.height

    long_edge = max(source_width, source_height)
    scale_factor = min(1.0, rules.target_long_edge / long_edge)

    normalized_width = max(1, math.floor(source_width * scale_factor))
    normalized_height = max(1, math.floor(source_height * scale_factor))

    if normalized_width >= normalized_height:
        target_canvas_width = rules.target_long_edge
        target_canvas_height = max(1, math.floor(rules.target_long_edge * (normalized_height / normalized_width)))
    else:
        target_canvas_height = rules.target_long_edge
        target_canvas_width = max(1, math.floor(rules.target_long_edge * (normalized_width / normalized_height)))

    return NormalizationPlan(
        normalization_version=rules.normalization_version,
        source_width=source_width,
        source_height=source_height,
        normalized_width=normalized_width,
        normalized_height=normalized_height,
        scale_factor=scale_factor,
        output_format=rules.output_format,
        preserve_alpha=rules.preserve_alpha,
        generate_background_removed_variant=rules.generate_background_removed_variant,
        orientation=orientation,
        target_canvas_width=target_canvas_width,
        target_canvas_height=target_canvas_height,
        estimated_garment_fill_ratio=rules.garment_fill_ratio,
    )
