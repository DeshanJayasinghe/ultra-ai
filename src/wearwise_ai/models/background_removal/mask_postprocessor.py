from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from time import perf_counter
from typing import Protocol

from wearwise_ai.application.background_removal_service import (
    BackgroundRemovalOutput,
    BackgroundRemovalRequest,
)
from wearwise_ai.models.background_removal.birefnet_adapter import ModelUnavailableError


@dataclass(frozen=True, slots=True)
class MaskPostprocessingConfig:
    output_content_type: str = "image/webp"
    output_quality: int = 92
    alpha_threshold: int = 12


class MaskPostprocessor(Protocol):
    def build_cutout(
        self,
        *,
        request: BackgroundRemovalRequest,
        mask_bytes: bytes,
        confidence: float,
        metrics: dict[str, int | float | str] | None = None,
    ) -> BackgroundRemovalOutput: ...


class PillowAlphaMaskPostprocessor:
    """Composites a segmentation mask into an alpha-channel WebP garment cutout."""

    def __init__(self, *, config: MaskPostprocessingConfig | None = None) -> None:
        self._config = config or MaskPostprocessingConfig()

    def build_cutout(
        self,
        *,
        request: BackgroundRemovalRequest,
        mask_bytes: bytes,
        confidence: float,
        metrics: dict[str, int | float | str] | None = None,
    ) -> BackgroundRemovalOutput:
        started_at = perf_counter()
        try:
            from PIL import Image
        except ImportError as exc:
            raise ModelUnavailableError(
                "Pillow is required for background-removal mask post-processing."
            ) from exc

        with Image.open(BytesIO(request.image_bytes)) as source_image:
            source = source_image.convert("RGBA")

        with Image.open(BytesIO(mask_bytes)) as mask_image:
            mask = mask_image.convert("L")

        if mask.size != source.size:
            mask = mask.resize(source.size)

        if self._config.alpha_threshold > 0:
            mask = mask.point(lambda pixel: 0 if pixel < self._config.alpha_threshold else pixel)

        mask_metrics = _measure_mask(mask)
        source.putalpha(mask)

        output_buffer = BytesIO()
        output_format = _output_format_for_content_type(self._config.output_content_type)
        source.save(output_buffer, format=output_format, quality=self._config.output_quality)
        image_bytes = output_buffer.getvalue()

        merged_metrics: dict[str, int | float | str] = {
            "postprocessing_duration_ms": round((perf_counter() - started_at) * 1000, 3),
            "output_width": source.width,
            "output_height": source.height,
            **mask_metrics,
        }
        if metrics:
            merged_metrics.update(metrics)

        return BackgroundRemovalOutput(
            image_bytes=image_bytes,
            content_type=self._config.output_content_type,
            confidence=confidence,
            width=source.width,
            height=source.height,
            metrics=merged_metrics,
        )


def _output_format_for_content_type(content_type: str) -> str:
    match content_type:
        case "image/webp":
            return "WEBP"
        case "image/png":
            return "PNG"
        case _:
            raise ValueError(f"Unsupported cutout content type: {content_type!r}.")


def _measure_mask(mask: object) -> dict[str, int | float]:
    width, height = mask.size
    total_pixels = width * height
    alpha = list(mask.getdata())
    foreground_pixels = sum(1 for pixel in alpha if pixel >= 128)
    transparent_pixels = sum(1 for pixel in alpha if pixel <= 10)

    top = [mask.getpixel((x, 0)) for x in range(width)]
    bottom = [mask.getpixel((x, height - 1)) for x in range(width)]
    left = [mask.getpixel((0, y)) for y in range(height)]
    right = [mask.getpixel((width - 1, y)) for y in range(height)]
    edge_pixels = top + bottom + left + right
    edge_foreground_pixels = sum(1 for pixel in edge_pixels if pixel >= 128)

    return {
        "foreground_coverage_ratio": round(foreground_pixels / total_pixels, 6),
        "transparent_coverage_ratio": round(transparent_pixels / total_pixels, 6),
        "edge_foreground_ratio": round(edge_foreground_pixels / len(edge_pixels), 6),
        "alpha_min": min(alpha),
        "alpha_max": max(alpha),
    }
