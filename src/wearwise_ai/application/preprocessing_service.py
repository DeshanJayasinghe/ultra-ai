from __future__ import annotations

from dataclasses import dataclass

from wearwise_ai.preprocessing import (
    ImageConstraints,
    ImageValidationResult,
    NormalizationPlan,
    plan_normalization,
    validate_image_input,
)


PREPROCESSING_VERSION = "preprocess-v1"


@dataclass(frozen=True, slots=True)
class PreprocessingRequest:
    asset_id: str
    content_type: str
    image_bytes: bytes
    checksum_sha256: str | None = None
    exif_orientation: int | None = None


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    asset_id: str
    status: str
    validation: ImageValidationResult
    normalization_plan: NormalizationPlan | None
    preprocessing_version: str = PREPROCESSING_VERSION

    def to_worker_result(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "status": self.status,
            "preprocessing_version": self.preprocessing_version,
            "validation": {
                "is_valid": self.validation.is_valid,
                "probe": None
                if self.validation.probe is None
                else {
                    "width": self.validation.probe.width,
                    "height": self.validation.probe.height,
                    "format_name": self.validation.probe.format_name,
                    "mime_type": self.validation.probe.mime_type,
                    "long_edge": self.validation.probe.long_edge,
                    "megapixels": round(self.validation.probe.megapixels, 4),
                },
                "issues": [
                    {
                        "code": issue.code,
                        "message": issue.message,
                        "severity": issue.severity,
                    }
                    for issue in self.validation.issues
                ],
            },
            "normalization_plan": None
            if self.normalization_plan is None
            else {
                "normalization_version": self.normalization_plan.normalization_version,
                "source_width": self.normalization_plan.source_width,
                "source_height": self.normalization_plan.source_height,
                "normalized_width": self.normalization_plan.normalized_width,
                "normalized_height": self.normalization_plan.normalized_height,
                "scale_factor": round(self.normalization_plan.scale_factor, 6),
                "output_format": self.normalization_plan.output_format,
                "preserve_alpha": self.normalization_plan.preserve_alpha,
                "generate_background_removed_variant": self.normalization_plan.generate_background_removed_variant,
                "orientation": {
                    "exif_orientation": self.normalization_plan.orientation.exif_orientation,
                    "swap_dimensions": self.normalization_plan.orientation.swap_dimensions,
                    "rotation_degrees": self.normalization_plan.orientation.rotation_degrees,
                    "mirrored": self.normalization_plan.orientation.mirrored,
                },
                "target_canvas_width": self.normalization_plan.target_canvas_width,
                "target_canvas_height": self.normalization_plan.target_canvas_height,
                "estimated_garment_fill_ratio": self.normalization_plan.estimated_garment_fill_ratio,
            },
        }


class PreprocessingService:
    def __init__(self, *, constraints: ImageConstraints | None = None) -> None:
        self._constraints = constraints or ImageConstraints()

    def process(self, request: PreprocessingRequest) -> PreprocessingResult:
        validation = validate_image_input(
            request.image_bytes,
            declared_mime_type=request.content_type,
            constraints=self._constraints,
        )

        if not validation.is_valid or validation.probe is None:
            return PreprocessingResult(
                asset_id=request.asset_id,
                status="failed",
                validation=validation,
                normalization_plan=None,
            )

        plan = plan_normalization(
            validation.probe,
            exif_orientation=request.exif_orientation,
        )

        return PreprocessingResult(
            asset_id=request.asset_id,
            status="ready",
            validation=validation,
            normalization_plan=plan,
        )

