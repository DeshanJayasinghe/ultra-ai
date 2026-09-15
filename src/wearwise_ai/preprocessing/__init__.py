"""Shared preprocessing primitives for WearWise AI pipelines."""

from wearwise_ai.preprocessing.image_decoder import (
    ImageDecodingError,
    ImageProbe,
    probe_image_bytes,
)
from wearwise_ai.preprocessing.image_normalizer import (
    CanvasNormalizationRule,
    NormalizationPlan,
    plan_normalization,
)
from wearwise_ai.preprocessing.input_validation import (
    ImageConstraints,
    ImageValidationIssue,
    ImageValidationResult,
    validate_image_input,
)
from wearwise_ai.preprocessing.orientation import (
    OrientationTransform,
    resolve_orientation_transform,
)

__all__ = [
    "CanvasNormalizationRule",
    "ImageConstraints",
    "ImageDecodingError",
    "ImageProbe",
    "ImageValidationIssue",
    "ImageValidationResult",
    "NormalizationPlan",
    "OrientationTransform",
    "plan_normalization",
    "probe_image_bytes",
    "resolve_orientation_transform",
    "validate_image_input",
]
