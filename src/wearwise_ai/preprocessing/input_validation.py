from __future__ import annotations

from dataclasses import dataclass, field

from wearwise_ai.preprocessing.image_decoder import ImageDecodingError, ImageProbe, probe_image_bytes


@dataclass(frozen=True, slots=True)
class ImageConstraints:
    max_bytes: int = 12 * 1024 * 1024
    min_width: int = 256
    min_height: int = 256
    min_long_edge: int = 512
    max_long_edge: int = 4096
    min_megapixels: float = 0.2
    max_megapixels: float = 20.0
    supported_mime_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")


@dataclass(frozen=True, slots=True)
class ImageValidationIssue:
    code: str
    message: str
    severity: str = "error"


@dataclass(frozen=True, slots=True)
class ImageValidationResult:
    is_valid: bool
    probe: ImageProbe | None
    issues: tuple[ImageValidationIssue, ...] = field(default_factory=tuple)


def validate_image_input(
    payload: bytes,
    *,
    declared_mime_type: str | None = None,
    constraints: ImageConstraints | None = None,
) -> ImageValidationResult:
    rules = constraints or ImageConstraints()
    issues: list[ImageValidationIssue] = []

    if not payload:
        issues.append(ImageValidationIssue("empty_payload", "Image payload is empty."))
        return ImageValidationResult(is_valid=False, probe=None, issues=tuple(issues))

    if len(payload) > rules.max_bytes:
        issues.append(
            ImageValidationIssue(
                "payload_too_large",
                f"Image payload exceeds the maximum size of {rules.max_bytes} bytes.",
            )
        )

    try:
        probe = probe_image_bytes(payload)
    except ImageDecodingError as exc:
        issues.append(ImageValidationIssue("unsupported_image", str(exc)))
        return ImageValidationResult(is_valid=False, probe=None, issues=tuple(issues))

    if probe.mime_type not in rules.supported_mime_types:
        issues.append(
            ImageValidationIssue(
                "unsupported_mime_type",
                f"Image mime type {probe.mime_type} is not supported.",
            )
        )

    if declared_mime_type and declared_mime_type != probe.mime_type:
        issues.append(
            ImageValidationIssue(
                "mime_type_mismatch",
                f"Declared mime type {declared_mime_type} does not match detected type {probe.mime_type}.",
            )
        )

    if probe.width < rules.min_width or probe.height < rules.min_height:
        issues.append(
            ImageValidationIssue(
                "image_dimensions_too_small",
                "Image dimensions are too small for reliable garment processing.",
            )
        )

    if probe.long_edge < rules.min_long_edge:
        issues.append(
            ImageValidationIssue(
                "long_edge_too_small",
                "Image long edge is too small for reliable preprocessing.",
            )
        )

    if probe.long_edge > rules.max_long_edge:
        issues.append(
            ImageValidationIssue(
                "long_edge_too_large",
                "Image long edge exceeds the current preprocessing limit.",
                severity="warning",
            )
        )

    if probe.megapixels < rules.min_megapixels:
        issues.append(
            ImageValidationIssue(
                "image_megapixels_too_small",
                "Image does not contain enough pixels for reliable AI processing.",
            )
        )

    if probe.megapixels > rules.max_megapixels:
        issues.append(
            ImageValidationIssue(
                "image_megapixels_too_large",
                "Image should be normalized before expensive inference.",
                severity="warning",
            )
        )

    return ImageValidationResult(
        is_valid=not any(issue.severity == "error" for issue in issues),
        probe=probe,
        issues=tuple(issues),
    )
