from __future__ import annotations

from typing import Protocol

from wearwise_ai.preprocessing import ImageConstraints, validate_image_input


class JobFailureReporter(Protocol):
    def fail_job(
        self,
        job_id: str,
        *,
        request_id: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        raise NotImplementedError


_CLEANED_ARTIFACT_CONSTRAINTS = ImageConstraints(
    min_width=128,
    min_height=128,
    min_long_edge=128,
    min_megapixels=0.01,
)


def validate_cleaned_artifact_or_fail(
    *,
    internal_api: JobFailureReporter,
    job_id: str,
    worker_id: str,
    request_id: str,
    image_bytes: bytes,
    content_type: str,
    error_code: str,
    status_message: str,
) -> bool:
    validation = validate_image_input(
        image_bytes,
        declared_mime_type=content_type,
        constraints=_CLEANED_ARTIFACT_CONSTRAINTS,
    )

    if validation.is_valid:
        return True

    issue_codes = [issue.code for issue in validation.issues]
    internal_api.fail_job(
        job_id,
        request_id=request_id,
        payload={
            "worker_id": worker_id,
            "error_code": error_code,
            "error_message": "Cleaned garment image validation failed before expensive inference: "
            + ", ".join(issue_codes),
            "status_message": status_message,
            "retryable": False,
        },
    )
    return False
