from __future__ import annotations

from io import BytesIO
import unittest

from wearwise_ai.application.attribute_extraction_service import (
    AttributeExtractionRule,
    AttributeExtractionService,
)
from wearwise_ai.workflows.attribute_extraction_job import AttributeExtractionJobWorkflow


class FakeInternalApiClient:
    def __init__(self, *, background_removed: dict[str, object], category_code: str | None = "tops") -> None:
        self.background_removed = background_removed
        self.category_code = category_code
        self.progress_updates: list[tuple[int, str | None]] = []
        self.completed_payloads: list[dict[str, object]] = []
        self.failed_payloads: list[dict[str, object]] = []

    def get_ai_job(self, job_id: str, *, request_id: str) -> dict[str, object]:
        options: dict[str, object] = {"background_removed": self.background_removed}
        if self.category_code is not None:
            options["category_code"] = self.category_code

        return {
            "data": {
                "id": job_id,
                "media_asset_id": "asset-1",
                "options": options,
            }
        }

    def report_progress(
        self,
        job_id: str,
        *,
        request_id: str,
        worker_id: str,
        progress_percent: int,
        status_message: str | None = None,
    ) -> dict[str, object]:
        self.progress_updates.append((progress_percent, status_message))
        return {"data": {"id": job_id}}

    def complete_job(
        self,
        job_id: str,
        *,
        request_id: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        self.completed_payloads.append(payload)
        return {"data": {"id": job_id}}

    def fail_job(
        self,
        job_id: str,
        *,
        request_id: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        self.failed_payloads.append(payload)
        return {"data": {"id": job_id}}


class FakeBinaryFetcher:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def fetch(self, url: str) -> bytes:
        self.urls.append(url)
        return self.payload


class FakeArtifactReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.keys: list[str] = []

    def read_bytes(self, key: str) -> bytes:
        self.keys.append(key)
        return self.payload


class AttributeExtractionJobWorkflowTests(unittest.TestCase):
    def test_workflow_completes_with_attributes_for_accepted_cutout(self) -> None:
        internal_api = FakeInternalApiClient(
            background_removed={
                "storage_key": "ai-outputs/asset-1/background_removed_v4.webp",
                "content_type": "image/png",
                "review_required": False,
                "quality_status": "accepted",
            }
        )
        artifact_reader = FakeArtifactReader(_cutout_bytes())
        workflow = AttributeExtractionJobWorkflow(
            internal_api=internal_api,
            attribute_extraction_service=AttributeExtractionService(
                rule=AttributeExtractionRule(min_visible_pixels=10, min_confidence=0.5)
            ),
            binary_fetcher=FakeBinaryFetcher(b""),
            artifact_reader=artifact_reader,
        )

        outcome = workflow.run(job_id="job-1", worker_id="worker-a", request_id="req-1")

        self.assertEqual(outcome.status, "ready")
        self.assertEqual(artifact_reader.keys, ["ai-outputs/asset-1/background_removed_v4.webp"])
        self.assertEqual(
            internal_api.progress_updates,
            [
                (15, "checking background removal quality"),
                (45, "fetching cleaned garment image"),
                (60, "validating cleaned garment image"),
                (75, "extracting garment attributes"),
            ],
        )
        payload = internal_api.completed_payloads[0]
        self.assertEqual(payload["status_message"], "attribute extraction completed")
        self.assertNotIn("attributes", payload["result"])
        self.assertIn("subcategory", payload["result"])
        self.assertFalse(payload["result"]["review_required"])

    def test_workflow_skips_when_background_removal_needs_review(self) -> None:
        internal_api = FakeInternalApiClient(
            background_removed={
                "access_url": "https://example.test/cutout.webp",
                "review_required": True,
                "quality_status": "needs_review",
                "quality_reasons": ["foreground_too_large"],
            }
        )
        fetcher = FakeBinaryFetcher(_cutout_bytes())
        workflow = AttributeExtractionJobWorkflow(
            internal_api=internal_api,
            attribute_extraction_service=AttributeExtractionService(),
            binary_fetcher=fetcher,
        )

        outcome = workflow.run(job_id="job-2", worker_id="worker-a", request_id="req-2")

        self.assertEqual(outcome.status, "needs_review")
        self.assertEqual(fetcher.urls, [])
        payload = internal_api.completed_payloads[0]
        self.assertEqual(
            payload["status_message"],
            "attribute extraction skipped; background removal needs review",
        )
        self.assertTrue(payload["result"]["review_required"])

    def test_workflow_fails_when_cleaned_image_access_is_missing(self) -> None:
        internal_api = FakeInternalApiClient(background_removed={"review_required": False})
        workflow = AttributeExtractionJobWorkflow(
            internal_api=internal_api,
            attribute_extraction_service=AttributeExtractionService(),
            binary_fetcher=FakeBinaryFetcher(_cutout_bytes()),
        )

        outcome = workflow.run(job_id="job-3", worker_id="worker-a", request_id="req-3")

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(
            internal_api.failed_payloads[0]["error_code"],
            "ATTRIBUTE_EXTRACTION_INPUT_MISSING",
        )

    def test_workflow_fails_invalid_cleaned_image_before_attribute_extraction(self) -> None:
        internal_api = FakeInternalApiClient(
            background_removed={
                "storage_key": "ai-outputs/asset-1/background_removed_v4.webp",
                "content_type": "image/png",
                "review_required": False,
                "quality_status": "accepted",
            }
        )
        workflow = AttributeExtractionJobWorkflow(
            internal_api=internal_api,
            attribute_extraction_service=AttributeExtractionService(),
            artifact_reader=FakeArtifactReader(b"not-an-image"),
        )

        outcome = workflow.run(job_id="job-4", worker_id="worker-a", request_id="req-4")

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(
            internal_api.failed_payloads[0]["error_code"],
            "ATTRIBUTE_EXTRACTION_INPUT_INVALID",
        )


def _cutout_bytes() -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (128, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 38, 82, 218), fill=(30, 90, 170, 255))
    draw.rectangle((8, 50, 45, 165), fill=(30, 90, 170, 255))
    draw.rectangle((82, 50, 120, 165), fill=(30, 90, 170, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
