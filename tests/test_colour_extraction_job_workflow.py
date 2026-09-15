from __future__ import annotations

import unittest
from io import BytesIO

from wearwise_ai.application.colour_extraction_service import (
    ColourExtractionRule,
    ColourExtractionService,
)
from wearwise_ai.workflows.colour_extraction_job import ColourExtractionJobWorkflow


class FakeInternalApiClient:
    def __init__(
        self,
        *,
        background_removed: dict[str, object],
        fail_progress_once: bool = False,
    ) -> None:
        self.background_removed = background_removed
        self.fail_progress_once = fail_progress_once
        self.progress_updates: list[tuple[int, str | None]] = []
        self.completed_payloads: list[dict[str, object]] = []
        self.failed_payloads: list[dict[str, object]] = []

    def get_ai_job(self, job_id: str, *, request_id: str) -> dict[str, object]:
        return {
            "data": {
                "id": job_id,
                "media_asset_id": "asset-1",
                "options": {
                    "background_removed": self.background_removed,
                },
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
        if self.fail_progress_once:
            self.fail_progress_once = False
            return {}

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


class ColourExtractionJobWorkflowTests(unittest.TestCase):
    def test_workflow_completes_with_colour_suggestions_for_accepted_cutout(self) -> None:
        internal_api = FakeInternalApiClient(
            background_removed={
                "storage_key": "ai-outputs/asset-1/background_removed_v4.webp",
                "content_type": "image/png",
                "review_required": False,
                "quality_status": "accepted",
            }
        )
        fetcher = FakeBinaryFetcher(_rgba_image_bytes())
        artifact_reader = FakeArtifactReader(_rgba_image_bytes())
        workflow = ColourExtractionJobWorkflow(
            internal_api=internal_api,
            colour_extraction_service=ColourExtractionService(
                rule=ColourExtractionRule(min_visible_pixels=2)
            ),
            binary_fetcher=fetcher,
            artifact_reader=artifact_reader,
        )

        outcome = workflow.run(job_id="job-1", worker_id="worker-a", request_id="req-1")

        self.assertEqual(outcome.status, "ready")
        self.assertEqual(fetcher.urls, [])
        self.assertEqual(artifact_reader.keys, ["ai-outputs/asset-1/background_removed_v4.webp"])
        self.assertEqual(
            internal_api.progress_updates,
            [
                (15, "checking background removal quality"),
                (45, "fetching cleaned garment image"),
                (60, "validating cleaned garment image"),
                (75, "extracting garment colours"),
            ],
        )
        payload = internal_api.completed_payloads[0]
        self.assertEqual(payload["status_message"], "colour extraction completed")
        self.assertEqual(payload["result"]["primary_colour"]["value"], "blue")
        self.assertFalse(payload["result"]["review_required"])
        self.assertEqual(len(internal_api.failed_payloads), 0)

    def test_workflow_can_fetch_accepted_cutout_from_access_url(self) -> None:
        internal_api = FakeInternalApiClient(
            background_removed={
                "access_url": "https://example.test/cutout.webp",
                "content_type": "image/png",
                "review_required": False,
                "quality_status": "accepted",
            }
        )
        fetcher = FakeBinaryFetcher(_rgba_image_bytes())
        artifact_reader = FakeArtifactReader(b"")
        workflow = ColourExtractionJobWorkflow(
            internal_api=internal_api,
            colour_extraction_service=ColourExtractionService(
                rule=ColourExtractionRule(min_visible_pixels=2)
            ),
            binary_fetcher=fetcher,
            artifact_reader=artifact_reader,
        )

        outcome = workflow.run(job_id="job-url", worker_id="worker-a", request_id="req-url")

        self.assertEqual(outcome.status, "ready")
        self.assertEqual(fetcher.urls, ["https://example.test/cutout.webp"])
        self.assertEqual(artifact_reader.keys, [])

    def test_workflow_completes_when_a_progress_update_is_dropped(self) -> None:
        internal_api = FakeInternalApiClient(
            background_removed={
                "storage_key": "ai-outputs/asset-1/background_removed_v4.webp",
                "content_type": "image/png",
                "review_required": False,
                "quality_status": "accepted",
            },
            fail_progress_once=True,
        )
        workflow = ColourExtractionJobWorkflow(
            internal_api=internal_api,
            colour_extraction_service=ColourExtractionService(
                rule=ColourExtractionRule(min_visible_pixels=2)
            ),
            artifact_reader=FakeArtifactReader(_rgba_image_bytes()),
        )

        outcome = workflow.run(job_id="job-progress", worker_id="worker-a", request_id="req")

        self.assertEqual(outcome.status, "ready")
        self.assertEqual(len(internal_api.completed_payloads), 1)
        self.assertEqual(len(internal_api.failed_payloads), 0)

    def test_workflow_skips_extraction_when_background_removal_needs_review(self) -> None:
        internal_api = FakeInternalApiClient(
            background_removed={
                "access_url": "https://example.test/cutout.webp",
                "review_required": True,
                "quality_status": "needs_review",
                "quality_reasons": ["foreground_too_large"],
            }
        )
        fetcher = FakeBinaryFetcher(_rgba_image_bytes())
        workflow = ColourExtractionJobWorkflow(
            internal_api=internal_api,
            colour_extraction_service=ColourExtractionService(),
            binary_fetcher=fetcher,
        )

        outcome = workflow.run(job_id="job-2", worker_id="worker-a", request_id="req-2")

        self.assertEqual(outcome.status, "needs_review")
        self.assertEqual(fetcher.urls, [])
        payload = internal_api.completed_payloads[0]
        self.assertEqual(
            payload["status_message"],
            "colour extraction skipped; background removal needs review",
        )
        self.assertTrue(payload["result"]["review_required"])
        self.assertEqual(payload["result"]["review_reasons"], ["foreground_too_large"])

    def test_workflow_fails_when_cleaned_image_access_is_missing(self) -> None:
        internal_api = FakeInternalApiClient(background_removed={"review_required": False})
        workflow = ColourExtractionJobWorkflow(
            internal_api=internal_api,
            colour_extraction_service=ColourExtractionService(),
            binary_fetcher=FakeBinaryFetcher(_rgba_image_bytes()),
        )

        outcome = workflow.run(job_id="job-3", worker_id="worker-a", request_id="req-3")

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(len(internal_api.completed_payloads), 0)
        self.assertEqual(
            internal_api.failed_payloads[0]["error_code"],
            "COLOUR_EXTRACTION_INPUT_MISSING",
        )

    def test_workflow_fails_invalid_cleaned_image_before_colour_extraction(self) -> None:
        internal_api = FakeInternalApiClient(
            background_removed={
                "storage_key": "ai-outputs/asset-1/background_removed_v4.webp",
                "content_type": "image/png",
                "review_required": False,
                "quality_status": "accepted",
            }
        )
        workflow = ColourExtractionJobWorkflow(
            internal_api=internal_api,
            colour_extraction_service=ColourExtractionService(),
            artifact_reader=FakeArtifactReader(b"not-an-image"),
        )

        outcome = workflow.run(job_id="job-4", worker_id="worker-a", request_id="req-4")

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(
            internal_api.failed_payloads[0]["error_code"],
            "COLOUR_EXTRACTION_INPUT_INVALID",
        )


def _rgba_image_bytes() -> bytes:
    from PIL import Image

    image = Image.new("RGBA", (128, 128))
    image.putdata(
        [(35, 85, 155, 255) if index < 8192 else (0, 0, 0, 0) for index in range(128 * 128)]
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
