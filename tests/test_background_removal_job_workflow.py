from __future__ import annotations

import unittest

from wearwise_ai.application.background_removal_service import (
    BackgroundRemovalModelInfo,
    BackgroundRemovalOutput,
    BackgroundRemovalRequest,
    BackgroundRemovalService,
)
from wearwise_ai.application.preprocessing_service import PreprocessingService
from wearwise_ai.preprocessing import ImageConstraints
from wearwise_ai.workflows.background_removal_job import BackgroundRemovalJobWorkflow


JPEG_640X480 = bytes.fromhex(
    "ffd8"
    "ffe000104a46494600010101006000600000"
    "ffc000110801e0028003012200021101031101"
    "ffda000c03010002110311003f00"
)


class FakeInternalApiClient:
    def __init__(self) -> None:
        self.progress_updates: list[tuple[int, str | None]] = []
        self.completed_payloads: list[dict[str, object]] = []
        self.failed_payloads: list[dict[str, object]] = []

    def get_ai_job(self, job_id: str, *, request_id: str) -> dict[str, object]:
        return {"data": {"id": job_id, "media_asset_id": "asset-1"}}

    def get_media_asset_access(self, asset_id: str, *, request_id: str) -> dict[str, object]:
        return {
            "data": {
                "media_asset": {
                    "id": asset_id,
                    "content_type": "image/jpeg",
                    "checksum_sha256": None,
                },
                "access": {
                    "url": "https://example.test/asset.jpg",
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

    def fetch(self, url: str) -> bytes:
        return self.payload


class FakeBackgroundRemover:
    def __init__(
        self,
        *,
        metrics: dict[str, int | float | str] | None = None,
    ) -> None:
        self.metrics = metrics or {"duration_ms": 180}

    @property
    def model_info(self) -> BackgroundRemovalModelInfo:
        return BackgroundRemovalModelInfo("birefnet", "birefnet-lite", "1.0.0")

    def remove_background(self, request: BackgroundRemovalRequest) -> BackgroundRemovalOutput:
        return BackgroundRemovalOutput(
            image_bytes=b"clean-cutout",
            content_type="image/webp",
            confidence=0.88,
            metrics=self.metrics,
        )


class FakeArtifactStore:
    def __init__(self) -> None:
        self.writes: list[tuple[str, bytes, str]] = []

    def write_bytes(self, key: str, payload: bytes, content_type: str) -> None:
        self.writes.append((key, payload, content_type))


class BackgroundRemovalWorkflowTests(unittest.TestCase):
    def test_workflow_completes_job_after_background_removal(self) -> None:
        internal_api = FakeInternalApiClient()
        artifact_store = FakeArtifactStore()
        workflow = BackgroundRemovalJobWorkflow(
            internal_api=internal_api,
            preprocessing_service=PreprocessingService(
                constraints=ImageConstraints(min_width=200, min_height=200, min_long_edge=400)
            ),
            background_removal_service=BackgroundRemovalService(
                remover=FakeBackgroundRemover(),
                artifact_store=artifact_store,
            ),
            binary_fetcher=FakeBinaryFetcher(JPEG_640X480),
        )

        outcome = workflow.run(job_id="job-1", worker_id="worker-a", request_id="req-1")

        self.assertEqual(outcome.preprocessing_status, "ready")
        self.assertIsNotNone(outcome.result)
        self.assertEqual(
            internal_api.progress_updates,
            [
                (10, "fetching media asset"),
                (35, "validating image input"),
                (65, "running background removal"),
            ],
        )
        self.assertEqual(len(internal_api.completed_payloads), 1)
        self.assertEqual(
            internal_api.completed_payloads[0]["status_message"],
            "background removal completed",
        )
        self.assertEqual(len(internal_api.failed_payloads), 0)
        self.assertEqual(
            artifact_store.writes,
            [("ai-outputs/asset-1/background_removed_v4.webp", b"clean-cutout", "image/webp")],
        )

    def test_workflow_fails_job_when_preprocessing_is_invalid(self) -> None:
        internal_api = FakeInternalApiClient()
        workflow = BackgroundRemovalJobWorkflow(
            internal_api=internal_api,
            preprocessing_service=PreprocessingService(),
            background_removal_service=BackgroundRemovalService(
                remover=FakeBackgroundRemover(),
                artifact_store=FakeArtifactStore(),
            ),
            binary_fetcher=FakeBinaryFetcher(b"bad-bytes"),
        )

        outcome = workflow.run(job_id="job-2", worker_id="worker-a", request_id="req-2")

        self.assertEqual(outcome.preprocessing_status, "failed")
        self.assertIsNone(outcome.result)
        self.assertEqual(len(internal_api.completed_payloads), 0)
        self.assertEqual(len(internal_api.failed_payloads), 1)
        self.assertEqual(
            internal_api.failed_payloads[0]["error_code"],
            "BACKGROUND_REMOVAL_INPUT_INVALID",
        )

    def test_workflow_completes_with_review_required_payload_when_quality_fails(self) -> None:
        internal_api = FakeInternalApiClient()
        workflow = BackgroundRemovalJobWorkflow(
            internal_api=internal_api,
            preprocessing_service=PreprocessingService(
                constraints=ImageConstraints(min_width=200, min_height=200, min_long_edge=400)
            ),
            background_removal_service=BackgroundRemovalService(
                remover=FakeBackgroundRemover(
                    metrics={
                        "foreground_coverage_ratio": 0.93,
                        "edge_foreground_ratio": 0.04,
                    }
                ),
                artifact_store=FakeArtifactStore(),
            ),
            binary_fetcher=FakeBinaryFetcher(JPEG_640X480),
        )

        outcome = workflow.run(job_id="job-3", worker_id="worker-a", request_id="req-3")

        self.assertEqual(outcome.result.quality_status, "needs_review")
        self.assertEqual(len(internal_api.completed_payloads), 1)
        payload = internal_api.completed_payloads[0]
        self.assertEqual(
            payload["status_message"],
            "background removal completed; review recommended",
        )
        self.assertTrue(payload["result"]["background_removed"]["review_required"])
        self.assertEqual(
            payload["result"]["background_removed"]["quality_reasons"],
            ["foreground_too_large"],
        )


if __name__ == "__main__":
    unittest.main()
