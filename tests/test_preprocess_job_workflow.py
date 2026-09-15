from __future__ import annotations

import unittest

from wearwise_ai.application.preprocessing_service import PreprocessingService
from wearwise_ai.preprocessing import ImageConstraints
from wearwise_ai.workflows.preprocess_job import PreprocessJobWorkflow


JPEG_640X480 = bytes.fromhex(
    "ffd8"
    "ffe000104a46494600010101006000600000"
    "ffc000110801e0028003012200021101031101"
    "ffda000c03010002110311003f00"
)


class FakeInternalApiClient:
    def __init__(self) -> None:
        self.progress_updates: list[tuple[int, str | None]] = []

    def get_ai_job(self, job_id: str, *, request_id: str) -> dict[str, object]:
        return {
            "data": {
                "id": job_id,
                "media_asset_id": "asset-1",
            }
        }

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


class FakeBinaryFetcher:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def fetch(self, url: str) -> bytes:
        self.urls.append(url)
        return self.payload


class PreprocessJobWorkflowTests(unittest.TestCase):
    def test_workflow_fetches_media_and_returns_ready_result(self) -> None:
        internal_api = FakeInternalApiClient()
        fetcher = FakeBinaryFetcher(JPEG_640X480)
        workflow = PreprocessJobWorkflow(
            internal_api=internal_api,
            preprocessing_service=PreprocessingService(
                constraints=ImageConstraints(min_width=200, min_height=200, min_long_edge=400)
            ),
            binary_fetcher=fetcher,
        )

        outcome = workflow.run(
            job_id="job-1",
            worker_id="worker-a",
            request_id="req-1",
        )

        self.assertEqual(outcome.job_id, "job-1")
        self.assertEqual(outcome.asset_id, "asset-1")
        self.assertEqual(outcome.result.status, "ready")
        self.assertEqual(fetcher.urls, ["https://example.test/asset.jpg"])
        self.assertEqual(
            internal_api.progress_updates,
            [
                (10, "fetching media asset"),
                (40, "validating and planning normalization"),
                (90, "preprocessing ready"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
