from __future__ import annotations

from io import BytesIO
import unittest

from wearwise_ai.application.clothing_classification_service import (
    ClothingClassificationRequest,
    ClothingClassificationService,
    RawCategoryPrediction,
)
from wearwise_ai.workflows.clothing_classification_job import ClothingClassificationJobWorkflow


class FakeInternalApiClient:
    def __init__(self, *, background_removed: dict[str, object]) -> None:
        self.background_removed = background_removed
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


class FakeClassifier:
    def __init__(self) -> None:
        self.requests: list[ClothingClassificationRequest] = []

    def classify(self, request: ClothingClassificationRequest) -> RawCategoryPrediction:
        self.requests.append(request)
        return RawCategoryPrediction(
            category_code="tops",
            confidence=0.9,
            model_family="siglip",
            model_name="test-siglip",
            model_version="rev-1",
        )


class ClothingClassificationJobWorkflowTests(unittest.TestCase):
    def test_workflow_completes_with_category_for_accepted_cutout(self) -> None:
        internal_api = FakeInternalApiClient(
            background_removed={
                "storage_key": "ai-outputs/asset-1/background_removed_v4.webp",
                "content_type": "image/png",
                "review_required": False,
                "quality_status": "accepted",
            }
        )
        classifier = FakeClassifier()
        artifact_reader = FakeArtifactReader(_rgb_image_bytes())
        workflow = ClothingClassificationJobWorkflow(
            internal_api=internal_api,
            classification_service=ClothingClassificationService(classifier=classifier),
            binary_fetcher=FakeBinaryFetcher(b""),
            artifact_reader=artifact_reader,
        )

        outcome = workflow.run(job_id="job-1", worker_id="worker-a", request_id="req-1")

        self.assertEqual(outcome.status, "ready")
        self.assertEqual(artifact_reader.keys, ["ai-outputs/asset-1/background_removed_v4.webp"])
        self.assertEqual(classifier.requests[0].image_bytes, _rgb_image_bytes())
        self.assertEqual(
            internal_api.progress_updates,
            [
                (15, "checking background removal quality"),
                (45, "fetching cleaned garment image"),
                (60, "validating cleaned garment image"),
                (75, "classifying garment category"),
            ],
        )
        payload = internal_api.completed_payloads[0]
        self.assertEqual(payload["status_message"], "clothing classification completed")
        self.assertEqual(payload["result"]["category"]["value"], "tops")
        self.assertIsNone(payload["result"]["subcategory"])
        self.assertFalse(payload["result"]["review_required"])
        self.assertEqual(len(internal_api.failed_payloads), 0)

    def test_workflow_skips_classification_when_background_removal_needs_review(self) -> None:
        internal_api = FakeInternalApiClient(
            background_removed={
                "access_url": "https://example.test/cutout.webp",
                "review_required": True,
                "quality_status": "needs_review",
                "quality_reasons": ["foreground_too_large"],
            }
        )
        fetcher = FakeBinaryFetcher(b"clean-cutout")
        workflow = ClothingClassificationJobWorkflow(
            internal_api=internal_api,
            classification_service=ClothingClassificationService(classifier=FakeClassifier()),
            binary_fetcher=fetcher,
        )

        outcome = workflow.run(job_id="job-2", worker_id="worker-a", request_id="req-2")

        self.assertEqual(outcome.status, "needs_review")
        self.assertEqual(fetcher.urls, [])
        payload = internal_api.completed_payloads[0]
        self.assertEqual(
            payload["status_message"],
            "clothing classification skipped; background removal needs review",
        )
        self.assertTrue(payload["result"]["review_required"])
        self.assertEqual(payload["result"]["review_reasons"], ["foreground_too_large"])

    def test_workflow_fails_when_cleaned_image_access_is_missing(self) -> None:
        internal_api = FakeInternalApiClient(background_removed={"review_required": False})
        workflow = ClothingClassificationJobWorkflow(
            internal_api=internal_api,
            classification_service=ClothingClassificationService(classifier=FakeClassifier()),
            binary_fetcher=FakeBinaryFetcher(b"clean-cutout"),
        )

        outcome = workflow.run(job_id="job-3", worker_id="worker-a", request_id="req-3")

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(len(internal_api.completed_payloads), 0)
        self.assertEqual(
            internal_api.failed_payloads[0]["error_code"],
            "CLOTHING_CLASSIFICATION_INPUT_MISSING",
        )

    def test_workflow_fails_invalid_cleaned_image_before_classification(self) -> None:
        internal_api = FakeInternalApiClient(
            background_removed={
                "storage_key": "ai-outputs/asset-1/background_removed_v4.webp",
                "content_type": "image/png",
                "review_required": False,
                "quality_status": "accepted",
            }
        )
        classifier = FakeClassifier()
        workflow = ClothingClassificationJobWorkflow(
            internal_api=internal_api,
            classification_service=ClothingClassificationService(classifier=classifier),
            artifact_reader=FakeArtifactReader(b"not-an-image"),
        )

        outcome = workflow.run(job_id="job-4", worker_id="worker-a", request_id="req-4")

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(classifier.requests, [])
        self.assertEqual(
            internal_api.failed_payloads[0]["error_code"],
            "CLOTHING_CLASSIFICATION_INPUT_INVALID",
        )


def _rgb_image_bytes() -> bytes:
    from PIL import Image

    image = Image.new("RGB", (128, 128), (35, 85, 155))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
