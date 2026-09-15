from __future__ import annotations

import unittest

from wearwise_ai.application.outfit_visual_generation_service import (
    OutfitVisualGenerationService,
    OutfitVisualGenerationRequest,
    OutfitVisualOutput,
)
from wearwise_ai.workflows.outfit_visual_generation_job import OutfitVisualGenerationJobWorkflow


class FakeInternalApiClient:
    def __init__(self) -> None:
        self.progress_updates: list[tuple[int, str | None]] = []
        self.completed_payloads: list[dict[str, object]] = []
        self.access_requests: list[tuple[str, str | None]] = []

    def get_ai_job(self, job_id: str, *, request_id: str) -> dict[str, object]:
        return {
            "data": {
                "id": job_id,
                "media_asset_id": "anchor-asset",
                "options": {
                    "title": "Classic Chic",
                    "items": [
                        {
                            "item_id": "item-1",
                            "media_asset_id": "asset-1",
                            "category_code": "tops",
                            "name": "red t shirt",
                        }
                    ],
                },
            }
        }

    def get_media_asset_access(
        self, asset_id: str, *, request_id: str, variant: str | None = None
    ) -> dict[str, object]:
        self.access_requests.append((asset_id, variant))
        return {
            "data": {
                "media_asset": {
                    "id": asset_id,
                    "content_type": "image/webp",
                },
                "access": {
                    "url": f"https://example.test/{asset_id}.webp",
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


class FakeBinaryFetcher:
    def fetch(self, url: str) -> bytes:
        return b"clean-garment"


class FakeGenerator:
    def __init__(self) -> None:
        self.requests: list[OutfitVisualGenerationRequest] = []

    def generate(self, request: OutfitVisualGenerationRequest) -> OutfitVisualOutput:
        self.requests.append(request)
        return OutfitVisualOutput(
            image_bytes=b"outfit-webp",
            content_type="image/webp",
            width=1024,
            height=1536,
            model_family="openai",
            model_name="gpt-image-test",
            model_version="images-v1",
            prompt_version="outfit-visual-flatlay-v1",
        )


class FakeArtifactStore:
    def __init__(self) -> None:
        self.writes: list[tuple[str, bytes, str]] = []

    def write_bytes(self, key: str, payload: bytes, content_type: str) -> None:
        self.writes.append((key, payload, content_type))


class OutfitVisualGenerationWorkflowTests(unittest.TestCase):
    def test_workflow_generates_and_completes_outfit_visual(self) -> None:
        internal_api = FakeInternalApiClient()
        artifact_store = FakeArtifactStore()
        generator = FakeGenerator()
        workflow = OutfitVisualGenerationJobWorkflow(
            internal_api=internal_api,
            generation_service=OutfitVisualGenerationService(
                generator=generator,
                artifact_store=artifact_store,
            ),
            binary_fetcher=FakeBinaryFetcher(),
        )

        outcome = workflow.run(job_id="job-1", worker_id="worker-a", request_id="req-1")

        self.assertEqual(outcome.anchor_media_asset_id, "anchor-asset")
        self.assertEqual(outcome.item_count, 1)
        self.assertEqual(generator.requests[0].items[0].image_bytes, b"clean-garment")
        self.assertEqual(internal_api.access_requests, [("asset-1", "presentation_v1")])
        self.assertEqual(
            artifact_store.writes,
            [("ai-outputs/anchor-asset/outfit_visual_v1.webp", b"outfit-webp", "image/webp")],
        )
        self.assertEqual(
            internal_api.progress_updates,
            [
                (15, "fetching outfit items"),
                (60, "generating outfit visual"),
            ],
        )
        payload = internal_api.completed_payloads[0]
        self.assertEqual(payload["status_message"], "outfit visual generated")
        self.assertEqual(payload["resource_metrics"], {"input_item_count": 1})
        self.assertEqual(
            payload["result"]["outfit_visual"]["variant"],
            "outfit_visual_v1",
        )


if __name__ == "__main__":
    unittest.main()
