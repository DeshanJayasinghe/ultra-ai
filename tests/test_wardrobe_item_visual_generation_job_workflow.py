from __future__ import annotations

import struct
import unittest

from wearwise_ai.application.wardrobe_item_visual_generation_service import (
    WardrobeItemVisualGenerationRequest,
    WardrobeItemVisualGenerationService,
    WardrobeItemVisualOutput,
)
from wearwise_ai.workflows.wardrobe_item_visual_generation_job import (
    WardrobeItemVisualGenerationJobWorkflow,
)


class FakeInternalApiClient:
    def __init__(self, *, include_background_removed_options: bool = True) -> None:
        self.include_background_removed_options = include_background_removed_options
        self.progress_updates: list[tuple[int, str | None]] = []
        self.completed_payloads: list[dict[str, object]] = []
        self.access_requests: list[tuple[str, str | None]] = []

    def get_ai_job(self, job_id: str, *, request_id: str) -> dict[str, object]:
        options: dict[str, object] = {
            "name": "white t shirt",
            "category_code": "tops",
        }
        if self.include_background_removed_options:
            options["background_removed"] = {
                "storage_key": "ai-outputs/asset-1/background_removed_v4.webp",
                "content_type": "image/png",
                "review_required": False,
            }

        return {
            "data": {
                "id": job_id,
                "media_asset_id": "asset-1",
                "options": options,
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
                    "content_type": "image/jpeg",
                    "variants": [
                        {
                            "variant": "background_removed_v4",
                            "content_type": "image/webp",
                        }
                    ],
                },
                "access": {
                    "url": f"https://example.test/{asset_id}/background_removed_v4.webp",
                    "variant": "background_removed_v4",
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
        raise AssertionError(f"unexpected failure payload: {payload}")


class FakeArtifactReader:
    def __init__(self, image_bytes: bytes) -> None:
        self.image_bytes = image_bytes
        self.keys: list[str] = []

    def read_bytes(self, key: str) -> bytes:
        self.keys.append(key)
        return self.image_bytes


class FakeArtifactStore:
    def __init__(self) -> None:
        self.writes: list[tuple[str, bytes, str]] = []

    def write_bytes(self, key: str, payload: bytes, content_type: str) -> None:
        self.writes.append((key, payload, content_type))


class FakeGenerator:
    def __init__(self) -> None:
        self.requests: list[WardrobeItemVisualGenerationRequest] = []

    def generate(self, request: WardrobeItemVisualGenerationRequest) -> WardrobeItemVisualOutput:
        self.requests.append(request)
        return WardrobeItemVisualOutput(
            image_bytes=b"presentation-webp",
            content_type="image/webp",
            width=1024,
            height=1024,
            model_family="openai",
            model_name="gpt-image-test",
            model_version="images-v1",
            prompt_version="wardrobe-item-presentation-v1",
        )


class WardrobeItemVisualGenerationWorkflowTests(unittest.TestCase):
    def test_workflow_generates_and_completes_presentation_image(self) -> None:
        internal_api = FakeInternalApiClient()
        artifact_reader = FakeArtifactReader(_png_image_bytes())
        artifact_store = FakeArtifactStore()
        generator = FakeGenerator()
        workflow = WardrobeItemVisualGenerationJobWorkflow(
            internal_api=internal_api,
            generation_service=WardrobeItemVisualGenerationService(
                generator=generator,
                artifact_store=artifact_store,
            ),
            artifact_reader=artifact_reader,
        )

        outcome = workflow.run(job_id="job-1", worker_id="worker-a", request_id="req-1")

        self.assertEqual(outcome.media_asset_id, "asset-1")
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(generator.requests[0].name, "white t shirt")
        self.assertEqual(generator.requests[0].category_code, "tops")
        self.assertEqual(
            artifact_reader.keys,
            ["ai-outputs/asset-1/background_removed_v4.webp"],
        )
        self.assertEqual(
            artifact_store.writes,
            [("ai-outputs/asset-1/presentation_v1.webp", b"presentation-webp", "image/webp")],
        )
        payload = internal_api.completed_payloads[0]
        self.assertEqual(payload["status_message"], "wardrobe presentation image generated")
        self.assertEqual(
            payload["result"]["presentation_image"]["variant"],
            "presentation_v1",
        )

    def test_workflow_uses_variant_content_type_when_fetching_cleaned_variant(self) -> None:
        internal_api = FakeInternalApiClient(include_background_removed_options=False)
        artifact_store = FakeArtifactStore()
        generator = FakeGenerator()
        workflow = WardrobeItemVisualGenerationJobWorkflow(
            internal_api=internal_api,
            generation_service=WardrobeItemVisualGenerationService(
                generator=generator,
                artifact_store=artifact_store,
            ),
            binary_fetcher=FakeBinaryFetcher(_webp_image_bytes()),
            artifact_reader=FakeArtifactReader(b"unused"),
        )

        outcome = workflow.run(job_id="job-1", worker_id="worker-a", request_id="req-1")

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(internal_api.access_requests, [("asset-1", "background_removed_v4")])
        self.assertEqual(generator.requests[0].content_type, "image/webp")


class FakeBinaryFetcher:
    def __init__(self, image_bytes: bytes) -> None:
        self.image_bytes = image_bytes

    def fetch(self, url: str) -> bytes:
        return self.image_bytes


def _png_image_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">I", 128)
        + struct.pack(">I", 128)
        + b"\x08\x06\x00\x00\x00"
        + b"\x00\x00\x00\x00"
    )


def _webp_image_bytes() -> bytes:
    return (
        b"RIFF"
        + (22).to_bytes(4, "little")
        + b"WEBP"
        + b"VP8X"
        + (10).to_bytes(4, "little")
        + b"\x10\x00\x00\x00"
        + (127).to_bytes(3, "little")
        + (127).to_bytes(3, "little")
    )


if __name__ == "__main__":
    unittest.main()
