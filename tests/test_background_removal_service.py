from __future__ import annotations

import unittest

from wearwise_ai.application.background_removal_service import (
    BackgroundRemovalModelInfo,
    BackgroundRemovalOutput,
    BackgroundRemovalRequest,
    BackgroundRemovalService,
)
from wearwise_ai.preprocessing import ImageProbe, plan_normalization


class FakeBackgroundRemover:
    @property
    def model_info(self) -> BackgroundRemovalModelInfo:
        return BackgroundRemovalModelInfo(
            family="birefnet",
            name="birefnet-lite",
            version="1.0.0",
        )

    def remove_background(self, request: BackgroundRemovalRequest) -> BackgroundRemovalOutput:
        return BackgroundRemovalOutput(
            image_bytes=b"clean-cutout",
            content_type="image/webp",
            confidence=0.91,
            width=1024,
            height=1536,
            metrics={
                "duration_ms": 140,
                "foreground_coverage_ratio": 0.5,
                "edge_foreground_ratio": 0.12,
            },
        )


class FakeArtifactStore:
    def __init__(self) -> None:
        self.writes: list[tuple[str, bytes, str]] = []

    def write_bytes(self, key: str, payload: bytes, content_type: str) -> None:
        self.writes.append((key, payload, content_type))


class BackgroundRemovalServiceTests(unittest.TestCase):
    def test_process_writes_background_removed_variant_and_builds_complete_payload(self) -> None:
        store = FakeArtifactStore()
        service = BackgroundRemovalService(
            remover=FakeBackgroundRemover(),
            artifact_store=store,
        )
        plan = plan_normalization(
            ImageProbe(
                width=640,
                height=480,
                format_name="jpeg",
                mime_type="image/jpeg",
            )
        )

        result = service.process(
            BackgroundRemovalRequest(
                asset_id="asset-1",
                image_bytes=b"original-bytes",
                normalization_plan=plan,
            )
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.quality_status, "accepted")
        self.assertEqual(result.model.family, "birefnet")
        self.assertEqual(result.artifact.variant, "background_removed_v4")
        self.assertEqual(result.artifact.width, 1024)
        self.assertEqual(result.artifact.height, 1536)
        self.assertEqual(
            store.writes,
            [("ai-outputs/asset-1/background_removed_v4.webp", b"clean-cutout", "image/webp")],
        )

        payload = result.to_complete_payload(worker_id="worker-a")
        self.assertEqual(payload["model_family"], "birefnet")
        self.assertEqual(payload["status_message"], "background removal completed")
        self.assertEqual(payload["result"]["background_removed"]["variant"], "background_removed_v4")
        self.assertEqual(payload["result"]["background_removed"]["width"], 1024)
        self.assertEqual(payload["result"]["background_removed"]["height"], 1536)
        self.assertEqual(payload["result"]["background_removed"]["quality_status"], "accepted")
        self.assertFalse(payload["result"]["background_removed"]["review_required"])
        self.assertEqual(payload["resource_metrics"]["background_removal_quality_status"], "accepted")

    def test_process_marks_review_required_when_quality_gate_fails(self) -> None:
        class ReviewBackgroundRemover(FakeBackgroundRemover):
            def remove_background(self, request: BackgroundRemovalRequest) -> BackgroundRemovalOutput:
                return BackgroundRemovalOutput(
                    image_bytes=b"review-cutout",
                    content_type="image/webp",
                    confidence=0.91,
                    metrics={
                        "foreground_coverage_ratio": 0.94,
                        "edge_foreground_ratio": 0.12,
                    },
                )

        service = BackgroundRemovalService(
            remover=ReviewBackgroundRemover(),
            artifact_store=FakeArtifactStore(),
        )
        plan = plan_normalization(
            ImageProbe(width=640, height=480, format_name="jpeg", mime_type="image/jpeg")
        )

        result = service.process(
            BackgroundRemovalRequest(
                asset_id="asset-review",
                image_bytes=b"original-bytes",
                normalization_plan=plan,
            )
        )

        payload = result.to_complete_payload(worker_id="worker-a")
        self.assertEqual(result.quality_status, "needs_review")
        self.assertEqual(result.quality_reasons, ("foreground_too_large",))
        self.assertEqual(
            payload["status_message"],
            "background removal completed; review recommended",
        )
        self.assertTrue(payload["result"]["background_removed"]["review_required"])


if __name__ == "__main__":
    unittest.main()
