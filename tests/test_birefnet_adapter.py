from __future__ import annotations

import unittest

from wearwise_ai.application.background_removal_service import (
    BackgroundRemovalOutput,
    BackgroundRemovalRequest,
    BackgroundRemovalService,
)
from wearwise_ai.models.background_removal import (
    BiRefNetBackgroundRemover,
    BiRefNetConfig,
    ModelUnavailableError,
    check_birefnet_dependencies,
)
from wearwise_ai.preprocessing import ImageProbe, plan_normalization


class FakeBiRefNetRunner:
    def __init__(self, *, content_type: str = "image/webp") -> None:
        self.content_type = content_type
        self.calls: list[tuple[str, str]] = []

    def remove_background(
        self,
        request: BackgroundRemovalRequest,
        config: BiRefNetConfig,
    ) -> BackgroundRemovalOutput:
        self.calls.append((request.asset_id, config.model_name))
        return BackgroundRemovalOutput(
            image_bytes=b"webp-cutout",
            content_type=self.content_type,
            confidence=0.88,
            width=512,
            height=512,
            metrics={"model": config.model_name},
        )


class FakeArtifactStore:
    def __init__(self) -> None:
        self.writes: list[tuple[str, bytes, str]] = []

    def write_bytes(self, key: str, payload: bytes, content_type: str) -> None:
        self.writes.append((key, payload, content_type))


def _request() -> BackgroundRemovalRequest:
    plan = plan_normalization(
        ImageProbe(
            width=720,
            height=960,
            format_name="jpeg",
            mime_type="image/jpeg",
        )
    )
    return BackgroundRemovalRequest(
        asset_id="asset-birefnet",
        image_bytes=b"jpeg-input",
        normalization_plan=plan,
    )


class BiRefNetAdapterTests(unittest.TestCase):
    def test_dependency_check_reports_missing_modules_without_importing_them(self) -> None:
        status = check_birefnet_dependencies(("wearwise_missing_test_module",))

        self.assertFalse(status.is_ready)
        self.assertEqual(status.missing_modules, ("wearwise_missing_test_module",))

    def test_default_adapter_fails_with_controlled_model_unavailable_error(self) -> None:
        remover = BiRefNetBackgroundRemover(
            config=BiRefNetConfig(required_modules=("wearwise_missing_test_module",))
        )

        with self.assertRaises(ModelUnavailableError) as error:
            remover.remove_background(_request())

        self.assertIn("wearwise_missing_test_module", str(error.exception))

    def test_adapter_delegates_to_configured_runner(self) -> None:
        runner = FakeBiRefNetRunner()
        remover = BiRefNetBackgroundRemover(
            config=BiRefNetConfig(model_name="BiRefNet-HR", model_version="2026-08"),
            runner=runner,
        )

        output = remover.remove_background(_request())

        self.assertEqual(output.image_bytes, b"webp-cutout")
        self.assertEqual(output.content_type, "image/webp")
        self.assertEqual(runner.calls, [("asset-birefnet", "BiRefNet-HR")])
        self.assertEqual(remover.model_info.family, "birefnet")
        self.assertEqual(remover.model_info.version, "2026-08")

    def test_adapter_plugs_into_background_removal_service(self) -> None:
        store = FakeArtifactStore()
        service = BackgroundRemovalService(
            remover=BiRefNetBackgroundRemover(runner=FakeBiRefNetRunner()),
            artifact_store=store,
        )

        result = service.process(_request())

        self.assertEqual(result.model.name, "BiRefNet")
        self.assertEqual(result.confidence, 0.88)
        self.assertEqual(
            store.writes,
            [
                (
                    "ai-outputs/asset-birefnet/background_removed_v4.webp",
                    b"webp-cutout",
                    "image/webp",
                )
            ],
        )

    def test_adapter_rejects_unexpected_runner_content_type(self) -> None:
        remover = BiRefNetBackgroundRemover(runner=FakeBiRefNetRunner(content_type="image/png"))

        with self.assertRaises(ValueError):
            remover.remove_background(_request())


if __name__ == "__main__":
    unittest.main()
