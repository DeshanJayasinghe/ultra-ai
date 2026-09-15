from __future__ import annotations

import unittest

from wearwise_ai.application.preprocessing_service import (
    PreprocessingRequest,
    PreprocessingService,
)
from wearwise_ai.preprocessing import ImageConstraints


JPEG_640X480 = bytes.fromhex(
    "ffd8"
    "ffe000104a46494600010101006000600000"
    "ffc000110801e0028003012200021101031101"
    "ffda000c03010002110311003f00"
)


class PreprocessingServiceTests(unittest.TestCase):
    def test_process_returns_ready_result_for_supported_image(self) -> None:
        service = PreprocessingService(
            constraints=ImageConstraints(min_width=200, min_height=200, min_long_edge=400)
        )

        result = service.process(
            PreprocessingRequest(
                asset_id="asset-1",
                content_type="image/jpeg",
                image_bytes=JPEG_640X480,
            )
        )

        self.assertEqual(result.status, "ready")
        self.assertIsNotNone(result.normalization_plan)
        self.assertTrue(result.validation.is_valid)
        self.assertEqual(result.to_worker_result()["normalization_plan"]["output_format"], "webp")

    def test_process_returns_failed_result_for_invalid_image(self) -> None:
        service = PreprocessingService()

        result = service.process(
            PreprocessingRequest(
                asset_id="asset-2",
                content_type="image/jpeg",
                image_bytes=b"not-an-image",
            )
        )

        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.normalization_plan)
        self.assertFalse(result.validation.is_valid)
        self.assertEqual(result.validation.issues[0].code, "unsupported_image")


if __name__ == "__main__":
    unittest.main()
