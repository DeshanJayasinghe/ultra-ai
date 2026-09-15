from __future__ import annotations

import unittest

from wearwise_ai.preprocessing import (
    ImageConstraints,
    plan_normalization,
    probe_image_bytes,
    resolve_orientation_transform,
    validate_image_input,
)


PNG_2X3 = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x02"
    b"\x00\x00\x00\x03"
    b"\x08\x06\x00\x00\x00"
    b"\xf4x\xd4\xfa"
)

JPEG_640X480 = bytes.fromhex(
    "ffd8"
    "ffe000104a46494600010101006000600000"
    "ffc000110801e0028003012200021101031101"
    "ffda000c03010002110311003f00"
)

WEBP_17X9 = (
    b"RIFF"
    b"\x1a\x00\x00\x00"
    b"WEBP"
    b"VP8X"
    b"\x0a\x00\x00\x00"
    b"\x10\x00\x00\x00"
    b"\x10\x00\x00"
    b"\x08\x00\x00"
)


class ImageProbeTests(unittest.TestCase):
    def test_probe_png_dimensions(self) -> None:
        probe = probe_image_bytes(PNG_2X3)

        self.assertEqual(probe.format_name, "png")
        self.assertEqual(probe.mime_type, "image/png")
        self.assertEqual(probe.width, 2)
        self.assertEqual(probe.height, 3)
        self.assertTrue(probe.has_alpha)

    def test_probe_jpeg_dimensions(self) -> None:
        probe = probe_image_bytes(JPEG_640X480)

        self.assertEqual(probe.format_name, "jpeg")
        self.assertEqual(probe.mime_type, "image/jpeg")
        self.assertEqual(probe.width, 640)
        self.assertEqual(probe.height, 480)

    def test_probe_extended_webp_dimensions(self) -> None:
        probe = probe_image_bytes(WEBP_17X9)

        self.assertEqual(probe.format_name, "webp")
        self.assertEqual(probe.width, 17)
        self.assertEqual(probe.height, 9)
        self.assertTrue(probe.has_alpha)


class ValidationTests(unittest.TestCase):
    def test_validation_accepts_supported_jpeg(self) -> None:
        result = validate_image_input(
            JPEG_640X480,
            declared_mime_type="image/jpeg",
            constraints=ImageConstraints(min_width=200, min_height=200, min_long_edge=400),
        )

        self.assertTrue(result.is_valid)
        self.assertIsNotNone(result.probe)
        self.assertEqual(result.issues, ())

    def test_validation_rejects_mime_type_mismatch(self) -> None:
        result = validate_image_input(
            JPEG_640X480,
            declared_mime_type="image/png",
            constraints=ImageConstraints(min_width=200, min_height=200, min_long_edge=400),
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.issues[0].code, "mime_type_mismatch")

    def test_validation_flags_small_images(self) -> None:
        result = validate_image_input(
            PNG_2X3,
            declared_mime_type="image/png",
            constraints=ImageConstraints(min_width=10, min_height=10, min_long_edge=10),
        )

        self.assertFalse(result.is_valid)
        self.assertTrue(any(issue.code == "image_dimensions_too_small" for issue in result.issues))


class OrientationAndNormalizationTests(unittest.TestCase):
    def test_orientation_six_swaps_dimensions(self) -> None:
        transform = resolve_orientation_transform(6)

        self.assertTrue(transform.swap_dimensions)
        self.assertEqual(transform.rotation_degrees, 90)
        self.assertFalse(transform.mirrored)

    def test_normalization_plan_scales_large_image(self) -> None:
        probe = probe_image_bytes(JPEG_640X480)
        plan = plan_normalization(probe)

        self.assertEqual(plan.normalization_version, 2)
        self.assertEqual(plan.normalized_width, 640)
        self.assertEqual(plan.normalized_height, 480)
        self.assertEqual(plan.target_canvas_width, 2048)
        self.assertEqual(plan.target_canvas_height, 1536)
        self.assertEqual(plan.output_format, "webp")

    def test_normalization_plan_respects_orientation_swap(self) -> None:
        probe = probe_image_bytes(JPEG_640X480)
        plan = plan_normalization(probe, exif_orientation=6)

        self.assertEqual(plan.source_width, 480)
        self.assertEqual(plan.source_height, 640)
        self.assertEqual(plan.orientation.rotation_degrees, 90)


if __name__ == "__main__":
    unittest.main()
