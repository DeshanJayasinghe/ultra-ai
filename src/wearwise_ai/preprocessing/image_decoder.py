from __future__ import annotations

from dataclasses import dataclass
import struct


class ImageDecodingError(ValueError):
    """Raised when image bytes cannot be identified safely."""


@dataclass(frozen=True, slots=True)
class ImageProbe:
    width: int
    height: int
    format_name: str
    mime_type: str
    frame_count: int = 1
    has_alpha: bool = False

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def long_edge(self) -> int:
        return max(self.width, self.height)

    @property
    def short_edge(self) -> int:
        return min(self.width, self.height)


def probe_image_bytes(payload: bytes) -> ImageProbe:
    if len(payload) < 12:
        raise ImageDecodingError("Image payload is too small to probe safely.")

    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return _probe_png(payload)

    if payload.startswith(b"\xff\xd8"):
        return _probe_jpeg(payload)

    if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return _probe_webp(payload)

    raise ImageDecodingError("Unsupported image format.")


def _probe_png(payload: bytes) -> ImageProbe:
    if len(payload) < 33:
        raise ImageDecodingError("PNG payload is incomplete.")

    chunk_length = struct.unpack(">I", payload[8:12])[0]
    chunk_type = payload[12:16]

    if chunk_type != b"IHDR" or chunk_length < 13:
        raise ImageDecodingError("PNG header chunk is invalid.")

    width = struct.unpack(">I", payload[16:20])[0]
    height = struct.unpack(">I", payload[20:24])[0]
    color_type = payload[25]
    has_alpha = color_type in {4, 6}

    return ImageProbe(
        width=width,
        height=height,
        format_name="png",
        mime_type="image/png",
        has_alpha=has_alpha,
    )


def _probe_jpeg(payload: bytes) -> ImageProbe:
    offset = 2
    size = len(payload)

    while offset + 9 < size:
        if payload[offset] != 0xFF:
            raise ImageDecodingError("JPEG marker stream is invalid.")

        marker = payload[offset + 1]
        offset += 2

        if marker in {0xD8, 0xD9}:
            continue

        if marker == 0xDA:
            break

        if offset + 2 > size:
            break

        segment_length = struct.unpack(">H", payload[offset:offset + 2])[0]

        if segment_length < 2 or offset + segment_length > size:
            raise ImageDecodingError("JPEG segment length is invalid.")

        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if segment_length < 7:
                raise ImageDecodingError("JPEG start-of-frame segment is incomplete.")

            height = struct.unpack(">H", payload[offset + 3:offset + 5])[0]
            width = struct.unpack(">H", payload[offset + 5:offset + 7])[0]

            return ImageProbe(
                width=width,
                height=height,
                format_name="jpeg",
                mime_type="image/jpeg",
            )

        offset += segment_length

    raise ImageDecodingError("JPEG dimensions could not be determined.")


def _probe_webp(payload: bytes) -> ImageProbe:
    if len(payload) < 30:
        raise ImageDecodingError("WEBP payload is incomplete.")

    chunk_type = payload[12:16]

    if chunk_type == b"VP8X":
        width_minus_one = int.from_bytes(payload[24:27], "little")
        height_minus_one = int.from_bytes(payload[27:30], "little")
        flags = payload[20]

        return ImageProbe(
            width=width_minus_one + 1,
            height=height_minus_one + 1,
            format_name="webp",
            mime_type="image/webp",
            has_alpha=bool(flags & 0b00010000),
        )

    raise ImageDecodingError("Only extended WEBP probing is currently supported.")
