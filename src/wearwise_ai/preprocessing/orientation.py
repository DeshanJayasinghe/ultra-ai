from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OrientationTransform:
    exif_orientation: int
    swap_dimensions: bool
    rotation_degrees: int
    mirrored: bool


_ORIENTATION_TRANSFORMS: dict[int, OrientationTransform] = {
    1: OrientationTransform(1, swap_dimensions=False, rotation_degrees=0, mirrored=False),
    2: OrientationTransform(2, swap_dimensions=False, rotation_degrees=0, mirrored=True),
    3: OrientationTransform(3, swap_dimensions=False, rotation_degrees=180, mirrored=False),
    4: OrientationTransform(4, swap_dimensions=False, rotation_degrees=180, mirrored=True),
    5: OrientationTransform(5, swap_dimensions=True, rotation_degrees=90, mirrored=True),
    6: OrientationTransform(6, swap_dimensions=True, rotation_degrees=90, mirrored=False),
    7: OrientationTransform(7, swap_dimensions=True, rotation_degrees=270, mirrored=True),
    8: OrientationTransform(8, swap_dimensions=True, rotation_degrees=270, mirrored=False),
}


def resolve_orientation_transform(exif_orientation: int | None) -> OrientationTransform:
    if exif_orientation is None:
        return _ORIENTATION_TRANSFORMS[1]

    return _ORIENTATION_TRANSFORMS.get(exif_orientation, _ORIENTATION_TRANSFORMS[1])
