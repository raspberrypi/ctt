# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# FrameSet: the source-agnostic unit of characterisation work.
#
# A FrameSet is one operating point — a burst of measurement-channel ROI crops
# plus the exposure/gain/pedestal that produced them. The offline path builds
# one from DNGs on disk (frameset_from_dngs); a live capture path can build one
# directly from raw arrays. All downstream statistics are identical.
#
# All DN quantities use the loader's left-justified 16-bit domain (DN16): DNG
# writers disagree about the container's significant bits (PiDNG writes
# 16-bit-scaled samples with WhiteLevel 65535, rpicam writes native-depth
# samples), so the shifted domain of image_loader.dng_load_image is the only
# representation uniform across sources. sigbits is carried for reporting.

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..core.image import Image
from ..core.image_loader import dng_load_image

logger = logging.getLogger(__name__)

# 16-bit white point of the loader domain.
WHITE_16 = 65535


class MismatchedGroupError(ValueError):
    """Raised when burst frames disagree on their operating point or geometry."""


@dataclass
class FrameSet:
    """One operating point: a burst of measurement-channel ROI crops (DN16).

    frames hold uint16 or float64 arrays; statistics promote to float64. The
    single measurement channel is Gr (or the sole plane, 'Y', on mono sensors):
    averaging the two greens halves the per-pixel variance and doubles the
    apparent conversion gain.
    """

    frames: list[np.ndarray]
    exposure_us: int
    gain: float
    blacklevel_16: float
    sigbits: int
    channel: str  # 'Gr' or 'Y'
    label: str = ''
    sources: list[str] = field(default_factory=list)


def centre_roi(shape: tuple[int, int], fraction: float = 0.5) -> tuple[slice, slice]:
    """Centred ROI slices covering `fraction` of each axis (0.5 -> 25% of area).

    A central region avoids lens shading and vignetting at the edges, keeping
    flat-field non-uniformity out of the temporal statistics.
    """
    h, w = shape
    rh, rw = max(1, round(h * fraction)), max(1, round(w * fraction))
    y0, x0 = (h - rh) // 2, (w - rw) // 2
    return slice(y0, y0 + rh), slice(x0, x0 + rw)


def gr_plane(img: Image) -> np.ndarray:
    """The single measurement plane: Gr via img.order, or the sole plane on mono."""
    if img.pattern == 128:
        return img.channels[0]
    return img.channels[img.order[1]]


def frameset_from_dngs(paths: list[str], roi_fraction: float = 0.5) -> FrameSet:
    """Build a FrameSet from a burst of DNGs, one frame at a time.

    Each frame's Gr ROI is cropped immediately and the rest of the image is
    dropped, so peak memory is ~ROI x N rather than burst x full-res. Frames
    must agree exactly on exposure, gain, geometry, sigbits and black level —
    discovery groups by these, so a mismatch here is an internal error.
    """
    if not paths:
        raise ValueError('frameset_from_dngs needs at least one path')
    frames: list[np.ndarray] = []
    first: Image | None = None
    for path in paths:
        img = dng_load_image(None, str(path), demosaic=False)
        plane = gr_plane(img)
        if first is None:
            first = img
            roi = centre_roi(plane.shape, roi_fraction)
        else:
            key = (img.exposure, img.againQ8_norm, img.sigbits, img.blacklevel_16, plane.shape)
            ref = (first.exposure, first.againQ8_norm, first.sigbits, first.blacklevel_16, gr_plane(first).shape)
            if key != ref:
                raise MismatchedGroupError(f'{path}: operating point differs from {first.name}: {key} != {ref}')
        frames.append(plane[roi].copy())  # copy releases the full-res parent array
    return FrameSet(
        frames=frames,
        exposure_us=first.exposure,
        gain=first.againQ8_norm,
        blacklevel_16=float(first.blacklevel_16),
        sigbits=first.sigbits,
        channel='Y' if first.pattern == 128 else 'Gr',
        sources=[str(p) for p in paths],
    )
