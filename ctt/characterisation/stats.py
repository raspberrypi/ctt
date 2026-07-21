# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Temporal and spatial statistics of a FrameSet.
#
# Temporal statistics (per pixel, across the burst) isolate read + shot noise;
# spatial statistics (across pixels of the time-averaged frame, after removing
# a low-order shading fit) give the fixed-pattern non-uniformity — DSNU on a
# dark burst, PRNU on an illuminated flat field.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import WHITE_16, FrameSet

# A pixel whose temporal mean sits this close to the 16-bit ceiling is counted
# as clipped: saturation collapses the temporal variance and drags fits down.
_CLIP_DN = 0.98 * WHITE_16


@dataclass
class TemporalStats:
    """Burst temporal statistics over the ROI (DN16, black-subtracted mean)."""

    mean_dn: float  # ROI mean of the per-pixel temporal mean, black-subtracted
    var_dn2: float  # ROI mean of the per-pixel temporal variance (ddof=1)
    n_frames: int
    clip_fraction: float  # fraction of ROI pixels with temporal mean near white


def temporal_stats(fs: FrameSet) -> TemporalStats | None:
    """Per-pixel temporal statistics across the burst; None below two frames."""
    if len(fs.frames) < 2:
        return None
    stack = np.stack([f.astype(np.float64) for f in fs.frames])
    per_pixel_mean = stack.mean(axis=0)
    per_pixel_var = stack.var(axis=0, ddof=1)
    return TemporalStats(
        mean_dn=float(per_pixel_mean.mean() - fs.blacklevel_16),
        var_dn2=float(per_pixel_var.mean()),
        n_frames=len(fs.frames),
        clip_fraction=float((per_pixel_mean >= _CLIP_DN).mean()),
    )


def shading_fit(mean_frame: np.ndarray, order: int = 2) -> np.ndarray:
    """Least-squares low-order 2D polynomial fit of a frame (the shading term).

    order=2 fits the six terms 1, x, y, x^2, xy, y^2 on coordinates normalised
    to [-1, 1] — enough to absorb lens shading and vignetting without eating
    the pixel-scale fixed pattern being measured.
    """
    h, w = mean_frame.shape
    y, x = np.mgrid[0:h, 0:w]
    x = 2.0 * x / max(w - 1, 1) - 1.0
    y = 2.0 * y / max(h - 1, 1) - 1.0
    cols = [np.ones_like(x)]
    for total in range(1, order + 1):
        for i in range(total + 1):
            cols.append(x ** (total - i) * y**i)
    basis = np.stack([c.ravel() for c in cols], axis=1)
    coeffs, *_ = np.linalg.lstsq(basis, mean_frame.ravel(), rcond=None)
    return (basis @ coeffs).reshape(mean_frame.shape)


@dataclass
class SpatialStats:
    """Fixed-pattern non-uniformity of the time-averaged frame (DN16)."""

    mean_dn: float  # mean of the averaged frame, black-subtracted
    residual_std_dn: float  # spatial sigma after shading removal
    nonuniformity_pct: float | None  # residual/mean * 100; None near zero mean
    n_frames: int


def spatial_stats(fs: FrameSet, order: int = 2) -> SpatialStats:
    """Fixed-pattern statistics: time-average, remove shading, take the residual.

    Time-averaging first suppresses the temporal noise by sqrt(N), so the
    residual is dominated by the fixed pattern: DSNU in DN16 on a dark burst,
    PRNU (as % of mean signal) on a flat field.
    """
    stack = np.stack([f.astype(np.float64) for f in fs.frames])
    mean_frame = stack.mean(axis=0)
    residual = mean_frame - shading_fit(mean_frame, order)
    mean_dn = float(mean_frame.mean() - fs.blacklevel_16)
    residual_std = float(residual.std())
    # Percentage non-uniformity is meaningless without signal (e.g. darks).
    pct = residual_std / mean_dn * 100.0 if mean_dn > 1.0 else None
    return SpatialStats(
        mean_dn=mean_dn,
        residual_std_dn=residual_std,
        nonuniformity_pct=pct,
        n_frames=len(fs.frames),
    )
