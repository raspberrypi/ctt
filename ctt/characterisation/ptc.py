# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Photon-transfer analysis: variance-vs-mean points, per-gain fits, and the
# derived metrics (conversion gain, read noise, and — given a sweep — SNR,
# linearity, full well and dynamic range).
#
# For a shot-noise-limited sensor the temporal variance is linear in the mean
# signal: slope = 1/K (K the conversion gain in e-/DN16), intercept = read
# noise squared. Clipped points are excluded from fits (saturation collapses
# the variance), and points are never pooled across analogue gains — the slope
# depends on gain, so each exact gain is its own fit family.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import WHITE_16, FrameSet
from .stats import temporal_stats


@dataclass
class PtcPoint:
    """One variance-vs-mean measurement at a single operating point (DN16)."""

    mean_dn: float
    var_dn2: float
    exposure_us: int
    gain: float
    n_frames: int
    clipped: bool  # excluded from fits
    label: str = ''
    source: str = 'flat'  # 'flat' | 'chart' | 'live'


def ptc_point(fs: FrameSet, clip_limit: float = 0.9, source: str = 'flat') -> PtcPoint | None:
    """One PTC point from a burst; None when temporal statistics are impossible.

    A point is flagged clipped when its black-subtracted mean exceeds
    clip_limit of the available swing, or when any meaningful fraction of ROI
    pixels sits at the ceiling — either way the variance is untrustworthy.
    """
    ts = temporal_stats(fs)
    if ts is None:
        return None
    swing = WHITE_16 - fs.blacklevel_16
    clipped = ts.mean_dn > clip_limit * swing or ts.clip_fraction > 0.001
    return PtcPoint(
        mean_dn=ts.mean_dn,
        var_dn2=ts.var_dn2,
        exposure_us=fs.exposure_us,
        gain=fs.gain,
        n_frames=ts.n_frames,
        clipped=clipped,
        label=fs.label,
        source=source,
    )


@dataclass
class PtcFit:
    """A straight-line PTC fit for one exact analogue gain."""

    gain: float
    k_e_per_dn: float  # conversion gain, e- per DN16 (native K = K16 * 2^(16-sigbits))
    read_noise_dn: float | None  # sqrt(intercept); None when the intercept is <= 0
    read_noise_e: float | None
    r2: float
    n_points: int
    span_ratio: float  # max(mean)/min(mean) of the fitted points
    reliable: bool


def fit_ptc(points: list[PtcPoint]) -> list[PtcFit]:
    """One fit per exact-gain family of unclipped flat/live points.

    Fits are always returned when arithmetically possible, but `reliable` is
    computed, not asserted: a two-point or narrow-span "fit" is reported as
    indicative only, never as the sensor's conversion gain. Chart-sourced
    points are excluded (a chart is not a flat field).
    """
    fits: list[PtcFit] = []
    usable = [p for p in points if not p.clipped and p.source != 'chart' and p.mean_dn > 0]
    for gain in sorted({round(p.gain, 4) for p in usable}):
        family = [p for p in usable if round(p.gain, 4) == gain]
        if len(family) < 2:
            continue
        means = np.array([p.mean_dn for p in family])
        variances = np.array([p.var_dn2 for p in family])
        slope, intercept = np.polyfit(means, variances, 1)
        if slope <= 0:
            continue  # non-physical; not worth reporting even as indicative
        predicted = slope * means + intercept
        ss_res = float(((variances - predicted) ** 2).sum())
        ss_tot = float(((variances - variances.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        k = 1.0 / slope
        read_noise_dn = float(np.sqrt(intercept)) if intercept > 0 else None
        span = float(means.max() / means.min())
        fits.append(
            PtcFit(
                gain=gain,
                k_e_per_dn=float(k),
                read_noise_dn=read_noise_dn,
                read_noise_e=read_noise_dn * k if read_noise_dn is not None else None,
                r2=r2,
                n_points=len(family),
                span_ratio=span,
                reliable=len(family) >= 3 and span >= 4.0 and r2 >= 0.99,
            )
        )
    return fits


# --- sweep-derived metrics (used by the live-sweep path) --------------------


def linearity(points: list[PtcPoint]) -> dict | None:
    """Straight-line fit of mean signal vs exposure for one gain family.

    Returns {'slope_dn_per_us', 'intercept_dn', 'r2', 'n_points'} from the
    unclipped points. None below three points: a two-point line always fits
    perfectly, so its R² would be confidence theatre.
    """
    usable = [p for p in points if not p.clipped and p.source != 'chart']
    if len(usable) < 3:
        return None
    x = np.array([p.exposure_us for p in usable], dtype=np.float64)
    y = np.array([p.mean_dn for p in usable])
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - predicted) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return {
        'slope_dn_per_us': float(slope),
        'intercept_dn': float(intercept),
        'r2': r2,
        'n_points': len(usable),
    }


def full_well(points: list[PtcPoint]) -> dict | None:
    """The saturation level in DN16: the highest clipped mean, if any point clipped.

    A proper knee needs a sweep that actually reaches saturation; with none of
    the points clipped the full well is not observable and None is returned.
    """
    clipped = [p for p in points if p.clipped]
    if not clipped:
        return None
    return {'full_well_dn': float(max(p.mean_dn for p in clipped))}


def snr_curve(points: list[PtcPoint]) -> list[dict]:
    """SNR (mean / temporal sigma) per unclipped point, for plotting."""
    out = []
    for p in sorted((p for p in points if not p.clipped and p.var_dn2 > 0), key=lambda p: p.mean_dn):
        out.append({'mean_dn': p.mean_dn, 'snr_db': float(20.0 * np.log10(p.mean_dn / np.sqrt(p.var_dn2)))})
    return out


def dynamic_range(full_well_e: float, read_noise_e: float) -> float:
    """Dynamic range in dB, both quantities in electrons."""
    return float(20.0 * np.log10(full_well_e / read_noise_e))
