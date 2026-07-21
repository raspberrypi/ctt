# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Ground-truth tests for the characterisation engine: synthetic bursts with a
# known conversion gain, read noise, DSNU and PRNU, asserting the statistics
# and fits recover them. No files, no hardware.

import numpy as np
import pytest

from ctt.characterisation import (
    FrameSet,
    centre_roi,
    fit_ptc,
    ptc_point,
    shading_fit,
    spatial_stats,
    temporal_stats,
)
from ctt.characterisation.ptc import full_well, linearity, snr_curve

BLACK_16 = 3840.0  # 240 << 4: the PiDNG 16-bit-scaled pedestal observed in real files
DN_PER_E = 1.5  # sensor response in DN16 per electron (K = 1/DN_PER_E e-/DN16)
READ_NOISE_DN = 15.0  # in DN16 (~= 10 e-)


def synth_burst(
    n_frames,
    signal_e,
    *,
    exposure_us=10_000,
    gain=1.0,
    read_noise_dn=READ_NOISE_DN,
    prnu_pct=0.0,
    dsnu_dn=0.0,
    shading=0.0,
    shape=(64, 64),
    rng=None,
    label='synth',
):
    """A burst of frames: black + DN_PER_E * Poisson(signal) + read noise.

    prnu_pct applies a fixed per-pixel gain map; dsnu_dn a fixed per-pixel
    offset map; shading a quadratic bowl of that relative depth — all constant
    across the burst, as real fixed patterns are.
    """
    rng = rng or np.random.default_rng(42)
    prnu_map = 1.0 + rng.normal(0.0, prnu_pct / 100.0, shape)
    dsnu_map = rng.normal(0.0, dsnu_dn, shape)
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    bowl = 1.0 - shading * (((x / max(w - 1, 1)) - 0.5) ** 2 + ((y / max(h - 1, 1)) - 0.5) ** 2)
    frames = []
    for _ in range(n_frames):
        electrons = rng.poisson(np.maximum(signal_e * prnu_map * bowl, 0.0))
        dn = BLACK_16 + DN_PER_E * electrons + dsnu_map + rng.normal(0.0, read_noise_dn, shape)
        frames.append(np.clip(dn, 0.0, 65535.0))
    return FrameSet(
        frames=frames,
        exposure_us=exposure_us,
        gain=gain,
        blacklevel_16=BLACK_16,
        sigbits=16,
        channel='Gr',
        label=label,
    )


def sweep(exposures_e, gain=1.0, n_frames=16, rng=None, **kw):
    """One PTC point per signal level, exposure proportional to signal."""
    rng = rng or np.random.default_rng(7)
    points = []
    for i, e in enumerate(exposures_e):
        fs = synth_burst(n_frames, e, exposure_us=int(e * 10), gain=gain, rng=rng, label=f'pt{i}', **kw)
        points.append(ptc_point(fs))
    return [p for p in points if p is not None]


def test_centre_roi_covers_quarter_area():
    ys, xs = centre_roi((100, 200), 0.5)
    assert (ys.stop - ys.start) == 50 and (xs.stop - xs.start) == 100
    assert ys.start == 25 and xs.start == 50


def test_temporal_stats_recover_mean_and_variance():
    signal_e = 4000.0
    fs = synth_burst(16, signal_e, rng=np.random.default_rng(1))
    ts = temporal_stats(fs)
    # Mean: black-subtracted signal in DN16.
    assert ts.mean_dn == pytest.approx(signal_e * DN_PER_E, rel=0.02)
    # Variance: shot (DN_PER_E^2 * e) + read (READ_NOISE_DN^2).
    expected_var = DN_PER_E**2 * signal_e + READ_NOISE_DN**2
    assert ts.var_dn2 == pytest.approx(expected_var, rel=0.05)
    assert ts.clip_fraction == 0.0


def test_temporal_stats_single_frame_is_none():
    assert temporal_stats(synth_burst(1, 100.0)) is None


def test_ptc_fit_recovers_conversion_gain_and_read_noise():
    points = sweep([50, 120, 300, 700, 1500, 3500, 8000, 15000])
    fits = fit_ptc(points)
    assert len(fits) == 1
    fit = fits[0]
    assert fit.reliable
    assert fit.k_e_per_dn == pytest.approx(1.0 / DN_PER_E, rel=0.05)
    assert fit.read_noise_dn == pytest.approx(READ_NOISE_DN, rel=0.25)  # intercept is noisy
    assert fit.read_noise_e == pytest.approx(READ_NOISE_DN / DN_PER_E, rel=0.25)


def test_clipped_points_do_not_move_the_fit():
    rng = np.random.default_rng(11)
    points = sweep([50, 120, 300, 700, 1500, 3500, 8000, 15000], rng=rng)
    k_clean = fit_ptc(points)[0].k_e_per_dn
    # Saturating points: mean driven into the ceiling, variance collapsed.
    saturated = sweep([50000, 60000], rng=rng)
    assert all(p.clipped for p in saturated)
    k_with_clipped = fit_ptc(points + saturated)[0].k_e_per_dn
    assert k_with_clipped == pytest.approx(k_clean, rel=1e-9)


def test_fits_are_per_gain_family_and_never_pooled():
    rng = np.random.default_rng(3)
    points = sweep([100, 500, 2000, 9000], gain=1.0, rng=rng) + sweep([100, 500, 2000, 9000], gain=1.07, rng=rng)
    fits = fit_ptc(points)
    assert [f.gain for f in fits] == [1.0, 1.07]
    assert all(f.n_points == 4 for f in fits)


def test_sparse_or_narrow_fits_are_not_reliable():
    # Two points: fit exists but must be flagged indicative.
    two = fit_ptc(sweep([1000, 1600]))
    assert len(two) == 1 and not two[0].reliable and two[0].n_points == 2
    # Three points but a narrow span: still not reliable.
    narrow = fit_ptc(sweep([1000, 1200, 1500]))
    assert len(narrow) == 1 and not narrow[0].reliable
    # A single point: no fit at all.
    assert fit_ptc(sweep([1000])) == []


def test_chart_points_are_excluded_from_fits():
    points = sweep([100, 500, 2000, 9000])
    for p in points:
        p.source = 'chart'
    assert fit_ptc(points) == []


def test_spatial_stats_recover_prnu_under_shading():
    # High signal + deep burst so the fixed pattern dominates the temporal residual.
    fs = synth_burst(64, 20000.0, prnu_pct=2.0, shading=0.3, rng=np.random.default_rng(5))
    ss = spatial_stats(fs)
    assert ss.nonuniformity_pct == pytest.approx(2.0, rel=0.10)


def test_spatial_stats_recover_dsnu_on_darks():
    fs = synth_burst(64, 0.0, dsnu_dn=10.0, rng=np.random.default_rng(6))
    ss = spatial_stats(fs)
    assert ss.residual_std_dn == pytest.approx(10.0, rel=0.10)
    assert ss.nonuniformity_pct is None  # no signal: a percentage is meaningless


def test_shading_fit_removes_a_quadratic_bowl():
    h, w = 64, 64
    y, x = np.mgrid[0:h, 0:w]
    surface = 1000.0 + 200.0 * (x / 63 - 0.5) ** 2 + 150.0 * (y / 63 - 0.5) * (x / 63 - 0.5)
    residual = surface - shading_fit(surface)
    assert float(np.abs(residual).max()) < 1e-6


def test_linearity_and_full_well_and_snr():
    rng = np.random.default_rng(9)
    points = sweep([100, 500, 2000, 9000, 20000], rng=rng)
    lin = linearity(points)
    assert lin['r2'] > 0.999
    assert full_well(points) is None  # nothing clipped: full well not observable
    saturated = sweep([60000], rng=rng)
    fw = full_well(points + saturated)
    assert fw is not None and fw['full_well_dn'] > 0.9 * (65535 - BLACK_16)
    curve = snr_curve(points)
    assert len(curve) == 5 and curve[-1]['snr_db'] > curve[0]['snr_db']


def test_mono_frameset_flows_through():
    fs = synth_burst(8, 3000.0, rng=np.random.default_rng(12))
    fs.channel = 'Y'
    assert temporal_stats(fs) is not None and ptc_point(fs) is not None
