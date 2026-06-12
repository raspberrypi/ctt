# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for ALSC post-correction residual prediction.

import numpy as np

from ctt.algorithms.alsc import alsc_residuals, get_grid

GRID = (16, 12)


def test_get_grid_uint16_matches_float_reference():
    # Channels are stored as uint16; cell sums far exceed the uint16 range, so
    # get_grid must accumulate in a wider dtype.
    rng = np.random.default_rng(7)
    chan = rng.integers(30000, 65536, (1232, 1640), dtype=np.uint16)
    grid_w, grid_h = GRID
    dx, dy = int((1640 - 1) // (grid_w - 1)), int((1232 - 1) // (grid_h - 1))
    out = get_grid(chan, dx, dy, GRID)
    ref = get_grid(chan.astype(np.float64), dx, dy, GRID)
    np.testing.assert_allclose(out, ref)


def _vignetted(corner_level: float) -> np.ndarray:
    """A radial flat-field: 1.0 at the exact grid centre, corner_level at the corners."""
    grid_w, grid_h = GRID
    cc, rr = np.meshgrid(np.arange(grid_w, dtype=float), np.arange(grid_h, dtype=float))
    # The centre index used by the residual maths is the 2x2 block at
    # (grid_h//2-1, grid_w//2-1); centre the cone on that block's middle.
    cy, cx = grid_h / 2 - 0.5, grid_w / 2 - 0.5
    radius = np.hypot((cc - cx) / (grid_w / 2), (rr - cy) / (grid_h / 2))
    radius /= radius.max()
    return (1.0 - (1.0 - corner_level) * radius**2).flatten()


def test_full_strength_correction_leaves_no_residual():
    measured = _vignetted(0.5)
    cg = 1.0 / measured  # the table CTT derives from this image (min = 1 at centre)
    lut = list(cg)
    out = alsc_residuals(np.array([cg]), np.array([5000]), ['a.dng'], lut, GRID, luminance_strength=1.0)
    assert out[0]['ct'] == 5000 and out[0]['name'] == 'a.dng'
    assert out[0]['corner_pct'] == 0.0
    assert out[0]['worst_pct'] <= 0.1  # centre 2x2 average vs exact centre: tiny


def test_partial_strength_leaves_designed_residual():
    # Corner at 0.5 with strength 0.8: gain = 1 + 0.8*(2-1) = 1.8, corrected
    # corner = 0.5 * 1.8 = 0.9 -> ~10% below the (fully corrected) centre.
    measured = _vignetted(0.5)
    cg = 1.0 / measured
    out = alsc_residuals(np.array([cg]), np.array([5000]), ['a.dng'], list(cg), GRID, luminance_strength=0.8)
    assert abs(out[0]['corner_pct'] - 10.0) < 0.5


def test_shared_lut_costs_show_per_ct():
    # Two illuminants with different vignetting share one averaged LUT: each
    # ends up with a residual even at strength 1, in opposite directions.
    m1, m2 = _vignetted(0.5), _vignetted(0.7)
    cg1, cg2 = 1.0 / m1, 1.0 / m2
    lut = list((cg1 + cg2) / 2)
    out = alsc_residuals(
        np.array([cg1, cg2]), np.array([3000, 5000]), ['a.dng', 'b.dng'], lut, GRID, luminance_strength=1.0
    )
    # Analytic: lut corner = (2 + 1/0.7)/2 ≈ 1.714, so image 1 lands at
    # 0.5*1.714 ≈ -14.3% and image 2 at 0.7*1.714 ≈ +20%.
    assert abs(out[0]['corner_pct'] - 14.3) < 0.5
    assert abs(out[1]['corner_pct'] - 20.0) < 0.5
