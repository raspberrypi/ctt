# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# RGB to LAB color space conversion using the colour-science package.

from __future__ import annotations

from functools import lru_cache

import colour
import numpy as np


def _xy_to_uv(xy: np.ndarray) -> np.ndarray:
    """CIE xy -> 1976 u', v'. Accepts (2,) or (..., 2), returns matching shape."""
    xy = np.asarray(xy, dtype=float)
    x, y = xy[..., 0], xy[..., 1]
    denom = -2 * x + 12 * y + 3
    return np.stack([4 * x / denom, 9 * y / denom], axis=-1)


def rgb_to_lab(rgb: list[float] | np.ndarray, scale: float = 255.0) -> np.ndarray:
    """
    Convert linear sRGB to CIE L*a*b* (D65).
    Values are divided by *scale* to normalise to 0–1 before conversion.
    Accepts a single (3,) triple or an (N, 3) batch.
    """
    rgb = np.atleast_1d(np.asarray(rgb, dtype=float)) / scale
    # Linear sRGB -> XYZ (no CCTF decoding; input is already linear)
    xyz = colour.RGB_to_XYZ(
        rgb,
        colourspace='sRGB',
        apply_cctf_decoding=False,
    )
    return np.asarray(colour.XYZ_to_Lab(xyz))


def rgb_to_uv(rgb: list[float] | np.ndarray) -> np.ndarray:
    """Convert linear sRGB to CIE 1976 u', v' chromaticity (D65).

    Chromaticity is scale-invariant, so no normalisation is needed (any positive
    scaling of the RGB gives the same u', v'). Accepts a (3,) triple or an (N, 3)
    batch and returns matching (..., 2). Non-positive/black inputs yield NaN, which
    the metrics serialiser turns into null.
    """
    rgb = np.clip(np.atleast_1d(np.asarray(rgb, dtype=float)), 0, None)
    xyz = colour.RGB_to_XYZ(rgb, colourspace='sRGB', apply_cctf_decoding=False)
    return _xy_to_uv(np.asarray(colour.XYZ_to_xy(xyz)))


@lru_cache(maxsize=1)
def gamut_reference() -> dict:
    """Reference geometry for a CIE 1976 u'v' gamut diagram, sourced from colour.

    These are fixed standards/physical constants — the sRGB/Rec.709 and Rec.2020
    primary triangles, the D65 white point, and the spectral locus (CIE 1931 2°,
    400-700 nm) — provided so the web UI can draw the diagram from a single source
    of truth rather than its own hard-coded copies. Values rounded to 4 d.p.
    """
    srgb = colour.RGB_COLOURSPACES['sRGB']
    rec2020 = colour.RGB_COLOURSPACES['ITU-R BT.2020']
    cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    locus_xyz = np.array([cmfs[int(w)] for w in range(400, 701, 10)])

    def uv(values: np.ndarray) -> list:
        return np.round(_xy_to_uv(np.asarray(values)), 4).tolist()

    return {
        'srgb': uv(srgb.primaries),  # R, G, B (Rec.709 shares these)
        'rec2020': uv(rec2020.primaries),
        'd65': uv(srgb.whitepoint),
        'locus': uv(colour.XYZ_to_xy(locus_xyz)),
    }
