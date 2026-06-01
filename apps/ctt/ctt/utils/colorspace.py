# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# RGB to LAB color space conversion using the colour-science package.

from __future__ import annotations

import colour
import numpy as np


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
