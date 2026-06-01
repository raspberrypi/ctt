# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# ALSC patch extraction (shared by AWB + CCM)

from __future__ import annotations

from bisect import bisect_left
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.image import Image


def get_alsc_patches(
    img: Image, colour_cals: dict | None, grey: bool = True, grid_size: tuple[int, int] = (16, 12)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get patch centre coordinates, image colour and patch values per channel (blacklevel subtracted).
    If grey then only greyscale patches (every 4th) are considered.
    """
    grid_w, grid_h = grid_size
    if grey:
        cen_coords = img.cen_coords[3::4]
        col = img.col
        patches = [np.array(img.patches[i]) for i in img.order]
        r_patches = patches[0][3::4] - img.blacklevel_16
        b_patches = patches[3][3::4] - img.blacklevel_16
        g_patches = (patches[1][3::4] + patches[2][3::4]) / 2 - img.blacklevel_16  # Two green chans averaged
    else:
        cen_coords = img.cen_coords
        col = img.col
        patches = [np.array(img.patches[i]) for i in img.order]
        r_patches = patches[0] - img.blacklevel_16
        b_patches = patches[3] - img.blacklevel_16
        g_patches = (patches[1] + patches[2]) / 2 - img.blacklevel_16

    if colour_cals is None:
        return r_patches, b_patches, g_patches
    # Find where image colour fits in ALSC colour calibration tables.
    cts = list(colour_cals.keys())
    pos = bisect_left(cts, col)
    if pos % len(cts) == 0:
        # Image CT below min or above max ALSC calibration; use extreme closest to image colour.
        col_tabs = np.array(colour_cals[cts[-pos // len(cts)]])
    else:
        # Linear interpolation between existing ALSC colour calibration tables.
        bef = cts[pos - 1]
        aft = cts[pos]
        da = col - bef
        db = aft - col
        bef_tabs = np.array(colour_cals[bef])
        aft_tabs = np.array(colour_cals[aft])
        col_tabs = (bef_tabs * db + aft_tabs * da) / (da + db)
    col_tabs = np.reshape(col_tabs, (2, grid_h, grid_w))
    # dx, dy as used when calculating the ALSC table.
    w, h = img.w / 2, img.h / 2
    dx, dy = int(-(-(w - 1) // grid_w)), int(-(-(h - 1) // grid_h))
    # For each patch centre, pick the (r_gain, b_gain) from the ALSC table at that cell.
    patch_gains = []
    for cen in cen_coords:
        x, y = cen[0] // dx, cen[1] // dy
        col_gains = (col_tabs[0][y][x], col_tabs[1][y][x])
        patch_gains.append(col_gains)

    # Multiply r and b in each patch by the respective gain (ALSC colour correction).
    for i, gains in enumerate(patch_gains):
        r_patches[i] = r_patches[i] * gains[0]
        b_patches[i] = b_patches[i] * gains[1]

    return r_patches, b_patches, g_patches
