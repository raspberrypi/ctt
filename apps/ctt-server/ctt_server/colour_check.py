# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Live colour-accuracy check: measured Macbeth patches vs reference colours.
#
# Operates on an ISP-processed preview frame with the chart located by the
# live detector, so the resulting delta E is the colour accuracy actually
# achieved by the loaded tuning (AWB + CCM + gamma through the real pipeline),
# not the calibration-time simulation. The same Lab/CIE2000 pipeline as the
# CCM calibration is reused so the numbers are directly comparable.

import numpy as np
from scipy.optimize import minimize_scalar

from ctt.algorithms.ccm import MACBETH_RGB, degamma, deltae_array
from ctt.utils.colorspace import rgb_to_lab


def _reference() -> tuple[np.ndarray, np.ndarray]:
    """Reference Lab + sRGB values in patch-detector order (matches ccm.py)."""
    m_rgb = np.array(MACBETH_RGB)
    m_lab = rgb_to_lab(degamma(m_rgb) / 256)
    m_lab = np.array([m_lab[i::6] for i in range(6)]).reshape((24, 3))
    m_rgb = np.array([m_rgb[i::6] for i in range(6)]).reshape((24, 3))
    return m_lab, m_rgb


def patch_means(frame_rgb: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Mean sRGB (0-255) of a window around each detected patch centre.

    The window scales with the chart: a fifth of the median nearest-neighbour
    spacing between centres, so it stays inside the patches at any chart size.
    """
    pts = np.asarray(centres, dtype=float)
    dists = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    np.fill_diagonal(dists, np.inf)
    half = max(2, int(np.median(dists.min(axis=1)) * 0.2))
    h, w = frame_rgb.shape[:2]
    out = []
    for x, y in pts.astype(int):
        x0, x1 = max(x - half, 0), min(x + half + 1, w)
        y0, y1 = max(y - half, 0), min(y + half + 1, h)
        out.append(frame_rgb[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0))
    return np.array(out)


def deltae_report(patch_rgb: np.ndarray) -> dict:
    """Delta E (CIE2000) of 24 measured patches against the Macbeth reference.

    patch_rgb: (24, 3) sRGB-encoded means (0-255) in patch-detector order.
    'colour' removes the overall-brightness offset (optimal global scale on
    linear RGB) exactly like the CCM calibration's de_norm, so exposure
    differences don't read as colour error.
    """
    m_lab, m_rgb = _reference()
    linear = degamma(np.asarray(patch_rgb, dtype=float)) / 256
    de = deltae_array(rgb_to_lab(linear), m_lab)

    def _norm_mean_de(s):
        return float(np.mean(deltae_array(rgb_to_lab(s * linear), m_lab)))

    s_opt = minimize_scalar(_norm_mean_de, bounds=(0.25, 4.0), method='bounded').x
    de_norm = deltae_array(rgb_to_lab(s_opt * linear), m_lab)
    patches = [
        {'de': round(float(d), 2), 'de_norm': round(float(dn), 2), 'rgb': [int(v) for v in ref]}
        for d, dn, ref in zip(de, de_norm, m_rgb, strict=True)
    ]
    return {
        'patches': patches,
        'mean': round(float(de.mean()), 2),
        'colour': round(float(de_norm.mean()), 2),
        'worst': round(float(de.max()), 2),
        'scale': round(float(s_opt), 3),
        'saturated': bool(np.asarray(patch_rgb).max() >= 250),
    }
