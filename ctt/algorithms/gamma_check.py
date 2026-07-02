# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Gamma / tone-curve verification.
#
# Diagnostic only: this does NOT tune rpi.contrast.gamma_curve (that curve is a
# fixed creative/standard transfer function carried by the template). It measures
# how closely the achieved tone response matches a target transfer function and
# records the result in cam.metrics['gamma'] for the Results page. Three checks:
#
#   1. Sensor linearity  - measured signal on the Macbeth greyscale row against
#                          the patches' known reflectance (a sensor diagnostic).
#   2. Curve vs target   - the template gamma_curve against the target OETF
#                          (sRGB / Rec.709 / Rec.2020 / power law) - no image.
#   3. End-to-end tone    - the measured greys pushed through the actual gamma
#                          curve, compared to the target's encoding of the same
#                          reflectance. Checks 1 + 2 explain any deviation here.

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import colour.models
import numpy as np

from .base import CalibrationAlgorithm
from .ccm import MACBETH_RGB

if TYPE_CHECKING:
    from ..core.camera import Camera
    from ..core.image import Image

logger = logging.getLogger(__name__)

# The Macbeth greyscale row: the last six reference patches, white 9.5 -> black 2.
# In CTT's column-major patch order these sit at indices 3, 7, 11, 15, 19, 23
# (i.e. patches[3::4]), matching how lux.py and ccm.py read the greys.
_NEUTRAL_SRGB = np.mean(np.array(MACBETH_RGB[18:24], dtype=float), axis=1)  # (6,) 0-255, white first


def target_oetf(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Map a target name to its opto-electronic transfer function (linear 0-1 -> code 0-1)."""
    if name == 'rec709':
        return colour.models.oetf_BT709
    if name == 'rec2020':
        return colour.models.oetf_BT2020
    if name.startswith('power:'):
        n = float(name.split(':', 1)[1])
        return lambda x: np.clip(np.asarray(x, dtype=float), 0, None) ** (1.0 / n)
    return colour.models.eotf_inverse_sRGB  # 'sRGB' (default)


def _grey_patch_means(channel_patches: list, bl: float) -> np.ndarray:
    """Black-subtracted mean of each greyscale patch in one channel (white first)."""
    return np.array([np.mean(np.clip(np.asarray(p, dtype=float) - bl, 0, None)) for p in channel_patches[3::4]])


def _grey_luminance(img: Image) -> np.ndarray | None:
    """Balanced luminance on the six neutral patches, or None if not measurable."""
    if img.patches is None:
        return None
    patches = [img.patches[i] for i in img.order]
    bl = img.blacklevel_16
    r = _grey_patch_means(patches[0], bl)
    g = (_grey_patch_means(patches[1], bl) + _grey_patch_means(patches[2], bl)) / 2
    b = _grey_patch_means(patches[3], bl)
    # Balance the greys (ratio-of-sums, robust to a near-zero channel) so luminance
    # is neutral; an unbalanced channel would otherwise tilt the measured tone curve.
    sr, sg, sb = np.sum(r), np.sum(g), np.sum(b)
    if sg <= 0 or sr <= 0 or sb <= 0:
        return None
    y = 0.299 * (r * sg / sr) + 0.587 * g + 0.114 * (b * sg / sb)
    if y[0] <= 0:  # white patch must carry signal to normalise against
        return None
    return y


class GammaCheck(CalibrationAlgorithm):
    # Synthetic key: this reads rpi.contrast for the curve but never writes any
    # tuning, and run() always returns None, so nothing is added to the output.
    # A dedicated key keeps it running even when rpi.contrast itself is disabled
    # (the sensor-linearity check is still useful) and labels the progress line.
    json_key = 'rpi.gamma_check'

    def __init__(self, camera: Camera, platform: object, target: str = 'sRGB') -> None:
        super().__init__(camera, platform)
        self.target = target

    def run(self) -> dict | None:
        cam = self.camera
        cam.log_new_sec('GAMMA')

        oetf = target_oetf(self.target)
        refl = colour.models.eotf_sRGB(_NEUTRAL_SRGB / 255)  # scene reflectance, white first
        metrics: dict = {'target': self.target}

        # --- Check 2: the template gamma curve against the target OETF ----------
        gc = cam.json.get('rpi.contrast', {}).get('gamma_curve')
        if gc and len(gc) >= 4:
            xs = np.asarray(gc[0::2], dtype=float) / 65535
            ys = np.asarray(gc[1::2], dtype=float) / 65535
            tgt = np.asarray(oetf(xs), dtype=float)
            dev = (ys - tgt) * 255
            metrics['curve'] = {
                'points': [
                    {'x': float(x), 'measured': float(y), 'target': float(t)}
                    for x, y, t in zip(xs, ys, tgt, strict=True)
                ],
                'max_dev_8bit': float(np.max(np.abs(dev))),
                'rms_dev_8bit': float(np.sqrt(np.mean(dev**2))),
            }
            curve = metrics['curve']
            cam.log += f'\nGamma curve vs {self.target}: '
            cam.log += f'RMS {curve["rms_dev_8bit"]:.2f} / max {curve["max_dev_8bit"]:.2f} (8-bit code)'
        else:
            cam.log += '\nNo rpi.contrast.gamma_curve found; skipping curve comparison'

        # --- Checks 1 & 3: measured greys (skip cleanly if no Macbeth captures) --
        ys_per_img = [y for y in (_grey_luminance(img) for img in cam.imgs) if y is not None]
        if ys_per_img:
            self._measure(cam, metrics, ys_per_img, refl, gc, oetf)
        else:
            cam.log += '\nNo greyscale patch measurements available; tone/linearity checks skipped'

        cam.metrics['gamma'] = metrics
        cam.log += '\nGamma verification written to metrics'
        return None  # diagnostic only: never modifies the tuning

    def _measure(self, cam, metrics, ys_per_img, refl, gc, oetf) -> None:
        # Linearity (check 1): fit signal = a * reflectance through the origin per
        # image; report mean R^2 and the residual of each patch as % of full white.
        r2s, residuals, signal_norms = [], [], []
        for y in ys_per_img:
            a = float(np.sum(y * refl) / np.sum(refl**2))
            pred = a * refl
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1.0
            r2s.append(1 - ss_res / ss_tot)
            residuals.append((y - pred) / (a * refl[0]) * 100)
            signal_norms.append(y / y[0])
        residual = np.mean(residuals, axis=0)
        signal_norm = np.mean(signal_norms, axis=0)
        metrics['linearity'] = {
            'r2': float(np.mean(r2s)),
            'patches': [
                {'reflectance': float(rf), 'signal_norm': float(sn), 'residual_pct': float(rp)}
                for rf, sn, rp in zip(refl, signal_norm, residual, strict=True)
            ],
        }
        cam.log += f'\nSensor linearity on greys: R2 = {metrics["linearity"]["r2"]:.4f}'

        if not (gc and len(gc) >= 4):
            return  # end-to-end tone needs the actual curve

        # End-to-end tone (check 3): each capture's greys, normalised so white = 1,
        # pushed through the actual gamma curve, against the target's encoding of the
        # same reflectance ratio. White is 1:1 by construction, so deviation reveals
        # shadow/mid-tone behaviour (sensor non-linearity + curve mismatch combined).
        xc = np.asarray(gc[0::2], dtype=float) / 65535
        yc = np.asarray(gc[1::2], dtype=float) / 65535
        refl_ratio = refl / refl[0]
        out_tgt = np.asarray(oetf(refl_ratio), dtype=float) * 255

        inputs, outs = [], []
        for y in ys_per_img:
            xn = np.clip(y / y[0], 0, 1)
            inputs.append(xn)
            outs.append(np.interp(xn, xc, yc) * 255)
        xn = np.mean(inputs, axis=0)
        out_meas = np.mean(outs, axis=0)
        err = out_meas - out_tgt
        metrics['tone'] = {
            'rms_err_8bit': float(np.sqrt(np.mean(err**2))),
            'max_err_8bit': float(np.max(np.abs(err))),
            # 'reflectance' is the patch's scene reflectance relative to white; the
            # Results page uses it to recompute the target output for a different
            # transfer function without re-running the calibration.
            'patches': [
                {
                    'reflectance': float(rr),
                    'input': float(xi),
                    'measured': float(m),
                    'target': float(t),
                    'err_8bit': float(e),
                }
                for rr, xi, m, t, e in zip(refl_ratio, xn, out_meas, out_tgt, err, strict=True)
            ],
        }
        cam.log += f'\nEnd-to-end tone vs {self.target}: '
        cam.log += f'RMS {metrics["tone"]["rms_err_8bit"]:.2f} / max {metrics["tone"]["max_err_8bit"]:.2f} (8-bit code)'
