# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Lux level calibration

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .base import CalibrationAlgorithm

if TYPE_CHECKING:
    from ..core.camera import Camera
    from ..core.image import Image

logger = logging.getLogger(__name__)


class LuxCalibration(CalibrationAlgorithm):
    json_key = 'rpi.lux'

    def __init__(
        self,
        camera: Camera,
        platform: object,
        reference_target: int = 1000,
        reference_method: str = 'trimmed-mean',
    ) -> None:
        super().__init__(camera, platform)
        # reference_target > 0: anchor Y on the single capture nearest this lux (the
        # long-standing behaviour). 0: derive Y from a robust average across captures,
        # combined via reference_method ('trimmed-mean' or 'median').
        self.reference_target = reference_target
        self.reference_method = reference_method

    def run(self) -> dict | None:
        cam = self.camera

        cam.log_new_sec('LUX')

        target = self.reference_target
        method = self.reference_method

        # Per-capture luminance slope k = Y / (lux * shutter * gain). Y is proportional to
        # that product, so k would be constant for an ideal sensor; how much it varies with
        # colour temperature is the sensor's luminance "spectral response", which we record
        # in the metrics so the Results page can plot it.
        samples = []
        for img in cam.imgs:
            y = lux_calc(cam, img, [img.patches[i] for i in img.order], [img.channels[i] for i in img.order])
            slope = y / (img.lux * img.exposure * img.againQ8_norm)
            samples.append({'name': img.name, 'ct': int(img.col), 'lux': int(img.lux), 'y': y, 'slope': slope})

        # Reference operating point carries the shutter/gain/lux the runtime divides by.
        ref = min(cam.imgs, key=lambda im: abs((target or 1000) - im.lux))
        ref_eg = ref.exposure * ref.againQ8_norm

        if target > 0:
            cam.log += f'\nReference lux target: {target} lx; image: {ref.name} ({ref.lux} lx)'
            if ref.lux < 50:
                cam.log += '\nWARNING: Low lux could cause inaccurate calibrations!'
            reference_Y = next(s['y'] for s in samples if s['name'] == ref.name)
            k = reference_Y / (ref.lux * ref_eg)
        else:
            # Robust slope across well-exposed captures: trimmed mean (drop the extremes)
            # or median, both resisting one odd capture skewing the whole lux scale.
            slopes = sorted(s['slope'] for s in samples if s['lux'] >= 50) or sorted(s['slope'] for s in samples)
            if method == 'median':
                k = float(np.median(slopes))
            else:
                trimmed = slopes[1:-1] if len(slopes) >= 4 else slopes
                k = float(np.mean(trimmed))
            cam.log += f'\nReference lux target: 0 (robust {method}); slope = {k:.6g}'
            reference_Y = int(round(k * ref.lux * ref_eg))

        cam.metrics['lux'] = {
            'reference_target': target,
            'reference_method': 'single' if target > 0 else method,
            'reference_ct': int(ref.col),
            'reference_slope': k,
            'samples': [{'ct': s['ct'], 'lux': s['lux'], 'slope': s['slope']} for s in samples],
        }
        cam.log += f'\nReference Y: {reference_Y}'
        cam.log += '\nLUX calibrations written to json file'
        return {
            'reference_shutter_speed': ref.exposure,
            'reference_gain': ref.againQ8_norm,
            'reference_lux': ref.lux,
            'reference_Y': reference_Y,
        }


def lux_calc(cam: Camera, img: Image, patches: list, channels: list) -> int:
    # Subtract the black level (clipped at zero) so Y is signal luminance. The runtime
    # lux algorithm divides reference_Y into the AGC Y statistic, which is gathered after
    # the ISP's black-level correction; reference_Y must be on the same footing or the
    # lux ~ Y * exposure * gain proportionality (and so every estimate) is biased.
    bl = img.blacklevel_16

    def grey_mean(channel_patches):
        return np.mean(np.clip(np.array(channel_patches[3::4], dtype=float) - bl, 0, None))

    def image_mean(channel):
        return np.mean(np.clip(channel.astype(float) - bl, 0, None))

    ap_r = grey_mean(patches[0])
    ap_g = (grey_mean(patches[1]) + grey_mean(patches[2])) / 2
    ap_b = grey_mean(patches[3])
    cam.log += '\nAverage channel values on grey patches (black subtracted):'
    cam.log += f'\nRed = {ap_r:.0f} Green = {ap_g:.0f} Blue = {ap_b:.0f}'
    gr = ap_g / ap_r
    gb = ap_g / ap_b
    cam.log += f'\nChannel gains: Red = {gr:.3f} Blue = {gb:.3f}'

    a_r = image_mean(channels[0]) * gr
    a_g = (image_mean(channels[1]) + image_mean(channels[2])) / 2
    a_b = image_mean(channels[3]) * gb
    cam.log += '\nAverage channel values over entire image scaled by channel gains (black subtracted):'
    cam.log += f'\nRed = {a_r:.0f} Green = {a_g:.0f} Blue = {a_b:.0f}'
    y = 0.299 * a_r + 0.587 * a_g + 0.114 * a_b
    cam.log += f'\nY value calculated: {int(y)}'
    return int(y)
