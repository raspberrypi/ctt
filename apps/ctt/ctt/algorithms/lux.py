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

    def __init__(self, camera: Camera, platform: object, reference_target: int = 1000) -> None:
        super().__init__(camera, platform)
        # reference_target > 0: calibrate Y from the single capture nearest this lux
        # (the long-standing behaviour). reference_target == 0: calibrate Y from a robust
        # average across all captures (less hostage to one image's noise / lux label).
        self.reference_target = reference_target

    def run(self) -> dict | None:
        cam = self.camera

        cam.log_new_sec('LUX')

        target = self.reference_target
        # Reference operating point carries the shutter/gain/lux the runtime divides by.
        ref = min(cam.imgs, key=lambda im: abs((target or 1000) - im.lux))

        if target > 0:
            # Single image nearest the target lux; Y taken straight from it.
            cam.log += f'\nReference lux target: {target} lx'
            cam.log += f'\nImage used (nearest {target} lx): {ref.name} ({ref.lux} lx)'
            if ref.lux < 50:
                cam.log += '\nWARNING: Low lux could cause inaccurate calibrations!'
            reference_Y = lux_calc(cam, ref, [ref.patches[i] for i in ref.order], [ref.channels[i] for i in ref.order])
        else:
            # Robust average. Y is proportional to lux*shutter*gain, so the slope
            # k = Y / (lux*shutter*gain) should agree across captures; average it robustly
            # (trim the extreme slopes) so one odd capture can't skew the whole lux scale.
            cam.log += '\nReference lux target: 0 (robust average across all captures)'
            cam.log += f'\nReference operating point: {ref.name} ({ref.lux} lx)'
            eligible = [im for im in cam.imgs if im.lux >= 50] or list(cam.imgs)
            slopes = []
            for img in eligible:
                y = lux_calc(cam, img, [img.patches[i] for i in img.order], [img.channels[i] for i in img.order])
                slopes.append(y / (img.lux * img.exposure * img.againQ8_norm))
            slopes.sort()
            trimmed = slopes[1:-1] if len(slopes) >= 4 else slopes
            k = float(np.mean(trimmed))
            cam.log += f'\nPer-image lux slopes Y/(lux*exp*gain): {[round(s, 6) for s in slopes]}'
            cam.log += f'\nRobust slope (trimmed mean of {len(trimmed)}/{len(slopes)}): {k:.6g}'
            # runtime lux uses reference_Y / (reference_lux * shutter * gain) == k.
            reference_Y = int(round(k * ref.lux * ref.exposure * ref.againQ8_norm))

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
