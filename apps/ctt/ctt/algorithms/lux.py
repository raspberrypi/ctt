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

    def run(self) -> dict | None:
        cam = self.camera

        cam.log_new_sec('LUX')

        luxes = [img.lux for img in cam.imgs]
        argmax = luxes.index(min(luxes, key=lambda lx: abs(1000 - lx)))
        img = cam.imgs[argmax]
        cam.log += f'\nLux found closest to 1000: {img.lux} lx'
        cam.log += f'\nImage used: {img.name}'
        if img.lux < 50:
            cam.log += '\nWARNING: Low lux could cause inaccurate calibrations!'

        lux_out, shutter_speed, gain = lux(cam, img)

        cam.log += '\nLUX calibrations written to json file'
        return {
            'reference_shutter_speed': shutter_speed,
            'reference_gain': gain,
            'reference_lux': img.lux,
            'reference_Y': lux_out,
        }


def lux(cam: Camera, img: Image) -> tuple[int, int, float]:
    shutter_speed = img.exposure
    gain = img.againQ8_norm
    aperture = 1
    cam.log += f'\nShutter speed = {shutter_speed}'
    cam.log += f'\nGain = {gain}'
    cam.log += f'\nAperture = {aperture}'
    patches = [img.patches[i] for i in img.order]
    channels = [img.channels[i] for i in img.order]
    return lux_calc(cam, img, patches, channels), shutter_speed, gain


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
