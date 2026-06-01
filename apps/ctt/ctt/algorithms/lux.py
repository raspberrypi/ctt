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
    ap_r = np.mean(patches[0][3::4])
    ap_g = (np.mean(patches[1][3::4]) + np.mean(patches[2][3::4])) / 2
    ap_b = np.mean(patches[3][3::4])
    cam.log += '\nAverage channel values on grey patches:'
    cam.log += f'\nRed = {ap_r:.0f} Green = {ap_g:.0f} Blue = {ap_b:.0f}'
    gr = ap_g / ap_r
    gb = ap_g / ap_b
    cam.log += f'\nChannel gains: Red = {gr:.3f} Blue = {gb:.3f}'

    a_r = np.mean(channels[0]) * gr
    a_g = (np.mean(channels[1]) + np.mean(channels[2])) / 2
    a_b = np.mean(channels[3]) * gb
    cam.log += '\nAverage channel values over entire image scaled by channel gains:'
    cam.log += f'\nRed = {a_r:.0f} Green = {a_g:.0f} Blue = {a_b:.0f}'
    y = 0.299 * a_r + 0.587 * a_g + 0.114 * a_b
    cam.log += f'\nY value calculated: {int(y)}'
    return int(y)
