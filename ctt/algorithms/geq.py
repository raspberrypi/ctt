# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# GEQ (green equalisation) calibration

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize

from .base import CalibrationAlgorithm

if TYPE_CHECKING:
    from ..core.camera import Camera
    from ..core.image import Image

logger = logging.getLogger(__name__)


class GeqCalibration(CalibrationAlgorithm):
    json_key = 'rpi.geq'

    def run(self) -> dict | None:
        cam = self.camera

        cam.log_new_sec('GEQ')

        do_plot = self.json_key in getattr(cam, 'plot', [])
        slope, offset = geq_fit(cam, plot=do_plot)

        cam.log += '\nGEQ calibrations written to json file'
        return {
            'offset': offset,
            'slope': slope,
        }


def geq_fit(cam: Camera, plot: bool = False) -> tuple[float, int]:
    imgs = cam.imgs
    geqs = np.array([geq(cam, img) * img.againQ8_norm for img in imgs])
    cam.log += '\nProcessed all images'
    geqs = geqs.reshape((-1, 2))
    geqs = np.array(sorted(geqs, key=lambda r: np.abs((r[1] - r[0]) / r[0])))

    length = len(geqs)
    g0 = geqs[length // 2 :, 0]
    g1 = geqs[length // 2 :, 1]
    gdiff = np.abs(g0 - g1)

    def f(params):
        m, c = params
        a = gdiff - (m * g0 + c)
        return np.sum(a**2 + 0.95 * np.abs(a) * a)

    initial_guess = [0.01, 500]
    result = optimize.minimize(f, initial_guess, method='Nelder-Mead')
    if result.success:
        slope, offset = result.x
        cam.log += f'\nFit result: slope = {slope:.5f} '
        cam.log += f'offset = {int(offset)}'

        if plot:
            x = np.linspace(max(g0) * 1.1, 100)
            y = slope * x + offset
            plt.title("GEQ Asymmetric 'Upper Bound' Fit")
            plt.plot(x, y, color='red', ls='--', label='fit')
            plt.scatter(g0, gdiff, color='b', label='data')
            plt.ylabel('Difference in green channels')
            plt.xlabel('Green value')

        slope *= 1.5
        offset += 201
        cam.log += f'\nFit after correction factors: slope = {slope:.5f}'
        cam.log += f' offset = {int(offset)}'
        if offset < 0:
            cam.log += '\nOffset raised to 0'
            offset = 0

        if plot:
            y2 = slope * x + offset
            plt.plot(x, y2, color='green', ls='--', label='scaled fit')
            plt.grid()
            plt.legend()
            plt.show()

    else:
        logger.error("\nError! Couldn't fit asymmetric least squares")
        logger.error(result.message)
        cam.log += '\nWARNING: Asymmetric least squares fit failed! '
        cam.log += 'Standard fit used could possibly lead to worse results'
        fit = np.polyfit(gdiff, g0, 1)
        offset, slope = -fit[1] / fit[0], 1 / fit[0]
        cam.log += f'\nFit result: slope = {slope:.5f} '
        cam.log += f'offset = {int(offset)}'

        if plot:
            x = np.linspace(max(g0) * 1.1, 100)
            y = slope * x + offset
            plt.title('GEQ Linear Fit')
            plt.plot(x, y, color='red', ls='--', label='fit')
            plt.scatter(g0, gdiff, color='b', label='data')
            plt.ylabel('Difference in green channels')
            plt.xlabel('Green value')

        slope *= 2.5
        offset += 301
        cam.log += f'\nFit after correction factors: slope = {slope:.5f}'
        cam.log += f' offset = {int(offset)}'

        if offset < 0:
            cam.log += '\nOffset raised to 0'
            offset = 0

        if plot:
            y2 = slope * x + offset
            plt.plot(x, y2, color='green', ls='--', label='scaled fit')
            plt.legend()
            plt.grid()
            plt.show()

    return round(slope, 5), int(offset)


def geq(cam: Camera, img: Image) -> np.ndarray:
    cam.log += f'\nProcessing image {img.name}'
    patches = [img.patches[i] for i in img.order][1:3]
    g_patches = np.array([(np.mean(patches[0][i]), np.mean(patches[1][i])) for i in range(24)])
    cam.log += '\n'
    return g_patches
