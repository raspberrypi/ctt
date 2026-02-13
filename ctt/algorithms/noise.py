# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Noise calibration

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from .base import CalibrationAlgorithm

if TYPE_CHECKING:
    from ..core.camera import Camera
    from ..core.image import Image

logger = logging.getLogger(__name__)


class NoiseCalibration(CalibrationAlgorithm):
    json_key = 'rpi.noise'

    def run(self) -> dict | None:
        cam = self.camera

        cam.log_new_sec('NOISE')

        do_plot = self.json_key in getattr(cam, 'plot', [])
        noise_out = sorted([noise(cam, img, plot=do_plot) for img in cam.imgs], key=lambda x: x[0])
        cam.log += '\nFinished processing images'
        length = len(noise_out)
        noise_out = np.mean(noise_out[length // 4 : 1 + 3 * length // 4], axis=0)
        cam.log += f'\nAverage noise profile: constant = {int(noise_out[1])} '
        cam.log += f'slope = {noise_out[0]:.3f}'
        cam.log += '\nNOISE calibrations written to json'
        return {
            'reference_constant': int(noise_out[1]),
            'reference_slope': round(1.4 * noise_out[0], 3),
        }


def noise(cam: Camera, img: Image, plot: bool = False) -> list:
    cam.log += f'\nProcessing image: {img.name}'
    all_patches = np.array([p for ch in img.patches for p in ch], dtype=float)
    all_patches = (all_patches - img.blacklevel_16) / img.againQ8_norm
    stds = np.std(all_patches, axis=1)
    means = np.clip(np.mean(all_patches, axis=1), 0, None)
    sq_means = np.sqrt(means)

    fit = np.polyfit(sq_means, stds, 1)
    cam.log += f'\nBlack level = {img.blacklevel_16}'
    cam.log += f'\nNoise profile: offset = {int(fit[1])}'
    cam.log += f' slope = {fit[0]:.3f}'
    fit_score = np.abs(stds - fit[0] * sq_means - fit[1])
    fit_std = np.std(stds)
    fit_score_norm = fit_score - fit_std
    anom_ind = np.where(fit_score_norm > 1)[0]
    fit_score_norm.sort()
    sq_means_clean = np.delete(sq_means, anom_ind)
    stds_clean = np.delete(stds, anom_ind)
    removed = len(stds) - len(stds_clean)
    if removed != 0:
        cam.log += f'\nIdentified and removed {removed} anomalies.'
        cam.log += '\nRecalculating fit'
        fit = np.polyfit(sq_means_clean, stds_clean, 1)
        cam.log += f'\nNoise profile: offset = {int(fit[1])}'
        cam.log += f' slope = {fit[0]:.3f}'

    corrected = 0
    fit2 = None
    if fit[1] < 0:
        corrected = 1
        ones = np.ones(len(means))
        y_data = stds / sq_means
        fit2 = np.polyfit(ones, y_data, 0)
        cam.log += '\nOffset below zero. Fit recalculated with zero offset'
        cam.log += '\nNoise profile: offset = 0'
        cam.log += f' slope = {fit2[0]:.3f}'

    if plot:
        x = np.arange(sq_means.max() // 0.88 + 1)
        fit_plot = x * fit[0] + fit[1]
        plt.scatter(sq_means, stds, label='data', color='blue')
        if len(anom_ind) > 0:
            plt.scatter(sq_means[anom_ind], stds[anom_ind], color='orange', label='anomalies')
        plt.plot(x, fit_plot, label='fit', color='red', ls=':')
        if fit2 is not None:
            fit_plot_2 = x * fit2[0]
            plt.plot(x, fit_plot_2, label='fit 0 intercept', color='green', ls='--')
        plt.title(f'Noise Plot — {img.name}')
        plt.legend(loc='upper left')
        plt.xlabel('Sqrt Pixel Value')
        plt.ylabel('Noise Standard Deviation')
        plt.grid()
        plt.show()

    cam.log += '\n'
    if corrected and fit2 is not None:
        fit = [fit2[0], 0]
        return fit

    return fit
