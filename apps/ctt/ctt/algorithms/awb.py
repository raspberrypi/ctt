# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# AWB (auto white balance) calibration

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin

from ..detection.patches import get_alsc_patches
from ..utils.tools import get_alsc_colour_cals, nudge_for_json
from .base import CalibrationAlgorithm

if TYPE_CHECKING:
    from ..core.camera import Camera

logger = logging.getLogger(__name__)


class AwbCalibration(CalibrationAlgorithm):
    json_key = 'rpi.awb'

    def __init__(self, camera: Camera, platform: object, greyworld: bool = False, do_alsc_colour: bool = True) -> None:
        super().__init__(camera, platform)
        self.greyworld = greyworld
        self.do_alsc_colour = do_alsc_colour

    def run(self) -> dict | None:
        cam = self.camera
        grid_size = self.platform.grid_size

        cam.log_new_sec('AWB')

        if cam.mono:
            logger.error("\nERROR: Can't do AWB on greyscale image!")
            cam.log += '\nERROR: Cannot perform AWB calibration '
            cam.log += 'on greyscale image!\nAWB aborted!'
            del cam.json['rpi.awb']
            return None

        result = {}
        if self.greyworld:
            result['bayes'] = 0
            cam.log += '\nGreyworld set'

        if ('rpi.alsc' not in cam.disable) and self.do_alsc_colour:
            colour_cals = get_alsc_colour_cals(cam.json)
            if colour_cals is not None:
                cam.log += '\nALSC tables found successfully'
            else:
                logger.error('ERROR, no ALSC calibrations found for AWB')
                logger.info('Performing AWB without ALSC tables')
                cam.log += '\nWARNING: No ALSC tables found.\nAWB calibration '
                cam.log += 'performed without ALSC correction...'
        else:
            colour_cals = None
            cam.log += '\nWARNING: No ALSC tables found.\nAWB calibration '
            cam.log += 'performed without ALSC correction...'

        do_plot = self.json_key in getattr(cam, 'plot', [])
        awb_out = awb(cam, colour_cals, grid_size, plot=do_plot)
        ct_curve, transverse_neg, transverse_pos = awb_out

        # Clip AWB mode bounds to the measured CT range.
        ct_min = ct_curve[0]  # ct_curve is [ct, r, b, ct, r, b, ...] increasing CT
        ct_max = ct_curve[-3]
        modes = cam.json.get('rpi.awb', {}).get('modes', {})
        for name, mode in modes.items():
            lo = max(mode['lo'], ct_min)
            hi = min(mode['hi'], ct_max)
            if lo > hi:
                lo = hi = ct_min if mode['hi'] < ct_min else ct_max
            if lo != mode['lo'] or hi != mode['hi']:
                cam.log += f'\nAWB mode {name}: clipped [{mode["lo"]}, {mode["hi"]}] -> [{lo}, {hi}]'
                mode['lo'] = lo
                mode['hi'] = hi

        result['ct_curve'] = ct_curve
        result['sensitivity_r'] = 1.0
        result['sensitivity_b'] = 1.0
        result['transverse_pos'] = transverse_pos
        result['transverse_neg'] = transverse_neg
        cam.log += '\nAWB calibration written to json file'
        return result


def awb(
    cam: Camera,
    colour_cals: dict | None,
    grid_size: tuple[int, int],
    plot: bool = False,
) -> tuple:
    """Obtain piecewise linear approximation for the colour curve."""
    imgs = cam.imgs
    # Obtain data from greyscale macbeth patches.
    rb_raw = []
    rbs_hat = []
    for img in imgs:
        cam.log += f'\nProcessing {img.name}'
        # Get greyscale patches with ALSC applied if enabled; if disabled colour_cals is None.
        r_patches, b_patches, g_patches = get_alsc_patches(img, colour_cals, grid_size=grid_size)
        # Calculate ratio of r, b to g.
        r_g = np.mean(r_patches / g_patches)
        b_g = np.mean(b_patches / g_patches)
        cam.log += f'\n       r : {r_g:.4f}       b : {b_g:.4f}'
        r_g_hat = r_g / (1 + r_g + b_g)
        b_g_hat = b_g / (1 + r_g + b_g)
        cam.log += f'\n   r_hat : {r_g_hat:.4f}   b_hat : {b_g_hat:.4f}'
        # Curve in so-called hatspace: r_hat = R/(R+B+G), b_hat = B/(R+B+G). Dehat: r = R/G, b = B/G.
        rbs_hat.append((r_g_hat, b_g_hat, img.col))
        rb_raw.append((r_g, b_g))
        cam.log += '\n'

    cam.log += '\nFinished processing images'
    # Sort all lists simultaneously by r_hat.
    rbs_zip = list(zip(rbs_hat, rb_raw, strict=True))
    rbs_zip.sort(key=lambda x: x[0][0])
    rbs_hat, rb_raw = list(zip(*rbs_zip, strict=True))
    rbs_hat = list(zip(*rbs_hat, strict=True))
    rb_raw = list(zip(*rb_raw, strict=True))
    # Fit quadratic to r_hat vs b_hat.
    a, b, c = np.polyfit(rbs_hat[0], rbs_hat[1], 2)
    cam.log += '\nFit quadratic curve in hatspace'

    # Approximate shortest distance from each point to the curve in dehatspace. Distance is used for:
    # 1) If CT does not strictly decrease with r/g, choose closest point from an increasing pair.
    # 2) Transverse pos/neg: max positive and negative distance from the line (upper bound).
    def f(x):
        return a * x**2 + b * x + c

    dists = []
    for _i, (R, B) in enumerate(zip(rbs_hat[0], rbs_hat[1], strict=True)):
        # Minimise squared distance from datapoint to point on curve (monotonic in radius).
        def f_min(x):
            y = f(x)
            return (x - R) ** 2 + (y - B) ** 2  # noqa: B023

        x_hat = fmin(f_min, R, disp=0)[0]
        y_hat = f(x_hat)
        x = x_hat / (1 - x_hat - y_hat)  # Dehat
        y = y_hat / (1 - x_hat - y_hat)
        rr = R / (1 - R - B)
        bb = B / (1 - R - B)
        dist = ((x - rr) ** 2 + (y - bb) ** 2) ** 0.5  # Euclidean distance in dehatspace
        if (x + y) > (rr + bb):
            dist *= -1  # Negative if point is below the fit curve
        dists.append(dist)
    cam.log += '\nFound closest point on fit line to each point in dehatspace'
    # Wiggle factors in AWB; 10% added as this is an upper bound.
    transverse_neg = -np.min(dists) * 1.1
    transverse_pos = np.max(dists) * 1.1
    cam.log += f'\nTransverse pos : {transverse_pos:.5f}'
    cam.log += f'\nTransverse neg : {transverse_neg:.5f}'
    # Minimum transverse wiggles 0.01; wiggle factors set how far off the curve we search.
    if transverse_pos < 0.01:
        transverse_pos = 0.01
        cam.log += '\nForced transverse pos to 0.01'
    if transverse_neg < 0.01:
        transverse_neg = 0.01
        cam.log += '\nForced transverse neg to 0.01'

    # Generate new b_hat at each r_hat from fit; transform from hatspace to dehatspace.
    r_hat_fit = np.array(rbs_hat[0])
    b_hat_fit = a * r_hat_fit**2 + b * r_hat_fit + c
    r_fit = r_hat_fit / (1 - r_hat_fit - b_hat_fit)
    b_fit = b_hat_fit / (1 - r_hat_fit - b_hat_fit)
    c_fit = np.round(rbs_hat[2], 0)
    # Round to 4 dp for storage.
    r_fit = nudge_for_json(r_fit, decimals=4)
    b_fit = nudge_for_json(b_fit, decimals=4)

    # Ensure colour temperature decreases with increasing r/g; iterate backwards for easier indexing.
    i = len(c_fit) - 1
    while i > 0:
        if c_fit[i] > c_fit[i - 1]:
            cam.log += '\nColour temperature increase found\n'
            cam.log += f'{c_fit[i - 1]} K at r = {r_fit[i - 1]} to '
            cam.log += f'{c_fit[i]} K at r = {r_fit[i]}'
            # If CT increases, discard the point furthest from the fit (in dehatspace).
            error_1 = abs(dists[i - 1])
            error_2 = abs(dists[i])
            cam.log += '\nDistances from fit:\n'
            cam.log += f'{c_fit[i]} K : {error_1:.5f} , '
            cam.log += f'{c_fit[i - 1]} K : {error_2:.5f}'
            bad = i - (error_1 < error_2)  # bad index (Python False=0, True=1)
            cam.log += f'\nPoint at {c_fit[bad]} K deleted as '
            cam.log += 'it is furthest from fit'
            r_fit = np.delete(r_fit, bad)
            b_fit = np.delete(b_fit, bad)
            c_fit = np.delete(c_fit, bad).astype(np.uint16)
        # If a point was discarded, length decreased so decrementing i reassesses the kept point.
        i -= 1

    # Return formatted CT curve, ordered by increasing colour temperature.
    ct_curve = list(np.array(list(zip(b_fit, r_fit, c_fit, strict=True))).flatten())[::-1]
    cam.log += '\nFinal CT curve:'
    for i in range(len(ct_curve) // 3):
        j = 3 * i
        cam.log += f'\n  ct: {ct_curve[j]}  '
        cam.log += f'  r: {ct_curve[j + 1]}  '
        cam.log += f'  b: {ct_curve[j + 2]}  '

    if plot:
        x = np.linspace(np.min(rbs_hat[0]), np.max(rbs_hat[0]), 100)
        y = a * x**2 + b * x + c
        plt.subplot(2, 1, 1)
        plt.title('hatspace')
        plt.plot(rbs_hat[0], rbs_hat[1], ls='--', color='blue')
        plt.plot(x, y, color='green', ls='-')
        plt.scatter(rbs_hat[0], rbs_hat[1], color='red')
        for i, ct in enumerate(rbs_hat[2]):
            plt.annotate(str(ct), (rbs_hat[0][i], rbs_hat[1][i]))
        plt.xlabel(r'$\hat{r}$')
        plt.ylabel(r'$\hat{b}$')
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.title('dehatspace')
        plt.plot(r_fit, b_fit, color='blue')
        plt.scatter(rb_raw[0], rb_raw[1], color='green')
        plt.scatter(r_fit, b_fit, color='red')
        for i, ct in enumerate(c_fit):
            plt.annotate(str(int(ct)), (r_fit[i], b_fit[i]))
        plt.xlabel('r')
        plt.ylabel('b')
        plt.grid()
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    return (ct_curve, np.round(transverse_pos, 5), np.round(transverse_neg, 5))
