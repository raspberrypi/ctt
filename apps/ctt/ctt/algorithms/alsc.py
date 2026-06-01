# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# ALSC (auto lens shading correction) calibration

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from ..utils.tools import nudge_for_json
from .base import CalibrationAlgorithm

if TYPE_CHECKING:
    from ..core.camera import Camera
    from ..core.image import Image

logger = logging.getLogger(__name__)


class AlscCalibration(CalibrationAlgorithm):
    json_key = 'rpi.alsc'

    def __init__(
        self,
        camera: Camera,
        platform: object,
        luminance_strength: float = 0.8,
        do_alsc_colour: bool = True,
        max_gain: float = 8.0,
    ) -> None:
        super().__init__(camera, platform)
        self.luminance_strength = luminance_strength
        self.do_alsc_colour = do_alsc_colour
        self.max_gain = max_gain

    def run(self) -> dict | None:
        cam = self.camera
        grid_size = self.platform.grid_size
        do_alsc_colour = self.do_alsc_colour
        luminance_strength = self.luminance_strength
        max_gain = self.max_gain

        cam.log_new_sec('ALSC')

        if len(cam.imgs_alsc) == 0:
            logger.error('\nError:\nNo alsc calibration images found')
            cam.log += '\nERROR: No ALSC calibration images found!'
            cam.log += '\nALSC calibration aborted!'
            return None

        result = {'luminance_strength': luminance_strength}

        if cam.mono and do_alsc_colour:
            logger.info('Greyscale camera so only luminance_lut calculated')
            do_alsc_colour = False
            cam.log += '\nWARNING: ALSC colour correction cannot be done on '
            cam.log += 'greyscale image!\nALSC colour corrections forced off!'

        do_plot = self.json_key in getattr(cam, 'plot', [])
        cal_cr_list, cal_cb_list, luminance_lut, av_corn = alsc_all(
            cam, do_alsc_colour, grid_size, max_gain=max_gain, plot=do_plot
        )

        if not do_alsc_colour:
            result['luminance_lut'] = luminance_lut
            result['n_iter'] = 0
            cam.log += '\nALSC calibrations written to json file'
            cam.log += '\nNo colour calibrations performed'
            return result

        result['calibrations_Cr'] = cal_cr_list
        result['calibrations_Cb'] = cal_cb_list
        result['luminance_lut'] = luminance_lut
        cam.log += '\nALSC colour and luminance tables written to json file'

        if len(cam.imgs_alsc) == 1:
            result['sigma'] = 0.005
            result['sigma_Cb'] = 0.005
            logger.info('\nWarning:\nOnly one alsc calibration found\nStandard sigmas used for adaptive algorithm.')
            cam.log += '\nWARNING: Only one colour temperature found in '
            cam.log += 'calibration images.\nStandard sigmas used for adaptive '
            cam.log += 'algorithm!'
            return result

        sigma_r, sigma_b = get_sigma(cam, cal_cr_list, cal_cb_list, grid_size)
        result['sigma'] = np.round(sigma_r, 5)
        result['sigma_Cb'] = np.round(sigma_b, 5)
        cam.log += '\nCalibrated sigmas written to json file'
        return result


def alsc_all(
    cam: Camera,
    do_alsc_colour: bool,
    grid_size: tuple[int, int] = (16, 12),
    max_gain: float = 8.0,
    plot: bool = False,
) -> tuple:
    """Perform ALSC calibration on a set of images."""
    imgs_alsc = cam.imgs_alsc
    grid_w, grid_h = grid_size
    # Create list of colour temperatures and associated calibration tables.
    list_col = []
    list_cr = []
    list_cb = []
    list_cg = []
    for img in imgs_alsc:
        col, cr, cb, cg, size = alsc(cam, img, do_alsc_colour, grid_size=grid_size, max_gain=max_gain)
        list_col.append(col)
        list_cr.append(cr)
        list_cb.append(cb)
        list_cg.append(cg)
        cam.log += '\n'
    cam.log += '\nFinished processing images'
    w, h, dx, dy = size
    cam.log += f'\nChannel dimensions: w = {int(w)}  h = {int(h)}'
    cam.log += f'\n16x12 grid rectangle size: w = {dx} h = {dy}'

    # Convert to numpy array for data manipulation.
    list_col = np.array(list_col)
    list_cr = np.array(list_cr)
    list_cb = np.array(list_cb)
    list_cg = np.array(list_cg)

    cal_cr_list = []
    cal_cb_list = []

    # Only do colour calculations if required.
    if do_alsc_colour:
        cam.log += '\nALSC colour tables'
        for ct in sorted(set(list_col)):
            cam.log += f'\nColour temperature: {ct} K'
            # Average tables for the same colour temperature.
            indices = np.where(list_col == ct)
            ct = int(ct)
            t_r = np.mean(list_cr[indices], axis=0)
            t_b = np.mean(list_cb[indices], axis=0)
            # Force numbers to be stored to 3 dp for JSON.
            t_r = nudge_for_json(t_r, decimals=3)
            t_b = nudge_for_json(t_b, decimals=3)
            r_corners = (t_r[0], t_r[grid_w - 1], t_r[-1], t_r[-grid_w])
            b_corners = (t_b[0], t_b[grid_w - 1], t_b[-1], t_b[-grid_w])
            middle_pos = (grid_h // 2 - 1) * grid_w + grid_w - 1
            r_cen = t_r[middle_pos] + t_r[middle_pos + 1] + t_r[middle_pos + grid_w] + t_r[middle_pos + grid_w + 1]
            r_cen = round(r_cen / 4, 3)
            b_cen = t_b[middle_pos] + t_b[middle_pos + 1] + t_b[middle_pos + grid_w] + t_b[middle_pos + grid_w + 1]
            b_cen = round(b_cen / 4, 3)
            cam.log += f'\nRed table corners: {r_corners}'
            cam.log += f'\nRed table centre: {r_cen}'
            cam.log += f'\nBlue table corners: {b_corners}'
            cam.log += f'\nBlue table centre: {b_cen}'
            cr_dict = {'ct': ct, 'table': list(t_r)}
            cb_dict = {'ct': ct, 'table': list(t_b)}
            cal_cr_list.append(cr_dict)
            cal_cb_list.append(cb_dict)
            cam.log += '\n'
    else:
        cal_cr_list, cal_cb_list = None, None

    # Average all values for luminance shading and return one table for all temperatures.
    lum_lut = np.mean(list_cg, axis=0)
    lum_lut = list(nudge_for_json(lum_lut, decimals=3))

    # Calculate average corner for LSC gain calculation further on.
    corners = (lum_lut[0], lum_lut[15], lum_lut[-1], lum_lut[-16])
    cam.log += f'\nLuminance table corners: {corners}'
    l_cen = lum_lut[5 * 16 + 7] + lum_lut[5 * 16 + 8] + lum_lut[6 * 16 + 7] + lum_lut[6 * 16 + 8]
    l_cen = round(l_cen / 4, 3)
    cam.log += f'\nLuminance table centre: {l_cen}'
    av_corn = np.sum(corners) / 4

    if plot and len(imgs_alsc) > 0:
        # Plot first image's grids as 3D surfaces.
        cr = np.reshape(list_cr[0], (grid_h, grid_w))
        cb = np.reshape(list_cb[0], (grid_h, grid_w))
        cg = np.reshape(np.array(list_cg[0]), (grid_h, grid_w))
        X, Y = np.meshgrid(range(grid_w), range(grid_h))
        fig = plt.figure(figsize=(8, 8))
        if do_alsc_colour:
            ax1 = fig.add_subplot(311, projection='3d')
            ax1.plot_surface(X, -Y, cr, cmap=cm.coolwarm, linewidth=0)
            ax1.set_title('ALSC cr')
            ax2 = fig.add_subplot(312, projection='3d')
            ax2.plot_surface(X, -Y, cb, cmap=cm.coolwarm, linewidth=0)
            ax2.set_title('cb')
            ax3 = fig.add_subplot(313, projection='3d')
            ax3.plot_surface(X, -Y, cg, cmap=cm.coolwarm, linewidth=0)
            ax3.set_title('cg')
        else:
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.plot_surface(X, -Y, cg, cmap=cm.coolwarm, linewidth=0)
            ax.set_title('ALSC (luminance only) cg')
        plt.show()

    return cal_cr_list, cal_cb_list, lum_lut, av_corn


def alsc(
    cam: Camera, img: Image, do_alsc_colour: bool, grid_size: tuple[int, int] = (16, 12), max_gain: float = 8.0
) -> tuple:
    """Calculate g/r and g/b for grid points for a single image."""
    cam.log += f'\nProcessing image: {img.name}'
    grid_w, grid_h = grid_size
    # Get channels in correct order.
    channels = [img.channels[i] for i in img.order]
    # Calculate size of single rectangle; divisions ensure final row/column of cells has non-zero pixels.
    w, h = img.w / 2, img.h / 2
    dx, dy = int((w - 1) // (grid_w - 1)), int((h - 1) // (grid_h - 1))

    # Average the green channels into one.
    av_ch_g = np.mean((channels[1:3]), axis=0)
    if do_alsc_colour:
        # Obtain grid_w x grid_h grid of intensities for each channel and subtract black level.
        g = get_grid(av_ch_g, dx, dy, grid_size) - img.blacklevel_16
        r = get_grid(channels[0], dx, dy, grid_size) - img.blacklevel_16
        b = get_grid(channels[3], dx, dy, grid_size) - img.blacklevel_16
        # Calculate ratios as 32-bit for medianBlur; then median blur to remove peaks.
        cr = np.reshape(g / r, (grid_h, grid_w)).astype('float32')
        cb = np.reshape(g / b, (grid_h, grid_w)).astype('float32')
        cg = np.reshape(1 / g, (grid_h, grid_w)).astype('float32')
        cr = cv2.medianBlur(cr, 3).astype('float64')
        cr = cr / np.min(cr)  # Gain tables easier to read if minimum is 1.0
        cb = cv2.medianBlur(cb, 3).astype('float64')
        cb = cb / np.min(cb)
        cg = cv2.medianBlur(cg, 3).astype('float64')
        cg = cg / np.min(cg)
        cg_clamp = [min(v, max_gain) for v in cg.flatten()]  # Never exceed max luminance gain

        return img.col, cr.flatten(), cb.flatten(), cg_clamp, (w, h, dx, dy)

    else:
        # Only perform calculations for luminance shading.
        g = get_grid(av_ch_g, dx, dy, grid_size) - img.blacklevel_16
        cg = np.reshape(1 / g, (grid_h, grid_w)).astype('float32')
        cg = cv2.medianBlur(cg, 3).astype('float64')
        cg = cg / np.min(cg)
        cg_clamp = [min(v, max_gain) for v in cg.flatten()]

        return img.col, None, None, cg_clamp, (w, h, dx, dy)


def get_grid(chan: np.ndarray, dx: int, dy: int, grid_size: tuple[int, int]) -> np.ndarray:
    """Compress channel down to a grid of the requested size."""
    grid_w, grid_h = grid_size
    h_total, w_total = chan.shape
    row_edges = np.arange(grid_h) * dy
    col_edges = np.arange(grid_w) * dx
    # reduceat sums contiguous blocks between edges; the last block extends to the array boundary.
    cell_sums = np.add.reduceat(np.add.reduceat(chan, row_edges, axis=0), col_edges, axis=1)
    row_sizes = np.diff(np.append(row_edges, h_total))
    col_sizes = np.diff(np.append(col_edges, w_total))
    cell_counts = row_sizes[:, np.newaxis] * col_sizes[np.newaxis, :]
    return (cell_sums / cell_counts).flatten()


def get_sigma(cam: Camera, cal_cr_list: list, cal_cb_list: list, grid_size: tuple[int, int]) -> tuple[float, float]:
    """Obtain sigmas for red and blue, a measure of the 'error' between adjacent colour temperatures."""
    cam.log += '\nCalculating sigmas'
    # With colour ALSC tables for two different CTs, sigma is from comparing adjacent calibrations.
    cts = [cal['ct'] for cal in cal_cr_list]
    sigma_rs = []
    sigma_bs = []
    # Calculate sigmas for each adjacent CT pair and return the worst one.
    for i in range(len(cts) - 1):
        sigma_rs.append(calc_sigma(cal_cr_list[i]['table'], cal_cr_list[i + 1]['table'], grid_size))
        sigma_bs.append(calc_sigma(cal_cb_list[i]['table'], cal_cb_list[i + 1]['table'], grid_size))
        cam.log += f'\nColour temperature interval {cts[i]} - {cts[i + 1]} K'
        cam.log += f'\nSigma red: {sigma_rs[-1]}'
        cam.log += f'\nSigma blue: {sigma_bs[-1]}'

    # Return maximum sigmas (not necessarily from the same CT interval).
    sigma_r = max(sigma_rs) if sigma_rs else 0.005
    sigma_b = max(sigma_bs) if sigma_bs else 0.005
    cam.log += f'\nMaximum sigmas: Red = {sigma_r} Blue = {sigma_b}'

    return sigma_r, sigma_b


def calc_sigma(g1: list, g2: list, grid_size: tuple[int, int]) -> float:
    """Calculate sigma from two adjacent gain tables."""
    grid_w, grid_h = grid_size
    g1 = np.reshape(g1, (grid_h, grid_w))
    g2 = np.reshape(g2, (grid_h, grid_w))
    # Ratio of gains between the two tables.
    gg = g1 / g2
    if np.mean(gg) < 1:
        gg = 1 / gg
    # For each internal patch, average difference between it and its 4 neighbours (border excluded).
    diffs = []
    for i in range(grid_h - 2):
        for j in range(grid_w - 2):
            # Indexing +1 because border patches are not counted.
            diff = np.abs(gg[i + 1][j + 1] - gg[i][j + 1])
            diff += np.abs(gg[i + 1][j + 1] - gg[i + 2][j + 1])
            diff += np.abs(gg[i + 1][j + 1] - gg[i + 1][j])
            diff += np.abs(gg[i + 1][j + 1] - gg[i + 1][j + 2])
            diffs.append(diff / 4)

    return np.round(np.mean(diffs), 5)
