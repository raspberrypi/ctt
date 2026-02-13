# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# CAC (Chromatic Aberration Correction) calibration

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..detection.dots import find_dots_locations
from .base import CalibrationAlgorithm

if TYPE_CHECKING:
    from ..core.camera import Camera

logger = logging.getLogger(__name__)


class CacCalibration(CalibrationAlgorithm):
    json_key = 'rpi.cac'

    def __init__(self, camera: Camera, platform: object, do_alsc_colour: bool = True) -> None:
        super().__init__(camera, platform)
        self.do_alsc_colour = do_alsc_colour

    def run(self) -> dict | None:
        cam = self.camera

        cam.log_new_sec('CAC')

        if len(cam.imgs_cac) == 0:
            logger.error('\nError:\nNo cac calibration images found')
            cam.log += '\nERROR: No CAC calibration images found!'
            cam.log += '\nCAC calibration aborted!'
            return None

        if cam.mono:
            logger.error("\nERROR: Can't do CAC on greyscale image!")
            cam.log += '\nERROR: Cannot perform CAC calibration '
            cam.log += 'on greyscale image!\nCAC aborted!'
            del cam.json['rpi.cac']
            return None

        if self.do_alsc_colour:
            try:
                cacs = cac(cam)
            except ArithmeticError:
                logger.error('ERROR: Matrix is singular!\nTake new pictures and try again...')
                cam.log += '\nERROR: Singular matrix encountered during fit!'
                cam.log += '\nCAC aborted!'
                return None
        else:
            cam.log += '\nWARNING: No ALSC tables found.\nCAC calibration '
            cam.log += 'performed without ALSC correction...'
            return None

        if cacs:
            cam.log += '\nCAC calibration written to json file'
            return {'cac': cacs}
        else:
            cam.log += '\nCAC calibration failed'
            return None


def shifts_to_yaml(
    red_shift: list, blue_shift: list, image_dimensions: list, output_grid_size: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    red_shifts = np.array(red_shift)
    blue_shifts = np.array(blue_shift)
    xrgrid = np.zeros((output_grid_size - 1, output_grid_size - 1))
    xbgrid = np.zeros((output_grid_size - 1, output_grid_size - 1))
    yrgrid = np.zeros((output_grid_size - 1, output_grid_size - 1))
    ybgrid = np.zeros((output_grid_size - 1, output_grid_size - 1))

    xrsgrid = []
    xbsgrid = []
    yrsgrid = []
    ybsgrid = []

    for x in range(output_grid_size - 1):
        xrsgrid.append([])
        yrsgrid.append([])
        xbsgrid.append([])
        ybsgrid.append([])
        for _y in range(output_grid_size - 1):
            xrsgrid[x].append([])
            yrsgrid[x].append([])
            xbsgrid[x].append([])
            ybsgrid[x].append([])

    image_size = (image_dimensions[0], image_dimensions[1])
    gridxsize = image_size[0] / (output_grid_size - 1)
    gridysize = image_size[1] / (output_grid_size - 1)

    for red_shift in red_shifts:
        xgridloc = int(red_shift[0] / gridxsize)
        ygridloc = int(red_shift[1] / gridysize)
        xrsgrid[xgridloc][ygridloc].append(red_shift[2])
        yrsgrid[xgridloc][ygridloc].append(red_shift[3])

    for blue_shift in blue_shifts:
        xgridloc = int(blue_shift[0] / gridxsize)
        ygridloc = int(blue_shift[1] / gridysize)
        xbsgrid[xgridloc][ygridloc].append(blue_shift[2])
        ybsgrid[xgridloc][ygridloc].append(blue_shift[3])

    grid_incomplete = False
    for x in range(output_grid_size - 1):
        for y in range(output_grid_size - 1):
            if xrsgrid[x][y]:
                xrgrid[x, y] = np.mean(xrsgrid[x][y])
            else:
                grid_incomplete = True
            if yrsgrid[x][y]:
                yrgrid[x, y] = np.mean(yrsgrid[x][y])
            else:
                grid_incomplete = True
            if xbsgrid[x][y]:
                xbgrid[x, y] = np.mean(xbsgrid[x][y])
            else:
                grid_incomplete = True
            if ybsgrid[x][y]:
                ybgrid[x, y] = np.mean(ybsgrid[x][y])
            else:
                grid_incomplete = True

    if grid_incomplete:
        raise RuntimeError(
            '\nERROR: CAC measurements do not span the image!'
            '\nConsider using improved CAC images, or remove them entirely.\n'
        )

    input_grids = np.array([xrgrid, yrgrid, xbgrid, ybgrid])
    output_grids = np.zeros((4, output_grid_size, output_grid_size))

    output_grids[:, 1:-1, 1:-1] = (
        input_grids[:, 1:, :-1] + input_grids[:, 1:, 1:] + input_grids[:, :-1, 1:] + input_grids[:, :-1, :-1]
    ) / 4

    output_grids[:, 1:-1, 0] = (
        (input_grids[:, :-1, 0] + input_grids[:, 1:, 0]) / 2 - output_grids[:, 1:-1, 1]
    ) * 2 + output_grids[:, 1:-1, 1]
    output_grids[:, 1:-1, -1] = (
        (input_grids[:, :-1, 7] + input_grids[:, 1:, 7]) / 2 - output_grids[:, 1:-1, -2]
    ) * 2 + output_grids[:, 1:-1, -2]
    output_grids[:, 0, 1:-1] = (
        (input_grids[:, 0, :-1] + input_grids[:, 0, 1:]) / 2 - output_grids[:, 1, 1:-1]
    ) * 2 + output_grids[:, 1, 1:-1]
    output_grids[:, -1, 1:-1] = (
        (input_grids[:, 7, :-1] + input_grids[:, 7, 1:]) / 2 - output_grids[:, -2, 1:-1]
    ) * 2 + output_grids[:, -2, 1:-1]

    output_grids[:, 0, 0] = (
        (output_grids[:, 0, 1] - output_grids[:, 1, 1])
        + (output_grids[:, 1, 0] - output_grids[:, 1, 1])
        + output_grids[:, 1, 1]
    )
    output_grids[:, 0, -1] = (
        (output_grids[:, 0, -2] - output_grids[:, 1, -2])
        + (output_grids[:, 1, -1] - output_grids[:, 1, -2])
        + output_grids[:, 1, -2]
    )
    output_grids[:, -1, 0] = (
        (output_grids[:, -1, 1] - output_grids[:, -2, 1])
        + (output_grids[:, -2, 0] - output_grids[:, -2, 1])
        + output_grids[:, -2, 1]
    )
    output_grids[:, -1, -1] = (
        (output_grids[:, -2, -1] - output_grids[:, -2, -2])
        + (output_grids[:, -1, -2] - output_grids[:, -2, -2])
        + output_grids[:, -2, -2]
    )

    output_grid_yr, output_grid_xr, output_grid_yb, output_grid_xb = output_grids * -1
    return output_grid_xr, output_grid_yr, output_grid_xb, output_grid_yb


def analyse_dot(dot: np.ndarray, dot_location: list | None = None) -> list:
    if dot_location is None:
        dot_location = [0, 0]
    red_channel = np.array(dot)[:, :, 0]
    y_num_pixels = len(red_channel[0])
    x_num_pixels = len(red_channel)
    yred_weight = np.sum(np.dot(red_channel, np.arange(y_num_pixels)))
    xred_weight = np.sum(np.dot(np.arange(x_num_pixels), red_channel))
    red_sum = np.sum(red_channel)

    green_channel = np.array(dot)[:, :, 1]
    ygreen_weight = np.sum(np.dot(green_channel, np.arange(y_num_pixels)))
    xgreen_weight = np.sum(np.dot(np.arange(x_num_pixels), green_channel))
    green_sum = np.sum(green_channel)

    blue_channel = np.array(dot)[:, :, 2]
    yblue_weight = np.sum(np.dot(blue_channel, np.arange(y_num_pixels)))
    xblue_weight = np.sum(np.dot(np.arange(x_num_pixels), blue_channel))
    blue_sum = np.sum(blue_channel)

    return [
        [
            int(dot_location[0]) + int(len(dot) / 2),
            int(dot_location[1]) + int(len(dot[0]) / 2),
            xred_weight / red_sum - xgreen_weight / green_sum,
            yred_weight / red_sum - ygreen_weight / green_sum,
        ],
        [
            dot_location[0] + int(len(dot) / 2),
            dot_location[1] + int(len(dot[0]) / 2),
            xblue_weight / blue_sum - xgreen_weight / green_sum,
            yblue_weight / blue_sum - ygreen_weight / green_sum,
        ],
    ]


def cac(cam: Camera) -> dict:
    filelist = cam.imgs_cac

    cam.log += f'\nCAC analysing files: {filelist}'
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    red_shift = []
    blue_shift = []
    for file in filelist:
        cam.log += '\nCAC processing file'
        logger.info('\n Processing file')
        rgb = file.rgb
        image_size = [file.h, file.w]
        rgb_image = rgb

        logger.info('Finding dots')
        cam.log += '\nFinding dots'
        dots, dots_locations = find_dots_locations(rgb_image)

        cam.log += f'\nDots found: {len(dots)}'
        logger.info(f'Dots found: {len(dots)}')

        for dot, dot_location in zip(dots, dots_locations, strict=True):
            if len(dot) > 0 and (dot_location[0] > 0) and (dot_location[1] > 0):
                ret = analyse_dot(dot, dot_location)
                red_shift.append(ret[0])
                blue_shift.append(ret[1])

    logger.info('\nCreating output grid')
    cam.log += '\nCreating output grid'
    try:
        rx, ry, bx, by = shifts_to_yaml(red_shift, blue_shift, image_size)
    except RuntimeError as e:
        logger.error(str(e))
        cam.log += '\nCAC correction failed! CAC will not be enabled.'
        return {}

    logger.info('CAC correction complete!')
    cam.log += '\nCAC correction complete!'

    return {
        'strength': 1.0,
        'lut_rx': list(rx.round(2).reshape(81)),
        'lut_ry': list(ry.round(2).reshape(81)),
        'lut_bx': list(bx.round(2).reshape(81)),
        'lut_by': list(by.round(2).reshape(81)),
    }
