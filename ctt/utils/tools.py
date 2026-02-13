# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Miscellaneous utility functions

import os

import cv2
import numpy as np


def nudge_for_json(arr: np.ndarray, decimals: int = 3) -> np.ndarray:
    """
    Nudge values to prevent JSON formatting corruption when values round to .x0 or .x9.

    Args:
        arr: Input array to nudge and round.
        decimals: Number of decimal places to round to (default: 3).

    Returns:
        Rounded array with nudged values.
    """
    factor = 10 ** (decimals - 1)
    epsilon = 10 ** (-decimals)
    arr = np.where((factor * arr) % 1 <= 0.05, arr + epsilon, arr)
    arr = np.where((factor * arr) % 1 >= 0.95, arr - epsilon, arr)
    return np.round(arr, decimals)


def get_alsc_colour_cals(cam_json: dict) -> dict[int, list] | None:
    """
    Extract ALSC colour calibration tables from cam.json as {ct: [cr_table, cb_table]}.

    Tables are normalised so minimum value is 1.0.

    Args:
        cam_json: Camera JSON dictionary containing 'rpi.alsc' section.

    Returns:
        Dictionary mapping colour temperature to [cr_table, cb_table] lists,
        or None if ALSC calibrations not found.
    """
    try:
        cal_cr_list = cam_json['rpi.alsc']['calibrations_Cr']
        cal_cb_list = cam_json['rpi.alsc']['calibrations_Cb']
    except KeyError:
        return None

    colour_cals = {}
    for cr, cb in zip(cal_cr_list, cal_cb_list, strict=True):
        cr_tab = cr['table']
        cb_tab = cb['table']
        cr_tab = cr_tab / np.min(cr_tab)  # Normalise tables so min value is 1
        cb_tab = cb_tab / np.min(cb_tab)
        colour_cals[cr['ct']] = [cr_tab, cb_tab]
    return colour_cals


def correlate(im1: np.ndarray, im2: np.ndarray) -> float:
    f1 = im1.flatten()
    f2 = im2.flatten()
    cor = np.corrcoef(f1, f2)
    return cor[0][1]


def get_photos(directory: str = 'photos') -> list[str]:
    filename_list = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path) and filename.lower().endswith('.dng'):
            filename_list.append(filename)
    return filename_list


def reshape(img: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    factor = width / img.shape[0]
    return cv2.resize(img, None, fx=factor, fy=factor), factor
