# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Miscellaneous utility functions

import os

import cv2
import numpy as np

from .errors import ArgError


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


def read_manifest(manifest_path: str, input_dir: str) -> list[str]:
    """Parse a run manifest: one .dng filename per line, relative to input_dir.

    Blank lines and lines starting with '#' are ignored; duplicates are removed
    preserving order. Raises ArgError if the manifest cannot be read, an entry
    contains a path or is not a .dng, a listed file is missing from input_dir,
    or no images remain.
    """
    try:
        with open(manifest_path) as f:
            lines = f.readlines()
    except OSError as err:
        raise ArgError(f'\n\nError: Could not read manifest file: {err}') from err

    images = []
    for line in lines:
        entry = line.strip()
        if not entry or entry.startswith('#'):
            continue
        if '/' in entry or '\\' in entry:
            raise ArgError(f'\n\nError: Manifest entries must be bare filenames, got: {entry!r}')
        if not entry.lower().endswith('.dng'):
            raise ArgError(f'\n\nError: Manifest entries must be .dng files, got: {entry!r}')
        if entry not in images:
            images.append(entry)

    if not images:
        raise ArgError(f'\n\nError: Manifest {manifest_path} lists no images')
    missing = [f for f in images if not os.path.isfile(os.path.join(input_dir, f))]
    if missing:
        raise ArgError(f'\n\nError: Manifest images not found in {input_dir}: {", ".join(missing)}')
    return images


def reshape(img: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    factor = width / img.shape[0]
    return cv2.resize(img, None, fx=factor, fy=factor), factor
