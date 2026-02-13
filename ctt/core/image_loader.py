# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# DNG image loading

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyexiv2 as pyexif
import rawpy as raw

from ..detection.macbeth import find_macbeth
from .image import Image

if TYPE_CHECKING:
    from .camera import Camera

logger = logging.getLogger(__name__)


def dng_load_image(cam: Camera, im_str: str) -> Image:
    try:
        img = Image()

        metadata = pyexif.ImageMetadata(im_str)
        metadata.read()

        img.ver = 100
        try:
            img.w = metadata['Exif.SubImage1.ImageWidth'].value
            subimage = 'SubImage1'
            photo = 'Photo'
        except KeyError:
            img.w = metadata['Exif.Image.ImageWidth'].value
            subimage = 'Image'
            photo = 'Image'
        img.pad = 0
        img.h = metadata[f'Exif.{subimage}.ImageLength'].value
        try:
            white = metadata[f'Exif.{subimage}.WhiteLevel'].value
        except (KeyError, AttributeError):
            white = (1 << 16) - 1  # mono DNGs may omit WhiteLevel
        img.sigbits = int(white).bit_length()
        img.fmt = (img.sigbits - 4) // 2
        img.exposure = int(metadata[f'Exif.{photo}.ExposureTime'].value * 1000000)
        img.againQ8 = metadata[f'Exif.{photo}.ISOSpeedRatings'].value * 256 / 100
        img.againQ8_norm = img.againQ8 / 256
        img.cam_name = metadata['Exif.Image.Model'].value
        blacks = metadata[f'Exif.{subimage}.BlackLevel'].value
        try:
            img.blacklevel = int(blacks[0])
        except TypeError:
            img.blacklevel = int(blacks)
        img.blacklevel_16 = img.blacklevel << (16 - img.sigbits)
        bayer_case = {
            '0 1 1 2': (0, (0, 1, 2, 3)),
            '1 2 0 1': (1, (2, 0, 3, 1)),
            '2 1 1 0': (2, (3, 2, 1, 0)),
            '1 0 2 1': (3, (1, 0, 3, 2)),
        }
        try:
            cfa_pattern = metadata[f'Exif.{subimage}.CFAPattern'].value
            img.pattern = bayer_case[cfa_pattern][0]
            img.order = bayer_case[cfa_pattern][1]
        except (KeyError, AttributeError):
            img.pattern = 128  # mono DNGs have no CFA
            img.order = (0, 1, 2, 3)

        raw_im = raw.imread(im_str)
        raw_data = raw_im.raw_image
        shift = 16 - img.sigbits
        c0 = np.left_shift(raw_data[0::2, 0::2].astype(np.int64), shift)
        c1 = np.left_shift(raw_data[0::2, 1::2].astype(np.int64), shift)
        c2 = np.left_shift(raw_data[1::2, 0::2].astype(np.int64), shift)
        c3 = np.left_shift(raw_data[1::2, 1::2].astype(np.int64), shift)
        img.channels = [c0, c1, c2, c3]
        img.rgb = raw_im.postprocess()

    except Exception:
        logger.error(f'\nERROR: failed to load DNG file {im_str}')
        logger.error('Either file does not exist or is incompatible')
        cam.log += '\nERROR: DNG file does not exist or is incompatible'
        raise

    img.name = Path(im_str).name
    return img


def load_image(cam: Camera, im_str: str, mac_config: tuple | None = None, mac: bool = True) -> Image | None:
    if '.dng' in im_str:
        img = dng_load_image(cam, im_str)

        if mac:
            av_chan = np.mean(np.array(img.channels), axis=0) / (2**16)
            av_val = np.mean(av_chan)
            if av_val < img.blacklevel_16 / (2**16) + 1 / 64:
                result = None
                logger.error('\nError: Image too dark!')
                cam.log += '\nWARNING: Image too dark!'
            else:
                result = find_macbeth(cam, av_chan, mac_config)

            if result is None or result[0] is None:
                return None
            coords_fit, confidence = result
            mac_cen_coords = coords_fit[1]

            img.get_patches(mac_cen_coords)
            if img.saturated:
                logger.error('\nERROR: Macbeth patches have saturated')
                cam.log += '\nWARNING: Macbeth patches have saturated!'
                return None

            if confidence is not None:
                img.macbeth_confidence = confidence

        return img

    else:
        return None
