# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# DNG image loading

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import exifread
import numpy as np
import rawpy as raw

from ..detection.macbeth import find_macbeth
from .image import Image

if TYPE_CHECKING:
    from .camera import Camera

logger = logging.getLogger(__name__)


def dng_load_image(cam: Camera, im_str: str) -> Image:
    try:
        img = Image()

        with open(im_str, 'rb') as f:
            tags = exifread.process_file(f, details=True)

        img.ver = 100

        def _tag(name):
            """Look up a tag across the DNG's IFDs, tolerating layout differences.

            Raspberry Pi DNGs vary by writer: rpicam/libcamera put geometry in
            'EXIF SubIFD0' and ExposureTime/ISO in the 'EXIF' IFD, whereas PiDNG
            (Picamera2) puts them in the main 'Image' IFD. Check all three.
            """
            for prefix in ('EXIF SubIFD0', 'Image', 'EXIF'):
                key = f'{prefix} {name}'
                if key in tags:
                    return tags[key]
            raise KeyError(name)

        img.w = _tag('ImageWidth').values[0]
        img.pad = 0
        img.h = _tag('ImageLength').values[0]
        try:
            white = tags['EXIF SubIFD0 Tag 0xC61D'].values[0]
        except KeyError:
            white = (1 << 16) - 1  # mono DNGs may omit WhiteLevel
        img.sigbits = int(white).bit_length()
        img.fmt = (img.sigbits - 4) // 2
        exp = _tag('ExposureTime').values[0]
        img.exposure = int(exp.num / exp.den * 1_000_000)
        img.againQ8 = _tag('ISOSpeedRatings').values[0] * 256 / 100
        img.againQ8_norm = img.againQ8 / 256
        img.cam_name = str(tags['Image Model']).strip()
        blacks = _tag('BlackLevel').values
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
            cfa_values = _tag('CFAPattern').values
            cfa_pattern = ' '.join(str(x) for x in cfa_values)
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


def apply_gamma(av_chan: np.ndarray, cam: Camera) -> np.ndarray:
    """Tone-map a linear 0-1 image with the tuning template's gamma curve.

    The template's rpi.contrast gamma_curve is a piecewise-linear LUT of flat
    [x0, y0, x1, y1, ...] pairs, input and output both 0-65535. Mapping the linear
    Bayer mean through it reproduces the preview tone curve, which is what makes the
    Macbeth chart reliably detectable. Falls back to a plain 2.2 gamma if the template
    has no contrast curve (e.g. it was disabled).
    """
    curve = cam.json.get('rpi.contrast', {}).get('gamma_curve') if isinstance(cam.json, dict) else None
    if not curve:
        return np.power(np.clip(av_chan, 0, 1), 1 / 2.2)
    pts = np.asarray(curve, dtype=np.float64).reshape(-1, 2)
    xs, ys = pts[:, 0], pts[:, 1]
    return np.interp(np.clip(av_chan, 0, 1) * 65535.0, xs, ys) / 65535.0


def _chart_size_warning(corners: np.ndarray, image_width: float) -> str | None:
    """Advise when the detected chart is a poor size in the frame.

    Too small wastes SNR (tiny patch windows); near-full-frame puts the
    outer patches into the lens-shading corners.
    """
    xs = np.asarray(corners, dtype=np.float64)[:, 0]
    fraction = (xs.max() - xs.min()) / image_width
    if fraction < 0.25:
        return f'Macbeth chart small in frame ({fraction:.0%} of width) — move closer for better patch statistics'
    if fraction > 0.9:
        return (
            f'Macbeth chart fills the frame ({fraction:.0%} of width) — outer patches sit in the lens-shading corners'
        )
    return None


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
                cam.add_warning('warn', 'Image too dark', image=Path(im_str).name)
            else:
                # Locate the chart on a tone-mapped copy. The raw Bayer mean is linear,
                # so patch-to-patch contrast in the shadows/mid-tones is compressed and
                # Canny edge detection misses the patch borders -- worst under high-CT
                # light, where the red channel is weak. Applying the tuning template's
                # gamma curve expands that low/mid contrast (matching the tone-mapped
                # preview in which the chart is always found) and makes detection robust.
                # Only the location search sees this; patch values are still read from the
                # untouched linear channels in get_patches, so calibration is unchanged.
                det_chan = apply_gamma(av_chan, cam)
                result = find_macbeth(cam, det_chan, mac_config)

            if result is None or result[0] is None:
                return None
            coords_fit, confidence = result
            mac_cen_coords = coords_fit[1]

            warning = _chart_size_warning(np.asarray(coords_fit[0][0]), av_chan.shape[1])
            if warning:
                cam.log += '\n' + warning
                cam.add_warning('warn', warning, image=Path(im_str).name)

            img.get_patches(mac_cen_coords)
            cam.log += f'\nPatch sampling window: {img.patch_size} px'
            if img.saturated:
                logger.error('\nERROR: Macbeth patches have saturated')
                cam.log += '\nWARNING: Macbeth patches have saturated!'
                cam.add_warning('warn', 'Macbeth patches saturated', image=Path(im_str).name)
                return None

            if confidence is not None:
                img.macbeth_confidence = confidence

        return img

    else:
        return None
