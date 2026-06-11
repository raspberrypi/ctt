# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Black level calibration
#
# Measures the sensor black level from dark frames (lens-cap/zero-light
# captures named dark_<idx>.dng). This is not dark-frame subtraction: only the
# per-channel mean pedestal is estimated. The measured value replaces the DNG
# metadata black level for every other algorithm in the run, and per-channel
# values are recorded in the metrics sidecar so the web UI can plot black
# level against total exposure (shutter x gain).

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .base import CalibrationAlgorithm

if TYPE_CHECKING:
    from ..core.camera import Camera
    from ..core.image import Image

logger = logging.getLogger(__name__)

# Diagnostic thresholds, in the 16-bit domain.
# A channel mean this far from the frame mean suggests a light leak (real
# black pedestals are near-identical across channels).
_CHANNEL_SPREAD_LIMIT = 0.005 * (2**16)
# A measurement this far from the DNG metadata value is worth flagging:
# either a light leak, or genuinely wrong metadata (the interesting case).
_METADATA_DELTA_LIMIT = 0.01 * (2**16)


def measure_dark_image(img: Image) -> dict:
    """Per-channel black level means of a loaded dark frame, 16-bit scaled.

    The spatial channels are reordered via img.order into (R, Gr, Gb, B) - the
    mapping established in image_loader.dng_load_image. Mono sensors (pattern
    128) report a single 'y' value instead of 'r'/'g'/'b'.
    """
    means = [float(np.mean(img.channels[i])) for i in img.order]
    stds = [float(np.std(img.channels[i])) for i in img.order]
    out = {
        'name': img.name,
        'black_level': float(np.mean(means)),
        'std': round(float(np.mean(stds)), 1),
        'exposure': img.exposure,
        'gain': round(img.againQ8_norm, 3),
        'total_exposure': round(img.exposure * img.againQ8_norm),
    }
    if img.pattern == 128:
        out['y'] = round(float(np.mean(means)), 1)
    else:
        r, gr, gb, b = means
        out.update({'r': round(r, 1), 'g': round((gr + gb) / 2, 1), 'b': round(b, 1)})
    return out


def measure_dark_dng(path: str) -> dict:
    """Measure a dark DNG straight from disk, skipping the demosaic.

    Shares the DNG parsing and the measurement with the calibration algorithm,
    so a quick measurement (e.g. the web UI seeding its black level field) is
    identical by construction to what a full run would compute.
    """
    from ..core.image_loader import dng_load_image

    img = dng_load_image(None, str(path), demosaic=False)
    return measure_dark_image(img)


class BlackLevelCalibration(CalibrationAlgorithm):
    json_key = 'rpi.black_level'

    def __init__(self, camera: Camera, platform: object, blacklevel_override: int = -1) -> None:
        super().__init__(camera, platform)
        self.blacklevel_override = blacklevel_override

    def run(self) -> dict | None:
        cam = self.camera

        cam.log_new_sec('BLACK LEVEL')
        if not cam.imgs_dark:
            cam.log += '\nNo dark frames found; black level taken from DNG metadata/config'
            return None

        frames = []
        for img in cam.imgs_dark:
            m = measure_dark_image(img)
            frames.append(m)
            cam.log += f'\nProcessing image: {img.name}'
            if 'y' in m:
                cam.log += f'\nChannel mean: y = {m["y"]}'
            else:
                cam.log += f'\nChannel means: r = {m["r"]} g = {m["g"]} b = {m["b"]}'
            cam.log += f'\nPixel std: {m["std"]}'
            cam.log += f'\nExposure: {m["exposure"]} us  Gain: {m["gain"]}'

            if 'r' in m:
                spread = max(abs(m[k] - m['black_level']) for k in ('r', 'g', 'b'))
                if spread > _CHANNEL_SPREAD_LIMIT:
                    warning = f'Dark frame channel means diverge by {spread:.0f} - possible light leak'
                    cam.log += f'\nWARNING: {warning}'
                    cam.add_warning('warn', warning, image=img.name)
            # Compare against the raw DNG metadata (img.blacklevel), not
            # blacklevel_16, which a config override may have replaced.
            metadata_16 = img.blacklevel << (16 - img.sigbits)
            if abs(m['black_level'] - metadata_16) > _METADATA_DELTA_LIMIT:
                warning = (
                    f'Measured black level {m["black_level"]:.0f} differs from DNG metadata '
                    f'{metadata_16} - light leak or wrong metadata'
                )
                cam.log += f'\nWARNING: {warning}'
                cam.add_warning('warn', warning, image=img.name)

        measured = round(float(np.mean([m['black_level'] for m in frames])))
        cam.log += f'\n\nMeasured black level: {measured} ({len(frames)} dark frame(s))'

        if self.blacklevel_override != -1:
            cam.log += f'\nConfig blacklevel override {self.blacklevel_override} takes precedence'
            cam.log += '\nMeasured value recorded for analysis only'
            source, black_level = 'override', self.blacklevel_override
        else:
            source, black_level = 'dark', measured
            # Propagate so every downstream algorithm uses the measurement.
            cam.blacklevel_16 = measured
            for other in cam.imgs + cam.imgs_alsc + cam.imgs_cac:
                other.blacklevel_16 = measured
            cam.log += '\nMeasured black level applied to all images'

        cam.metrics['black_level'] = {
            'black_level': black_level,
            'measured': measured,
            'source': source,
            'frames': frames,
        }
        info = f'\t\tBlack level: {measured} measured from {len(frames)} dark frame(s)'
        if self.blacklevel_override != -1:
            info += f' (config override {self.blacklevel_override} in use)'
        logger.info(info)

        if self.blacklevel_override != -1:
            return None
        return {'black_level': measured}
