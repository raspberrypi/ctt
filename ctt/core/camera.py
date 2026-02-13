# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Camera data holder

import logging
import re
import sys
import time
from pathlib import Path

from ..output.json_formatter import pretty_print
from ..utils.tools import get_photos
from .image_loader import load_image

_RED = '\033[31m' if sys.stdout.isatty() else ''
_RESET = '\033[0m' if sys.stdout.isatty() else ''

logger = logging.getLogger(__name__)


class _LogBuffer:
    """Accumulate log text via += without O(n**2) string copying."""

    __slots__ = ('_parts',)

    def __init__(self, initial: str = '') -> None:
        self._parts: list[str] = [initial] if initial else []

    def __iadd__(self, other: str) -> '_LogBuffer':
        self._parts.append(other)
        return self

    def __str__(self) -> str:
        return ''.join(self._parts)


def get_col_lux(string: str) -> tuple[int | None, int | None]:
    col = re.search(r'([0-9]+)[kK](\.(jpg|jpeg|dng)|_.*\.(jpg|jpeg|dng))$', string)
    lux = re.search(r'([0-9]+)[lL](\.(jpg|jpeg|dng)|_.*\.(jpg|jpeg|dng))$', string)
    try:
        col = col.group(1)
    except AttributeError:
        return None, None
    try:
        lux = lux.group(1)
    except AttributeError:
        return col, None
    return int(col), int(lux)


class Camera:
    def __init__(self, jfile: str, json: dict) -> None:
        self.imgs = []
        self.imgs_alsc = []
        self.imgs_cac = []
        self.log = _LogBuffer(f'Log created : {time.asctime(time.localtime(time.time()))}')
        self.log_separator = f'\n{"-" * 70}\n'
        self.jf = jfile
        self.json = json
        self.output_dir = Path.cwd()
        self.disable = []
        self.plot = []
        self.mono = False
        self.blacklevel_16 = 0

    def write_json(self, version: float = 2.0, target: str = 'bcm2835', grid_size: tuple[int, int] = (16, 12)) -> None:
        out_json = {
            'version': version,
            'target': target if target != 'vc4' else 'bcm2835',
            'algorithms': [{name: data} for name, data in self.json.items()],
        }

        with open(self.jf, 'w') as f:
            f.write(pretty_print(out_json, custom_elems={'table': grid_size[0], 'luminance_lut': grid_size[0]}))

    def log_new_sec(self, section: str, cal: bool = True) -> None:
        self.log += f'\n{self.log_separator}'
        self.log += section
        if cal:
            self.log += ' Calibration'
        self.log += f'{self.log_separator}'

    def log_user_input(self, json_output: str, directory: str, config: object, log_output: str | None) -> None:
        self.log_new_sec('User Arguments', cal=False)
        self.log += f'\nJson file output: {json_output}'
        self.log += f'\nCalibration images directory: {directory}'
        if config is None:
            self.log += '\nNo configuration file input... using default options'
        elif config is False:
            self.log += '\nWARNING: Invalid configuration file path...'
            self.log += ' using default options'
        elif config is True:
            self.log += '\nWARNING: Invalid syntax in configuration file...'
            self.log += ' using default options'
        else:
            self.log += f'\nConfiguration file: {config}'
        if log_output is None:
            self.log += '\nNo log file path input... using default: ctt_log.txt'
        else:
            self.log += f'\nLog file output: {log_output}'

    def write_log(self, filename: str | None) -> None:
        if filename is None:
            filename = 'ctt_log.txt'
        self.log += f'\n{self.log_separator}'
        with open(filename, 'w') as logfile:
            logfile.write(str(self.log))

    def add_imgs(self, directory: str, mac_config: tuple, blacklevel: int = -1) -> None:
        self.log_new_sec('Image Loading', cal=False)
        logger.info(f'\nLoading images from {directory}')
        self.log += f'\nDirectory: {directory}'
        filename_list = get_photos(directory)
        logger.info(f'Files found: {len(filename_list)}')
        self.log += f'\nFiles found: {len(filename_list)}'
        logger.info('Loading')
        filename_list.sort()
        for filename in filename_list:
            address = directory + filename
            self.log += f'\n\nImage: {filename}'
            col, lux = get_col_lux(filename)
            if 'alsc' in filename:
                img = load_image(self, address, mac=False)
                self.log += '\nIdentified as an ALSC image'
                if img is None:
                    logger.info(f'\nDISCARDED: {filename}')
                    self.log += '\nImage discarded!'
                    continue
                if col is not None:
                    img.col = col
                    img.name = filename
                    self.log += f'\nColour temperature: {col} K'
                    self.imgs_alsc.append(img)
                    if blacklevel != -1:
                        img.blacklevel_16 = blacklevel
                    print(f'\t{filename}', flush=True)
                    continue
                else:
                    logger.error('Error! No colour temperature found!')
                    self.log += '\nWARNING: Error reading colour temperature'
                    self.log += '\nImage discarded!'
                    logger.info(f'DISCARDED: {filename}')
            elif 'cac' in filename:
                img = load_image(self, address, mac=False)
                self.log += '\nIdentified as an CAC image'
                img.name = filename
                self.log += f'\nColour temperature: {col} K'
                self.imgs_cac.append(img)
                if blacklevel != -1:
                    img.blacklevel_16 = blacklevel
                print(f'\t{filename}', flush=True)
                continue
            else:
                self.log += '\nIdentified as macbeth chart image'
                if col is None or lux is None:
                    print(f'\t{filename}', flush=True)
                    print(
                        f'\t\t{_RED}✗ Colour temp/lux not found in filename (expected e.g. 5000k_800l.dng){_RESET}',
                        flush=True,
                    )
                    self.log += '\nWARNING: Error reading colour temp/lux from filename'
                    self.log += '\nImage discarded!'
                    continue
                img = load_image(self, address, mac_config)
                if img is None:
                    print(f'\t{filename}', flush=True)
                    print(f'\t\t{_RED}✗ Macbeth chart not found in image{_RESET}', flush=True)
                    self.log += '\nImage discarded!'
                    continue
                img.col, img.lux = col, lux
                img.name = filename
                self.log += f'\nColour temperature: {col} K'
                self.log += f'\nLux value: {lux} lx'
                if blacklevel != -1:
                    img.blacklevel_16 = blacklevel
                print(f'\t{filename}', flush=True)
                if getattr(img, 'macbeth_confidence', None) is not None:
                    print(f'\t\t✓ Macbeth found (confidence {img.macbeth_confidence:.3f})', flush=True)
                self.imgs.append(img)

    def check_imgs(self, macbeth: bool = True) -> bool:
        self.log += '\n\nImages found:'
        self.log += f'\nMacbeth : {len(self.imgs)}'
        self.log += f'\nALSC : {len(self.imgs_alsc)} '
        self.log += f'\nCAC: {len(self.imgs_cac)} '
        self.log += '\n\nCamera metadata'
        if len(self.imgs) == 0 and macbeth:
            logger.error('\nERROR: No usable macbeth chart images found')
            self.log += '\nERROR: No usable macbeth chart images found'
            return False
        elif len(self.imgs) == 0 and len(self.imgs_alsc) == 0 and len(self.imgs_cac) == 0:
            logger.error('\nERROR: No usable images found')
            self.log += '\nERROR: No usable images found'
            return False
        all_imgs = self.imgs + self.imgs_alsc + self.imgs_cac
        cam_names = list({img.cam_name for img in all_imgs})
        patterns = list({img.pattern for img in all_imgs})
        sigbitss = list({img.sigbits for img in all_imgs})
        blacklevels = list({img.blacklevel_16 for img in all_imgs})
        sizes = list({(img.w, img.h) for img in all_imgs})

        self.mono = patterns[0] == 128
        self.blacklevel_16 = blacklevels[0]
        self.log += f'\nName: {cam_names[0]}'
        self.log += f'\nBayer pattern case: {patterns[0]}'
        if self.mono:
            self.log += '\nGreyscale camera identified'
        self.log += f'\nSignificant bits: {sigbitss[0]}'
        self.log += f'\nBlacklevel: {blacklevels[0]}'
        self.log += f'\nImage size: w = {sizes[0][0]} h = {sizes[0][1]}'
        return True

    def json_remove(self, disable: list) -> None:
        self.log_new_sec('Disabling Options', cal=False)
        if len(self.disable) == 0:
            self.log += '\nNothing disabled!'
            return
        for key in disable:
            try:
                del self.json[key]
                self.log += f'\nDisabled: {key}'
            except KeyError:
                self.log += f'\nERROR: {key} not found!'
