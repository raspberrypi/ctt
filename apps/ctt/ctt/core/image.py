# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Image data class for CTT

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Image:
    channels: list = field(default_factory=list)
    patches: list | None = None
    saturated: bool = False
    w: int = 0
    h: int = 0
    pad: int = 0
    fmt: int = 0
    sigbits: int = 0
    pattern: int = 0
    order: tuple = (0, 1, 2, 3)
    exposure: int = 0
    againQ8: float = 0
    againQ8_norm: float = 0
    cam_name: str = ''
    blacklevel: int = 0
    blacklevel_16: int = 0
    col: int | None = None
    lux: int | None = None
    name: str = ''
    rgb: np.ndarray | None = None
    cen_coords: list | None = None
    macbeth_confidence: float | None = None
    ver: int = 0

    def get_patches(self, cen_coords: list, size: int = 16) -> int:
        cen_coords = list(np.array(cen_coords[0]).astype(np.int32))
        self.cen_coords = cen_coords
        half = size // 2
        all_patches = []
        for ch in self.channels:
            ch_h, ch_w = ch.shape
            ch_patches = []
            for cen in cen_coords:
                # Clamp center so the full patch fits within the channel.
                cy = np.clip(cen[1], half - 1, ch_h - half - 1)
                cx = np.clip(cen[0], half - 1, ch_w - half - 1)
                patch = ch[cy - half + 1 : cy + half + 1, cx - half + 1 : cx + half + 1].flatten()
                patch.sort()
                if patch[-5] == (2**self.sigbits - 1) * 2 ** (16 - self.sigbits):
                    self.saturated = True
                ch_patches.append(patch)
            all_patches.append(ch_patches)
        self.patches = all_patches
        return 1
