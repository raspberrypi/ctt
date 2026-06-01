# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Macbeth chart visualisation - generates comparison images showing
# ideal vs original vs optimised colour correction results

import numpy as np
from PIL import Image


def visualise_macbeth_chart(
    macbeth_rgb: list,
    original_rgb: list,
    new_rgb: list,
    output_filename: str,
    *,
    matrix_selection: str | None = None,
) -> None:
    image = np.zeros((1050, 1550, 3), dtype=np.uint8)
    colorindex = -1
    for y in range(6):
        for x in range(4):
            colorindex += 1
            x1 = 50 + 250 * x
            y1 = 50 + 250 * y
            image[x1 : x1 + 100, y1 : y1 + 200] = macbeth_rgb[colorindex]
            x2 = 150 + 250 * x
            image[x2 : x2 + 100, y1 : y1 + 100] = original_rgb[colorindex]
            image[x2 : x2 + 100, y1 + 100 : y1 + 200] = new_rgb[colorindex]

    img = Image.fromarray(image, 'RGB')
    suffix = f'_{matrix_selection}' if matrix_selection else ''
    img.save(f'{output_filename}{suffix}_generated_macbeth_chart.png')
