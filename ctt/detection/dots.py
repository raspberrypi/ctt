# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# CAC dot detection

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def find_dots_locations(
    rgb_image: np.ndarray,
    color_threshold: int = 100,
    dots_edge_avoid: int = 75,
    image_edge_avoid: int = 10,
    search_path_length: int = 500,
    grid_scan_step_size: int = 10,
    logfile: object = None,
) -> tuple[list, list]:
    pixels = Image.fromarray(rgb_image)
    pixels = pixels.convert('L')
    enhancer = ImageEnhance.Contrast(pixels)
    im_output = enhancer.enhance(1.4)
    im_output = im_output.filter(ImageFilter.GaussianBlur(radius=2))
    bw_image = np.array(im_output)

    random.seed(42)
    location = [0, 0]
    dots = []
    dots_location = []
    for x in range(dots_edge_avoid, len(bw_image) - dots_edge_avoid, grid_scan_step_size):
        for y in range(dots_edge_avoid, len(bw_image[0]) - dots_edge_avoid, grid_scan_step_size):
            location = [x, y]
            scrap_dot = False
            if (bw_image[location[0], location[1]] < color_threshold) and not (scrap_dot):
                heading = 'south'
                coords = []
                for _i in range(search_path_length):
                    if (
                        (image_edge_avoid < location[0] < len(bw_image) - image_edge_avoid)
                        and (image_edge_avoid < location[1] < len(bw_image[0]) - image_edge_avoid)
                    ) and not (scrap_dot):
                        if heading == 'south':
                            if bw_image[location[0] + 1, location[1]] < color_threshold:
                                location[0] = location[0] + 1
                                location[1] = location[1] + 1
                                heading = 'south'
                            else:
                                dir = random.randint(1, 2)
                                if dir == 1:
                                    heading = 'west'
                                if dir == 2:
                                    heading = 'east'

                        if heading == 'east':
                            if bw_image[location[0], location[1] + 1] < color_threshold:
                                location[1] = location[1] + 1
                                heading = 'east'
                            else:
                                dir = random.randint(1, 2)
                                if dir == 1:
                                    heading = 'north'
                                if dir == 2:
                                    heading = 'south'

                        if heading == 'west':
                            if bw_image[location[0], location[1] - 1] < color_threshold:
                                location[1] = location[1] - 1
                                heading = 'west'
                            else:
                                dir = random.randint(1, 2)
                                if dir == 1:
                                    heading = 'north'
                                if dir == 2:
                                    heading = 'south'

                        if heading == 'north':
                            if bw_image[location[0] - 1, location[1]] < color_threshold:
                                location[0] = location[0] - 1
                                heading = 'north'
                            else:
                                dir = random.randint(1, 2)
                                if dir == 1:
                                    heading = 'west'
                                if dir == 2:
                                    heading = 'east'
                        coords.append([location[0], location[1]])
                    else:
                        scrap_dot = True
                if not scrap_dot:
                    x_coords = np.array(coords)[:, 0]
                    y_coords = np.array(coords)[:, 1]
                    hsquaresize = max(list(x_coords)) - min(list(x_coords))
                    vsquaresize = max(list(y_coords)) - min(list(y_coords))
                    extra_space_factor = 0.45
                    top_left_x = min(list(x_coords)) - int(hsquaresize * extra_space_factor)
                    btm_right_x = max(list(x_coords)) + int(hsquaresize * extra_space_factor)
                    top_left_y = min(list(y_coords)) - int(vsquaresize * extra_space_factor)
                    btm_right_y = max(list(y_coords)) + int(vsquaresize * extra_space_factor)
                    bw_image[top_left_x:btm_right_x, top_left_y:btm_right_y] = 255
                    dots.append(rgb_image[top_left_x:btm_right_x, top_left_y:btm_right_y])
                    dots_location.append([top_left_x, top_left_y])
    return dots, dots_location
