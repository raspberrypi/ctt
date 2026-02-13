# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# RANSAC geometry helpers for Macbeth chart locator

import numpy as np

scale = 2


def get_square_verts(c_err: float = 0.05, scale: int = scale) -> tuple[np.ndarray, np.ndarray]:
    b_bord_x, b_bord_y = scale * 8.5, scale * 13
    s_bord = 6 * scale
    side = 41 * scale
    x_max = side * 6 + 5 * s_bord + 2 * b_bord_x
    y_max = side * 4 + 3 * s_bord + 2 * b_bord_y
    c1 = (0, 0)
    c2 = (0, y_max)
    c3 = (x_max, y_max)
    c4 = (x_max, 0)
    mac_norm = np.array((c1, c2, c3, c4), np.float32)
    mac_norm = np.array([mac_norm])

    square_verts = []
    square_0 = np.array(((0, 0), (0, side), (side, side), (side, 0)), np.float32)
    offset_0 = np.array((b_bord_x, b_bord_y), np.float32)
    c_off = side * c_err
    offset_cont = np.array(((c_off, c_off), (c_off, -c_off), (-c_off, -c_off), (-c_off, c_off)), np.float32)
    square_0 += offset_0
    square_0 += offset_cont
    for i in range(6):
        shift_i = np.array(((i * side, 0), (i * side, 0), (i * side, 0), (i * side, 0)), np.float32)
        shift_bord = np.array(((i * s_bord, 0), (i * s_bord, 0), (i * s_bord, 0), (i * s_bord, 0)), np.float32)
        square_i = square_0 + shift_i + shift_bord
        for j in range(4):
            shift_j = np.array(((0, j * side), (0, j * side), (0, j * side), (0, j * side)), np.float32)
            shift_bord = np.array(((0, j * s_bord), (0, j * s_bord), (0, j * s_bord), (0, j * s_bord)), np.float32)
            square_j = square_i + shift_j + shift_bord
            square_verts.append(square_j)
    return np.array(square_verts, np.float32), mac_norm


def get_square_centres(c_err: float = 0.05, scale: int = scale) -> np.ndarray:
    verts, mac_norm = get_square_verts(c_err, scale=scale)
    centres = np.mean(verts, axis=1)
    return np.array(centres, np.float32)
