# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi

import numpy as np

from ctt.core.image import Image
from ctt.detection.patches import get_alsc_patches
from ctt.detection.ransac import get_square_centres, get_square_verts


class TestGetAlscPatches:
    def test_uint16_patches_promote_before_arithmetic(self):
        """Channels are stored as uint16: the black-level subtraction must not
        wrap below black, nor the two-green sum overflow before halving."""
        img = Image()
        img.col = 5000
        img.blacklevel_16 = 4096
        img.cen_coords = list(range(24))
        # 24 patches of 4 pixels per channel: reds sit below black level,
        # greens are bright enough that their sum exceeds the uint16 range.
        img.patches = [
            [np.full(4, 2000, dtype=np.uint16)] * 24,  # R: below black
            [np.full(4, 60000, dtype=np.uint16)] * 24,  # G1
            [np.full(4, 60000, dtype=np.uint16)] * 24,  # G2
            [np.full(4, 2000, dtype=np.uint16)] * 24,  # B: below black
        ]
        r, b, g = get_alsc_patches(img, None, grey=False)
        assert np.all(r == 2000 - 4096)
        assert np.all(b == 2000 - 4096)
        assert np.all(g == 60000 - 4096)


class TestGetSquareVerts:
    def test_returns_24_squares(self):
        verts, mac_norm = get_square_verts()
        assert verts.shape[0] == 24

    def test_each_square_has_4_vertices(self):
        verts, _ = get_square_verts()
        assert verts.shape[1] == 4

    def test_vertices_2d(self):
        verts, _ = get_square_verts()
        assert verts.shape[2] == 2

    def test_all_vertices_within_chart(self):
        verts, mac_norm = get_square_verts()
        x_max = mac_norm[0, 3, 0]  # top-right x
        y_max = mac_norm[0, 1, 1]  # bottom-left y
        assert np.all(verts[:, :, 0] >= 0)
        assert np.all(verts[:, :, 1] >= 0)
        assert np.all(verts[:, :, 0] <= x_max)
        assert np.all(verts[:, :, 1] <= y_max)

    def test_chart_boundary_shape(self):
        _, mac_norm = get_square_verts()
        assert mac_norm.shape == (1, 4, 2)

    def test_different_scale(self):
        verts_s1, _ = get_square_verts(scale=1)
        verts_s2, _ = get_square_verts(scale=2)
        # Scale 2 should produce coordinates roughly 2x those of scale 1
        ratio = np.mean(verts_s2) / np.mean(verts_s1)
        assert ratio > 1.9 and ratio < 2.1


class TestGetSquareCentres:
    def test_returns_24_centres(self):
        centres = get_square_centres()
        assert centres.shape[0] == 24

    def test_centres_2d(self):
        centres = get_square_centres()
        assert centres.shape[1] == 2

    def test_centres_within_squares(self):
        verts, _ = get_square_verts()
        centres = get_square_centres()
        for i in range(24):
            cx, cy = centres[i]
            xs = verts[i, :, 0]
            ys = verts[i, :, 1]
            assert cx >= np.min(xs) and cx <= np.max(xs)
            assert cy >= np.min(ys) and cy <= np.max(ys)
