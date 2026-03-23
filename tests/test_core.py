# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi

import numpy as np

from ctt.core.camera import _LogBuffer, get_col_lux
from ctt.core.image import Image

# --- _LogBuffer ---


class TestLogBuffer:
    def test_empty(self):
        buf = _LogBuffer()
        assert str(buf) == ''

    def test_initial(self):
        buf = _LogBuffer('hello')
        assert str(buf) == 'hello'

    def test_accumulate(self):
        buf = _LogBuffer()
        buf += 'hello'
        buf += ' world'
        assert str(buf) == 'hello world'

    def test_iadd_returns_self(self):
        buf = _LogBuffer()
        result = buf.__iadd__('test')
        assert result is buf


# --- get_col_lux ---


class TestGetColLux:
    def test_col_and_lux(self):
        assert get_col_lux('d65_5500k_1000l.dng') == (5500, 1000)

    def test_col_only(self):
        assert get_col_lux('alsc_3000k.dng') == (3000, None)

    def test_uppercase_k(self):
        assert get_col_lux('image_4000K_500L.dng') == (4000, 500)

    def test_no_col(self):
        assert get_col_lux('invalid.dng') == (None, None)

    def test_no_extension(self):
        assert get_col_lux('image_5000k') == (None, None)

    def test_jpg_extension(self):
        assert get_col_lux('image_5000k_200l.jpg') == (5000, 200)

    def test_col_with_suffix(self):
        assert get_col_lux('alsc_3000k_0.dng') == (3000, None)


# --- Image.get_patches ---


class TestImageGetPatches:
    def _make_image(self, h=64, w=64, sigbits=10):
        img = Image()
        img.sigbits = sigbits
        img.channels = [np.full((h, w), 100, dtype=np.int64) for _ in range(4)]
        return img

    def test_basic_extraction(self):
        img = self._make_image()
        coords = [[[32, 32]]]  # Single patch at centre
        img.get_patches(coords)
        assert img.patches is not None
        assert len(img.patches) == 4  # One per channel
        assert len(img.patches[0]) == 1  # One patch

    def test_patch_size(self):
        img = self._make_image()
        coords = [[[32, 32]]]
        img.get_patches(coords, size=8)
        assert len(img.patches[0][0]) == 64  # 8x8 flattened

    def test_not_saturated(self):
        img = self._make_image(sigbits=10)
        coords = [[[32, 32]]]
        img.get_patches(coords)
        assert img.saturated is False

    def test_saturated_detection(self):
        img = self._make_image(sigbits=10)
        # Fill with the saturation value: (2^10 - 1) * 2^(16-10) = 1023 * 64
        sat_val = (2**10 - 1) * 2 ** (16 - 10)
        for ch in img.channels:
            ch[:] = sat_val
        coords = [[[32, 32]]]
        img.get_patches(coords)
        assert img.saturated is True

    def test_edge_clamping(self):
        img = self._make_image()
        # Patch at corner should be clamped, not error
        coords = [[[0, 0]]]
        img.get_patches(coords)
        assert img.patches is not None

    def test_multiple_patches(self):
        img = self._make_image()
        coords = [[[10, 10], [30, 30], [50, 50]]]
        img.get_patches(coords)
        assert len(img.patches[0]) == 3
