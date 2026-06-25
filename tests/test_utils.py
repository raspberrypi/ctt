# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi

import numpy as np
import pytest

from ctt.utils.colorspace import rgb_to_lab, rgb_to_uv
from ctt.utils.errors import ArgError
from ctt.utils.tools import correlate, get_alsc_colour_cals, nudge_for_json, read_manifest, reshape

# --- nudge_for_json ---


class TestNudgeForJson:
    def test_normal_values_unchanged(self):
        arr = np.array([1.234, 2.567, 3.891])
        result = nudge_for_json(arr, decimals=3)
        np.testing.assert_array_equal(result, np.round(arr, 3))

    def test_nudges_trailing_zero(self):
        # 1.200 → factor*arr % 1 = 12.0 % 1 = 0.0 ≤ 0.05, so nudged up
        arr = np.array([1.200])
        result = nudge_for_json(arr, decimals=3)
        assert result[0] != 1.200

    def test_nudges_trailing_nine(self):
        # decimals=1, factor=1: (1 * 0.99) % 1 = 0.99 ≥ 0.95 → nudged down
        # Without nudge: round(0.99, 1) = 1.0; with nudge: round(0.89, 1) = 0.9
        arr = np.array([0.99])
        result = nudge_for_json(arr, decimals=1)
        assert result[0] == pytest.approx(0.9)

    def test_decimals_4(self):
        arr = np.array([0.5, 1.0, 2.5])
        result = nudge_for_json(arr, decimals=4)
        for val in result:
            # All results should round to 4 dp
            assert val == round(val, 4)

    def test_empty_array(self):
        arr = np.array([])
        result = nudge_for_json(arr)
        assert len(result) == 0


# --- correlate ---


class TestCorrelate:
    def test_identical(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        assert correlate(a, a) == pytest.approx(1.0)

    def test_anticorrelated(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        assert correlate(a, b) == pytest.approx(-1.0)

    def test_constant_array(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.ones(3)
        # Correlation with constant is NaN
        assert np.isnan(correlate(a, b))


# --- get_alsc_colour_cals ---


class TestGetAlscColourCals:
    def test_missing_key_returns_none(self):
        assert get_alsc_colour_cals({}) is None
        assert get_alsc_colour_cals({'rpi.alsc': {}}) is None

    def test_valid_json(self):
        cam_json = {
            'rpi.alsc': {
                'calibrations_Cr': [
                    {'ct': 3000, 'table': np.array([2.0, 4.0, 6.0])},
                    {'ct': 5000, 'table': np.array([1.0, 3.0, 5.0])},
                ],
                'calibrations_Cb': [
                    {'ct': 3000, 'table': np.array([3.0, 6.0, 9.0])},
                    {'ct': 5000, 'table': np.array([2.0, 4.0, 8.0])},
                ],
            }
        }
        result = get_alsc_colour_cals(cam_json)
        assert result is not None
        assert set(result.keys()) == {3000, 5000}
        # Tables should be normalised so min is 1.0
        for _ct, (cr, cb) in result.items():
            assert np.min(cr) == pytest.approx(1.0)
            assert np.min(cb) == pytest.approx(1.0)


# --- reshape ---


class TestReshape:
    def test_output_dimensions(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        resized, factor = reshape(img, 50)
        assert resized.shape[0] == 50
        assert factor == pytest.approx(0.5)

    def test_factor_calculation(self):
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        _, factor = reshape(img, 100)
        assert factor == pytest.approx(0.5)


# --- rgb_to_lab ---


class TestRgbToLab:
    def test_black(self):
        lab = rgb_to_lab([0, 0, 0])
        assert lab[0] == pytest.approx(0.0, abs=0.01)

    def test_white(self):
        lab = rgb_to_lab([255, 255, 255])
        assert lab[0] == pytest.approx(100.0, abs=0.5)

    def test_red_positive_a(self):
        lab = rgb_to_lab([255, 0, 0])
        assert lab[1] > 0  # red has positive a*

    def test_green_negative_a(self):
        lab = rgb_to_lab([0, 255, 0])
        assert lab[1] < 0  # green has negative a*

    def test_blue_negative_b(self):
        lab = rgb_to_lab([0, 0, 255])
        assert lab[2] < 0  # blue has negative b*

    def test_batch(self):
        rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        lab = rgb_to_lab(rgb)
        assert lab.shape == (3, 3)


# --- rgb_to_uv ---


class TestRgbToUv:
    def test_neutral_is_d65(self):
        # Any neutral grey sits at the D65 white point in u'v'.
        assert rgb_to_uv([0.5, 0.5, 0.5]) == pytest.approx([0.1978, 0.4683], abs=1e-3)

    def test_primaries_match_srgb(self):
        # Pure linear-sRGB primaries land on the known sRGB primary chromaticities.
        assert rgb_to_uv([1, 0, 0]) == pytest.approx([0.4507, 0.5229], abs=1e-3)
        assert rgb_to_uv([0, 1, 0]) == pytest.approx([0.1250, 0.5625], abs=1e-3)
        assert rgb_to_uv([0, 0, 1]) == pytest.approx([0.1754, 0.1579], abs=1e-3)

    def test_scale_invariant(self):
        # Chromaticity depends only on direction, not magnitude.
        assert rgb_to_uv([0.2, 0.4, 0.1]) == pytest.approx(rgb_to_uv([2.0, 4.0, 1.0]), abs=1e-9)

    def test_batch(self):
        assert rgb_to_uv(np.array([[1, 0, 0], [0, 1, 0]])).shape == (2, 2)


class TestGamutReference:
    def test_primaries_and_whitepoint(self):
        from ctt.utils.colorspace import gamut_reference

        g = gamut_reference()
        assert set(g) == {'srgb', 'rec2020', 'd65', 'locus'}
        # sRGB red primary and D65 white, in u'v', from the colour library.
        assert g['srgb'][0] == pytest.approx([0.4507, 0.5229], abs=1e-3)
        assert g['d65'] == pytest.approx([0.1978, 0.4683], abs=1e-3)
        # Rec.2020 has a wider gamut: its green sits at a lower u' than sRGB's.
        assert g['rec2020'][1][0] < g['srgb'][1][0]
        assert len(g['locus']) > 10 and all(len(p) == 2 for p in g['locus'])


# --- read_manifest ---


class TestReadManifest:
    def _dir_with(self, tmp_path, names):
        for n in names:
            (tmp_path / n).write_bytes(b'DNG')
        return str(tmp_path)

    def test_happy_path_comments_blanks_dedupe(self, tmp_path):
        d = self._dir_with(tmp_path, ['a.dng', 'b.dng'])
        m = tmp_path / 'manifest.txt'
        m.write_text('# calibration subset\n\na.dng\nb.dng\na.dng\n')
        assert read_manifest(str(m), d) == ['a.dng', 'b.dng']

    def test_missing_file_named_in_error(self, tmp_path):
        d = self._dir_with(tmp_path, ['a.dng'])
        m = tmp_path / 'manifest.txt'
        m.write_text('a.dng\nmissing.dng\n')
        with pytest.raises(ArgError, match='missing.dng'):
            read_manifest(str(m), d)

    def test_path_entry_rejected(self, tmp_path):
        m = tmp_path / 'manifest.txt'
        m.write_text('sub/a.dng\n')
        with pytest.raises(ArgError, match='bare filenames'):
            read_manifest(str(m), str(tmp_path))

    def test_non_dng_rejected(self, tmp_path):
        m = tmp_path / 'manifest.txt'
        m.write_text('a.jpg\n')
        with pytest.raises(ArgError, match='.dng'):
            read_manifest(str(m), str(tmp_path))

    def test_empty_manifest_rejected(self, tmp_path):
        m = tmp_path / 'manifest.txt'
        m.write_text('# nothing here\n\n')
        with pytest.raises(ArgError, match='lists no images'):
            read_manifest(str(m), str(tmp_path))

    def test_unreadable_manifest(self, tmp_path):
        with pytest.raises(ArgError, match='Could not read'):
            read_manifest(str(tmp_path / 'nope.txt'), str(tmp_path))
