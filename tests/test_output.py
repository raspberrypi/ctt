# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi

import json

import numpy as np
import pytest

from ctt.output.converter import convert_target, convert_v2, interp_2d
from ctt.output.json_formatter import Encoder, pretty_print

# --- pretty_print ---


class TestPrettyPrint:
    @pytest.fixture()
    def valid_json(self):
        return {
            'version': 2.0,
            'target': 'pisp',
            'algorithms': [{'rpi.black_level': {'black_level': 4096}}],
        }

    def test_valid_json(self, valid_json):
        result = pretty_print(valid_json)
        assert isinstance(result, str)
        # Output should be parseable JSON
        parsed = json.loads(result)
        assert parsed['version'] == 2.0

    def test_missing_version_raises(self):
        with pytest.raises(RuntimeError):
            pretty_print({'target': 'pisp', 'algorithms': []})

    def test_missing_target_raises(self):
        with pytest.raises(RuntimeError):
            pretty_print({'version': 2.0, 'algorithms': []})

    def test_missing_algorithms_raises(self):
        with pytest.raises(RuntimeError):
            pretty_print({'version': 2.0, 'target': 'pisp'})

    def test_old_version_raises(self):
        with pytest.raises(RuntimeError):
            pretty_print({'version': 1.0, 'target': 'pisp', 'algorithms': []})

    def test_custom_elems(self, valid_json):
        valid_json['algorithms'] = [{'rpi.awb': {'ct_curve': [3000, 0.5, 0.4, 5000, 0.3, 0.6]}}]
        result = pretty_print(valid_json, custom_elems={'ct_curve': 3})
        # ct_curve should be chunked in groups of 3
        assert 'ct_curve' in result


# --- Encoder ---


class TestEncoder:
    def test_flat_list_inline(self):
        enc = Encoder(indent=4, sort_keys=False)
        result = enc.encode([1, 2, 3])
        assert '1' in result and '2' in result and '3' in result

    def test_dict_encoding(self):
        enc = Encoder(indent=4, sort_keys=False)
        result = enc.encode({'key': 'value'})
        parsed = json.loads(result)
        assert parsed['key'] == 'value'

    def test_nested_list(self):
        enc = Encoder(indent=4, sort_keys=False)
        result = enc.encode([[1, 2], [3, 4]])
        assert '1' in result

    def test_empty_dict(self):
        enc = Encoder(indent=4, sort_keys=False)
        result = enc.encode({'empty': {}})
        assert '{ }' in result


# --- interp_2d ---


class TestInterp2d:
    def test_identity(self):
        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = interp_2d(grid, 2, 2, 2, 2)
        np.testing.assert_array_almost_equal(result, grid)

    def test_corners_preserved(self):
        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = interp_2d(grid, 2, 2, 4, 4)
        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, -1] == pytest.approx(2.0)
        assert result[-1, 0] == pytest.approx(3.0)
        assert result[-1, -1] == pytest.approx(4.0)

    def test_upscale_shape(self):
        grid = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = interp_2d(grid, 3, 2, 6, 4)
        assert result.shape == (4, 6)

    def test_uniform_grid_stays_uniform(self):
        grid = np.ones((3, 3)) * 5.0
        result = interp_2d(grid, 3, 3, 6, 6)
        np.testing.assert_array_almost_equal(result, np.ones((6, 6)) * 5.0)


# --- convert_target / convert_v2 ---


def _make_pisp_json():
    """Build a minimal PiSP tuning JSON with ALSC tables and AGC."""
    rng = np.random.default_rng(42)
    pisp_w, pisp_h = 32, 32
    table_size = pisp_w * pisp_h

    alsc_table_cr = np.round(1.0 + rng.random(table_size) * 0.5, 3).tolist()
    alsc_table_cb = np.round(1.0 + rng.random(table_size) * 0.3, 3).tolist()
    luminance_lut = np.round(1.0 + rng.random(table_size) * 0.2, 3).tolist()

    return {
        'version': 2.0,
        'target': 'pisp',
        'algorithms': [
            {
                'rpi.alsc': {
                    'omega': 1.3,
                    'n_iter': 100,
                    'luminance_strength': 0.8,
                    'calibrations_Cr': [{'ct': 3000, 'table': alsc_table_cr}],
                    'calibrations_Cb': [{'ct': 3000, 'table': alsc_table_cb}],
                    'luminance_lut': luminance_lut,
                }
            },
            {
                'rpi.denoise': {
                    'normal': {'noise_constant': 0, 'noise_slope': 2.0},
                }
            },
            {
                'rpi.agc': {
                    'channels': [
                        {
                            'metering_modes': {
                                'centre-weighted': {'weights': [1] * 225},
                                'spot': {'weights': [0] * 225},
                                'matrix': {'weights': [1] * 225},
                            },
                            'exposure_modes': {'normal': {'shutter': [100], 'gain': [1.0]}},
                            'constraint_modes': {'normal': []},
                        }
                    ]
                }
            },
            {'rpi.hdr': {'cadence': [1, 2]}},
            {'rpi.ccm': {'ccms': [{'ct': 3000, 'ccm': [1, 0, 0, 0, 1, 0, 0, 0, 1]}]}},
        ],
    }


def _make_vc4_json():
    """Build a minimal VC4 tuning JSON with ALSC tables and AGC."""
    rng = np.random.default_rng(42)
    vc4_w, vc4_h = 16, 12
    table_size = vc4_w * vc4_h

    alsc_table_cr = np.round(1.0 + rng.random(table_size) * 0.5, 3).tolist()
    alsc_table_cb = np.round(1.0 + rng.random(table_size) * 0.3, 3).tolist()
    luminance_lut = np.round(1.0 + rng.random(table_size) * 0.2, 3).tolist()

    return {
        'version': 2.0,
        'target': 'bcm2835',
        'algorithms': [
            {
                'rpi.alsc': {
                    'omega': 1.3,
                    'n_iter': 100,
                    'luminance_strength': 0.7,
                    'calibrations_Cr': [{'ct': 3000, 'table': alsc_table_cr}],
                    'calibrations_Cb': [{'ct': 3000, 'table': alsc_table_cb}],
                    'luminance_lut': luminance_lut,
                }
            },
            {
                'rpi.sdn': {
                    'noise_constant': 0,
                    'noise_slope': 2.0,
                }
            },
            {
                'rpi.agc': {
                    'metering_modes': {
                        'centre-weighted': {'weights': [1] * 15},
                        'spot': {'weights': [0] * 15},
                        'matrix': {'weights': [1] * 15},
                    },
                    'exposure_modes': {'normal': {'shutter': [100], 'gain': [1.0]}},
                    'constraint_modes': {'normal': []},
                }
            },
            {'rpi.ccm': {'ccms': [{'ct': 3000, 'ccm': [1, 0, 0, 0, 1, 0, 0, 0, 1]}]}},
        ],
    }


class TestConvertTarget:
    def test_vc4_to_pisp_alsc_table_size(self):
        vc4_json = _make_vc4_json()
        result = convert_target(vc4_json, 'pisp')
        alsc = next(a for a in result['algorithms'] if 'rpi.alsc' in a)['rpi.alsc']
        # PiSP grid is 32x32 = 1024
        assert len(alsc['calibrations_Cr'][0]['table']) == 32 * 32
        assert len(alsc['calibrations_Cb'][0]['table']) == 32 * 32
        assert len(alsc['luminance_lut']) == 32 * 32

    def test_vc4_to_pisp_agc_weights_replaced(self):
        vc4_json = _make_vc4_json()
        result = convert_target(vc4_json, 'pisp')
        agc = next(a for a in result['algorithms'] if 'rpi.agc' in a)['rpi.agc']
        # VC4 flat metering_modes should get PiSP template weights (225 elements)
        weights = agc['metering_modes']['centre-weighted']['weights']
        assert len(weights) == 225

    def test_vc4_to_pisp_sdn_becomes_denoise(self):
        vc4_json = _make_vc4_json()
        result = convert_target(vc4_json, 'pisp')
        algo_keys = [k for a in result['algorithms'] for k in a]
        assert 'rpi.denoise' in algo_keys
        assert 'rpi.sdn' not in algo_keys

    def test_vc4_to_pisp_uniform_alsc(self):
        vc4_json = _make_vc4_json()
        alsc = next(a for a in vc4_json['algorithms'] if 'rpi.alsc' in a)['rpi.alsc']
        alsc['calibrations_Cr'][0]['table'] = [1.0] * (16 * 12)
        alsc['calibrations_Cb'][0]['table'] = [1.0] * (16 * 12)
        alsc['luminance_lut'] = [1.0] * (16 * 12)

        result = convert_target(vc4_json, 'pisp')
        alsc_out = next(a for a in result['algorithms'] if 'rpi.alsc' in a)['rpi.alsc']
        np.testing.assert_array_almost_equal(alsc_out['calibrations_Cr'][0]['table'], [1.0] * (32 * 32))

    def test_vc4_to_pisp_preserves_alsc_range(self):
        vc4_json = _make_vc4_json()
        alsc = next(a for a in vc4_json['algorithms'] if 'rpi.alsc' in a)['rpi.alsc']
        original_min = min(alsc['calibrations_Cr'][0]['table'])
        original_max = max(alsc['calibrations_Cr'][0]['table'])

        result = convert_target(vc4_json, 'pisp')
        alsc_out = next(a for a in result['algorithms'] if 'rpi.alsc' in a)['rpi.alsc']
        result_min = min(alsc_out['calibrations_Cr'][0]['table'])
        result_max = max(alsc_out['calibrations_Cr'][0]['table'])
        # Interpolation should not produce values outside the original range
        assert result_min >= original_min - 0.001
        assert result_max <= original_max + 0.001


class TestConvertV2:
    def test_vc4_to_pisp_produces_valid_json(self):
        vc4_json = _make_vc4_json()
        result_str = convert_v2(vc4_json, 'pisp')
        parsed = json.loads(result_str)
        assert parsed['target'] == 'pisp'
        assert parsed['version'] == 2.0

    def test_same_target_no_conversion(self):
        pisp_json = _make_pisp_json()
        alsc_before = next(a for a in pisp_json['algorithms'] if 'rpi.alsc' in a)['rpi.alsc']
        table_before = list(alsc_before['calibrations_Cr'][0]['table'])
        result_str = convert_v2(pisp_json, 'pisp')
        parsed = json.loads(result_str)
        alsc_after = next(a for a in parsed['algorithms'] if 'rpi.alsc' in a)['rpi.alsc']
        assert alsc_after['calibrations_Cr'][0]['table'] == table_before

    def test_v1_to_v2_conversion(self):
        v1_json = {
            'version': 1.0,
            'rpi.black_level': {'black_level': 4096},
        }
        result_str = convert_v2(v1_json, 'pisp')
        parsed = json.loads(result_str)
        assert parsed['version'] == 2.0
        assert parsed['target'] == 'pisp'
