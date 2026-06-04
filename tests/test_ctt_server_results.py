# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for ctt_server.results: tuning-file parsing and metrics-sidecar merge.

import json

from ctt_server import results


def _write_tuning(tmp_path, target='pisp'):
    """Write a minimal but realistic tuning file and return its path."""
    cols, rows = results._GRID[target]
    # ALSC grid: centre gain ~1.0, corners higher (typical lens falloff).
    lut = [1.0] * (cols * rows)
    lut[0] = lut[cols - 1] = lut[-1] = lut[-cols] = 1.8  # four corners
    data = {
        'version': 2.0,
        'target': target,
        'algorithms': [
            {'rpi.black_level': {'black_level': 4096}},
            {'rpi.awb': {'ct_curve': [2500, 0.8, 0.5, 5000, 0.6, 0.7, 7000, 0.5, 0.9]}},
            {'rpi.ccm': {'ccms': [{'ct': 2500, 'ccm': [1] * 9}, {'ct': 5000, 'ccm': [1] * 9}]}},
            {'rpi.alsc': {'luminance_lut': lut}},
            {'rpi.lux': {'reference_lux': 998, 'reference_gain': 1.0, 'reference_shutter_speed': 2000}},
            {'rpi.noise': {'reference_constant': 12.0, 'reference_slope': 2.5}},
        ],
    }
    path = tmp_path / 'cam_pisp.json'
    path.write_text(json.dumps(data))
    return path


def test_deltae_banding():
    assert results._deltae_band(2.0) == 'good'
    assert results._deltae_band(4.0) == 'fair'
    assert results._deltae_band(9.0) == 'poor'
    assert results._deltae_band(None) is None


def test_ccm_quality_band_from_colour():
    # Band grades on the brightness-normalised colour ΔE (de_norm), not raw de:
    # here colour 2.0 → 'good' even though raw mean (6.0) alone would read 'poor'.
    q = results._ccm_quality(
        [
            {
                'ct': 2500,
                'patches': [{'de': 5.0, 'de_norm': 1.5}, {'de': 6.0, 'de_norm': 2.0}, {'de': 7.0, 'de_norm': 2.5}],
            },
        ]
    )
    assert q['mean'] == 6.0
    assert q['worst'] == 7.0
    assert q['colour'] == 2.0  # mean of de_norm
    assert q['band'] == 'good'


def test_ccm_quality_fallback_without_patches():
    # Older sidecars (no per-patch data) fall back to the per-CT metric_after,
    # and colour falls back to the raw mean.
    q = results._ccm_quality(
        [
            {'ct': 2500, 'metric_after': 1.2, 'max_after': 2.5},
            {'ct': 5000, 'metric_after': 2.0, 'max_after': 5.0},
        ]
    )
    assert q['mean'] == 1.6  # mean of metric_after
    assert q['worst'] == 5.0
    assert q['colour'] == 1.6
    assert q['band'] == 'good'


def test_alsc_falloff():
    f = results._alsc_falloff([[1.8, 1.0, 1.8], [1.0, 1.0, 1.0], [1.8, 1.0, 1.8]])
    assert f['centre'] == 1.0
    assert f['corner_max'] == 1.8


def test_parse_without_sidecar(tmp_path):
    out = results.parse_tuning_file(_write_tuning(tmp_path))
    assert out['summary']['target'] == 'pisp'
    assert out['summary']['black_level'] == 4096
    assert 'awb' in out['charts'] and 'alsc' in out['charts']
    # ALSC falloff derived from the grid.
    assert out['charts']['alsc']['centre'] == 1.0
    assert out['charts']['alsc']['corner_max'] == 1.8
    # No sidecar → no metrics block, and parsing must not fail.
    assert 'metrics' not in out


def test_parse_with_sidecar(tmp_path):
    json_path = _write_tuning(tmp_path)
    sidecar = {
        'target': 'pisp',
        'mode': 'Full',
        'ccm': [
            {
                'ct': 5000,
                'metric': 'average',
                'metric_before': 3.1,
                'metric_after': 1.2,
                'max_before': 8.0,
                'max_after': 2.5,
            },
            {
                'ct': 2500,
                'metric': 'average',
                'metric_before': 4.0,
                'metric_after': 2.0,
                'max_before': 9.0,
                'max_after': 5.0,
            },
        ],
        'warnings': [{'level': 'warn', 'message': 'Image too dark', 'image': 'd65_5000k_800l.dng'}],
        'counts': {'macbeth': 5, 'alsc': 3, 'cac': 0},
        'coverage': {'ct_min': 2500, 'ct_max': 7000, 'ct_count': 3, 'ccm_matrices': 2, 'awb_points': 3},
        'config': {'max_gain': 8.0, 'matrix_selection': 'average'},
    }
    metrics_path = tmp_path / 'cam_pisp_metrics.json'
    metrics_path.write_text(json.dumps(sidecar))

    out = results.parse_tuning_file(json_path, metrics_path)
    m = out['metrics']
    # CCM list sorted by CT for stable charting.
    assert [c['ct'] for c in m['ccm']] == [2500, 5000]
    # Band is driven by the mean (metric_after 1.2/2.0 → 1.6 → 'good'); worst is secondary.
    assert m['ccm_quality']['worst'] == 5.0
    assert m['ccm_quality']['band'] == 'good'
    assert m['warnings'][0]['message'] == 'Image too dark'
    assert m['counts']['macbeth'] == 5
    assert m['coverage']['ct_max'] == 7000
    assert m['config']['max_gain'] == 8.0


def test_parse_with_default_comparison(tmp_path):
    json_path = _write_tuning(tmp_path)
    default_path = tmp_path / 'default.json'
    default_path.write_text(
        json.dumps(
            {
                'version': 2.0,
                'target': 'pisp',
                'algorithms': [
                    {'rpi.awb': {'ct_curve': [3000, 0.9, 0.5, 5000, 0.7, 0.7]}},
                ],
            }
        )
    )
    sidecar = {
        'ccm': [{'ct': 5000, 'patches': [{'de': 1.0, 'de_norm': 1.0}, {'de': 3.0, 'de_norm': 2.0}]}],
        'ccm_default': [{'ct': 5000, 'patches': [{'de': 4.0, 'de_norm': 3.0}, {'de': 8.0, 'de_norm': 6.0}]}],
        'default_tuning_path': str(default_path),
    }
    metrics_path = tmp_path / 'cam_pisp_metrics.json'
    metrics_path.write_text(json.dumps(sidecar))

    m = results.parse_tuning_file(json_path, metrics_path)['metrics']
    assert m['ccm_quality']['colour'] == 1.5  # mean de_norm of new calibration
    assert m['ccm_default_quality']['colour'] == 4.5  # default tuning is worse
    assert m['ccm_default_quality']['band'] == 'poor'
    assert m['default']['charts']['awb']['points'][0]['ct'] == 3000  # default curves parsed


def test_parse_with_corrupt_sidecar(tmp_path):
    # A malformed sidecar must degrade gracefully (no metrics, no exception).
    json_path = _write_tuning(tmp_path)
    metrics_path = tmp_path / 'cam_pisp_metrics.json'
    metrics_path.write_text('{ not valid json')
    out = results.parse_tuning_file(json_path, metrics_path)
    assert 'metrics' not in out
