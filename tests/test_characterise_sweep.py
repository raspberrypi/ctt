# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the live exposure sweep: a synthetic-sensor camera stands in for
# Picamera2, so the whole path — probing, sweeping, fitting, results merge and
# control restoration — runs hardware-free with known ground truth.

import json

import numpy as np
import pytest

from ctt_server import characterise, sessions

DN_PER_E = 1.5  # sensor response at gain 1 (K = 1/1.5 e-/DN16)
READ_NOISE_DN = 15.0
BLACK_16 = 4096.0
# Electron full well above the DN ceiling at gain 1: the container clips first,
# as real sensors do at base gain, so the effective full well is swing / DN_PER_E.
FULL_WELL_E = 45_000
SWING = 65535.0 - BLACK_16
EFFECTIVE_FULL_WELL_E = SWING / DN_PER_E


class FakeCamera:
    """A synthetic sensor behind the Picamera2Camera control/burst interface."""

    def __init__(self, flux_e_per_us=2.0):
        self.flux = flux_e_per_us
        self.controls = {'auto_exposure': True, 'exposure': 10_000, 'gain': 1.0, 'fps': 30, 'exposure_min': 13}
        self.set_history = []
        self.rng = np.random.default_rng(0)

    def get_controls(self):
        return dict(self.controls)

    def set_controls(self, controls):
        self.set_history.append(dict(controls))
        self.controls.update({k: v for k, v in controls.items() if v is not None})
        return self.get_controls()

    def capture_raw_burst(self, frames, exposure_us, gain, roi_fraction=0.5):
        mean_e = self.flux * exposure_us
        shape = (32, 32)
        crops = []
        for _ in range(frames):
            electrons = np.minimum(self.rng.poisson(mean_e, shape), FULL_WELL_E)
            dn = BLACK_16 + DN_PER_E * gain * electrons + self.rng.normal(0, READ_NOISE_DN, shape)
            crops.append(np.clip(dn, 0, 65535))
        return {
            'frames': crops,
            'exposure_us': int(exposure_us),
            'gain': float(gain),
            'blacklevel_16': BLACK_16,
            'sigbits': 12,
        }


class BrokenCamera(FakeCamera):
    def capture_raw_burst(self, *a, **kw):
        raise RuntimeError('controls did not settle')


@pytest.fixture
def project(tmp_path):
    return sessions.Workspace(tmp_path).create_project('imx662')


def _run(project, camera, **kw):
    return list(characterise.sweep_stream(project, camera, **kw))


def test_sweep_recovers_conversion_gain_end_to_end(project):
    cam = FakeCamera()
    lines = _run(project, cam, gains=[1.0], points_per_gain=10, frames=8)
    assert lines[-1] == 'CHAR_EXIT 0'

    results = json.loads(characterise.results_path(project).read_text())
    fits = results['ptc']['fits']
    assert len(fits) == 1 and fits[0]['reliable']
    assert fits[0]['k_e_per_dn'] == pytest.approx(1.0 / DN_PER_E, rel=0.05)
    assert results['ptc']['unavailable_reason'] is None
    assert results['linearity']['available'] and results['linearity']['r2'] > 0.999
    # The deliberate over-exposure point makes the full well observable; at
    # base gain the container clips first, so it is the DN-limited well.
    assert results['full_well']['available']
    assert results['full_well']['full_well_e'] == pytest.approx(EFFECTIVE_FULL_WELL_E, rel=0.10)
    assert results['dynamic_range']['available']
    expected_dr = 20 * np.log10(EFFECTIVE_FULL_WELL_E / (READ_NOISE_DN / DN_PER_E))
    assert results['dynamic_range']['db'] == pytest.approx(expected_dr, abs=1.5)
    assert len(results['snr_curve']) > 3


def test_sweep_gain_families_scale(project):
    cam = FakeCamera()
    lines = _run(project, cam, gains=[1.0, 2.0], points_per_gain=8, frames=8)
    assert lines[-1] == 'CHAR_EXIT 0'
    results = json.loads(characterise.results_path(project).read_text())
    summaries = {g['gain']: g for g in results['gain_sweep']['gains']}
    assert results['gain_sweep']['available']
    assert summaries[2.0]['k_e_per_dn'] == pytest.approx(summaries[1.0]['k_e_per_dn'] / 2.0, rel=0.10)


def test_sweep_merges_into_offline_results(project):
    # Pre-existing offline results: a dark section awaiting a conversion gain.
    path = characterise.results_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                'version': 1,
                'inputs': [],
                'groups': [],
                'dark': {
                    'available': True,
                    'read_noise_dn': 15.96,
                    'dsnu_dn': 5.82,
                    'gain': 1.0,
                    'read_noise_e': None,
                    'dsnu_e': None,
                },
                'ptc': {
                    'points': [
                        {
                            'mean_dn': 1000.0,
                            'var_dn2': 2000.0,
                            'exposure_us': 100,
                            'gain': 1.07,
                            'n_frames': 8,
                            'clipped': False,
                            'label': 'alsc_3500k',
                            'source': 'flat',
                        }
                    ],
                    'fits': [],
                    'unavailable_reason': 'x',
                },
                'prnu': {'available': True, 'groups': [], 'best_pct': 0.57},
                'warnings': [],
            }
        )
    )
    lines = _run(project, FakeCamera(), gains=[1.0], points_per_gain=8, frames=8)
    assert lines[-1] == 'CHAR_EXIT 0'
    results = json.loads(path.read_text())
    # Offline sections survive; the flat point is still there alongside live ones.
    assert results['prnu']['best_pct'] == 0.57
    sources = {p['source'] for p in results['ptc']['points']}
    assert sources == {'flat', 'live'}
    # The dark metrics gain electron units from the sweep's K.
    assert results['dark']['read_noise_e'] == pytest.approx(15.96 / DN_PER_E, rel=0.06)
    assert results['dark']['dsnu_e'] is not None


def test_sweep_restores_camera_controls(project):
    cam = FakeCamera()
    _run(project, cam, gains=[1.0], points_per_gain=8, frames=4)
    final = cam.set_history[-1]
    assert final['auto_exposure'] is True and final['awb'] is True and final['fps'] == 30


def test_sweep_failure_still_restores_and_exits(project):
    cam = BrokenCamera()
    lines = _run(project, cam, gains=[1.0])
    assert lines[-1] == 'CHAR_EXIT 1'
    assert any(ln.startswith('ERROR:') for ln in lines)
    assert cam.set_history[-1]['auto_exposure'] is True


def test_sweep_route_mode_live(project, tmp_path, monkeypatch):
    from ctt_server import app as app_module

    cam = FakeCamera()
    monkeypatch.setattr(app_module, 'get_shared_camera', lambda: cam)
    client = app_module.create_app(str(tmp_path)).test_client()
    r = client.get('/projects/imx662/characterisation/analyse/stream?mode=live&gains=1&points=8&frames=4')
    assert r.status_code == 200
    lines = [json.loads(x[6:]) for x in r.data.decode().splitlines() if x.startswith('data: ')]
    assert lines[-1] == 'CHAR_EXIT 0'
    assert any('live sweep' in ln for ln in lines)
