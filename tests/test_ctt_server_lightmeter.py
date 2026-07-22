# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the ctt-server light-meter API routes. The meter accessor is monkeypatched,
# so these run without any backend or hardware.

import pytest

from ctt_server import app as app_module
from ctt_server.sessions import Workspace
from devices import LightmeterError, Measurement
from devices.lightmeter import MeasurementLimits, Spectrum


class FakeMeter:
    """A stand-in light meter recording what the routes ask of it."""

    LIMITS = MeasurementLimits(1.0, 200_000.0, colour_min_lux=5.0, cct_min=1563.0, cct_max=100_000.0)

    def __init__(self, fail: bool = False, under: bool = False):
        self.fail = fail
        self.under = under  # return an out-of-range reading
        self.calls = []
        self._latest = None

    def info(self):
        return {'model': 'FakeMeter', 'serial': 'FM-1', 'limits': self.LIMITS.to_dict()}

    @property
    def limits(self):
        return self.LIMITS

    def measure(self):
        self.calls.append('measure')
        if self.fail:
            raise LightmeterError('measurement failed')
        if self.under:
            self._latest = Measurement(illuminance_lux=-100.0, cct=0.0, in_range=False)
        else:
            self._latest = Measurement(
                illuminance_lux=234.5,
                cct=6543.0,
                duv=0.0021,
                spectrum_5nm=Spectrum(start_nm=380.0, step_nm=5.0, values=(0.5,) * 81),
                spectrum_1nm=Spectrum(start_nm=380.0, step_nm=1.0, values=(0.5,) * 401),
            )
        return self._latest

    def read_latest(self):
        return self._latest


class FakeCam:
    def capture_burst(self, frames, quality=95):
        return [(b'DNG', b'JPG', {'exposure': 1})] * frames


@pytest.fixture
def client(tmp_path):
    return app_module.create_app(tmp_path).test_client()


def test_status_present_is_read_only(client, monkeypatch):
    meter = FakeMeter()
    meter.measure()  # even with a stored reading, GET must not return or trigger one
    monkeypatch.setattr(app_module, 'get_shared_lightmeter', lambda: meter)
    r = client.get('/api/lightmeter')
    data = r.get_json()
    assert r.status_code == 200
    assert data['present'] is True
    assert data['model'] == 'FakeMeter'
    assert 'reading' not in data  # GET is identity-only; readings come from POST sample


def test_status_absent(client, monkeypatch):
    def _raise():
        raise LightmeterError('backend not available')

    monkeypatch.setattr(app_module, 'get_shared_lightmeter', _raise)
    r = client.get('/api/lightmeter')
    assert r.status_code == 200
    assert r.get_json() == {'present': False}


def test_status_includes_limits(client, monkeypatch):
    monkeypatch.setattr(app_module, 'get_shared_lightmeter', lambda: FakeMeter())
    data = client.get('/api/lightmeter').get_json()
    assert data['limits']['colour_min_lux'] == 5.0
    assert data['limits']['illuminance_min'] == 1.0


def test_post_sample_returns_reading(client, monkeypatch):
    meter = FakeMeter()
    monkeypatch.setattr(app_module, 'get_shared_lightmeter', lambda: meter)
    r = client.post('/api/lightmeter', json={'action': 'sample'})
    data = r.get_json()
    assert r.status_code == 200
    assert 'measure' in meter.calls
    assert data['reading']['illuminance_lux'] == 234.5
    assert data['reading']['in_range'] is True
    assert data['limits']['colour_min_lux'] == 5.0


def test_post_sample_caps_out_of_range(client, monkeypatch):
    # The API must never emit the device sentinels: illuminance is clamped to the
    # limit and the (invalid) colour metrics are dropped.
    monkeypatch.setattr(app_module, 'get_shared_lightmeter', lambda: FakeMeter(under=True))
    reading = client.post('/api/lightmeter', json={'action': 'sample'}).get_json()['reading']
    assert reading['in_range'] is False
    assert reading['illuminance_lux'] == 1.0  # clamped to the min, not -100
    assert 'cct' not in reading and 'cri_ra' not in reading  # colour dropped


def test_post_unknown_action_is_400(client, monkeypatch):
    monkeypatch.setattr(app_module, 'get_shared_lightmeter', lambda: FakeMeter())
    r = client.post('/api/lightmeter', json={'action': 'frobnicate'})
    assert r.status_code == 400
    assert 'unknown action' in r.get_json()['error']


def test_post_measure_error_is_400(client, monkeypatch):
    monkeypatch.setattr(app_module, 'get_shared_lightmeter', lambda: FakeMeter(fail=True))
    r = client.post('/api/lightmeter', json={'action': 'sample'})
    assert r.status_code == 400
    assert 'measurement failed' in r.get_json()['error']


def test_post_when_absent_is_503(client, monkeypatch):
    monkeypatch.setattr(app_module, 'get_shared_lightmeter', lambda: (_ for _ in ()).throw(LightmeterError('no meter')))
    assert client.post('/api/lightmeter', json={'action': 'sample'}).status_code == 503


# --- capture-time recording ---------------------------------------------------
@pytest.fixture
def capture_client(tmp_path, monkeypatch):
    Workspace(tmp_path).create_project('cam')
    monkeypatch.setattr(app_module, 'get_shared_camera', lambda: FakeCam())
    return app_module.create_app(tmp_path).test_client()


def test_capture_records_a_reading(capture_client, tmp_path, monkeypatch):
    meter = FakeMeter()
    monkeypatch.setattr(app_module, 'get_shared_lightmeter', lambda: meter)
    r = capture_client.post('/projects/cam/capture', json={'image_type': 'macbeth', 'colour_temp': 5000, 'lux': 800})
    assert r.status_code == 200
    recorded = r.get_json()['added'][0]['lightmeter']
    assert recorded['illuminance_lux'] == 234.5
    assert recorded['cct'] == 6543.0
    assert 'spectrum_5nm' in recorded
    assert 'spectrum_1nm' not in recorded  # trimmed to keep the sidecar compact
    # The reading is persisted in the project sidecar.
    proj = Workspace(tmp_path).get_project('cam')
    assert proj.captures[0].lightmeter['illuminance_lux'] == 234.5


def test_burst_frames_share_one_reading(capture_client, monkeypatch):
    meter = FakeMeter()
    monkeypatch.setattr(app_module, 'get_shared_lightmeter', lambda: meter)
    r = capture_client.post('/projects/cam/capture', json={'image_type': 'alsc', 'colour_temp': 5000, 'frames': 3})
    added = r.get_json()['added']
    assert len(added) == 3
    assert meter.calls == ['measure']  # one reading tags the whole burst
    assert all(c['lightmeter']['cct'] == 6543.0 for c in added)


def test_capture_without_meter_records_none(capture_client, monkeypatch):
    monkeypatch.setattr(app_module, 'get_shared_lightmeter', lambda: (_ for _ in ()).throw(LightmeterError('no meter')))
    r = capture_client.post('/projects/cam/capture', json={'image_type': 'macbeth', 'colour_temp': 5000, 'lux': 800})
    assert r.status_code == 200
    assert r.get_json()['added'][0]['lightmeter'] is None


def test_capture_survives_a_meter_failure(capture_client, monkeypatch):
    monkeypatch.setattr(app_module, 'get_shared_lightmeter', lambda: FakeMeter(fail=True))
    r = capture_client.post('/projects/cam/capture', json={'image_type': 'macbeth', 'colour_temp': 5000, 'lux': 800})
    assert r.status_code == 200
    assert r.get_json()['added'][0]['lightmeter'] is None


def test_capture_ignores_out_of_range_reading(capture_client, monkeypatch):
    # An under-range reading must not be recorded as if it were a real measurement.
    monkeypatch.setattr(app_module, 'get_shared_lightmeter', lambda: FakeMeter(under=True))
    r = capture_client.post('/projects/cam/capture', json={'image_type': 'macbeth', 'colour_temp': 5000, 'lux': 800})
    assert r.status_code == 200
    assert r.get_json()['added'][0]['lightmeter'] is None
