# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the ctt-server light-meter API routes. The meter accessor is monkeypatched,
# so these run without any backend or hardware.

import pytest

from ctt_server import app as app_module
from devices import LightmeterError, Measurement


class FakeMeter:
    """A stand-in light meter recording what the routes ask of it."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = []
        self._latest = None

    def info(self):
        return {'model': 'FakeMeter', 'serial': 'FM-1'}

    def measure(self):
        self.calls.append('measure')
        if self.fail:
            raise LightmeterError('measurement failed')
        self._latest = Measurement(illuminance_lux=234.5, cct=6543.0, duv=0.0021)
        return self._latest

    def read_latest(self):
        return self._latest


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


def test_post_sample_returns_reading(client, monkeypatch):
    meter = FakeMeter()
    monkeypatch.setattr(app_module, 'get_shared_lightmeter', lambda: meter)
    r = client.post('/api/lightmeter', json={'action': 'sample'})
    data = r.get_json()
    assert r.status_code == 200
    assert 'measure' in meter.calls
    assert data['reading']['illuminance_lux'] == 234.5


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
