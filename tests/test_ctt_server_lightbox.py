# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the ctt-server lightbox API routes. The lightbox accessor is
# monkeypatched, so these run without pyusb or hardware.

import pytest

from ctt.devices import LightboxError, LightboxState
from ctt_server import app as app_module


class FakeBox:
    """A stand-in lightbox recording what the routes ask of it."""

    illuminants = {1: 'F12', 4: 'D65'}

    def __init__(self):
        self._channel = 1
        self._intensity = 100.0
        self.calls = []

    def info(self):
        return {
            'model': 'FakeBox',
            'serial': 'FB-1',
            'channel': self._channel,
            'illuminant': self.illuminants.get(self._channel),
            'intensity': self._intensity,
            'illuminants': self.illuminants,
        }

    def get_state(self):
        return LightboxState(self._channel, self.illuminants.get(self._channel), self._intensity)

    def set_illuminant(self, illuminant, percent=None):
        self.calls.append(('set_illuminant', illuminant, percent))
        if isinstance(illuminant, int):
            self._channel = illuminant
        else:
            self._channel = next(c for c, n in self.illuminants.items() if n.lower() == str(illuminant).lower())
        self._intensity = 100.0 if percent is None else percent

    def off(self):
        self.calls.append(('off',))
        self._intensity = 0.0


@pytest.fixture
def client(tmp_path):
    return app_module.create_app(tmp_path).test_client()


def test_status_present(client, monkeypatch):
    box = FakeBox()
    monkeypatch.setattr(app_module, 'get_shared_lightbox', lambda: box)
    r = client.get('/api/lightbox')
    data = r.get_json()
    assert r.status_code == 200
    assert data['present'] is True
    assert data['model'] == 'FakeBox'
    assert data['illuminants'] == {'1': 'F12', '4': 'D65'}  # JSON stringifies int keys


def test_status_absent(client, monkeypatch):
    def _raise():
        raise LightboxError('pyusb not available')

    monkeypatch.setattr(app_module, 'get_shared_lightbox', _raise)
    r = client.get('/api/lightbox')
    assert r.status_code == 200
    assert r.get_json() == {'present': False}


def test_post_sets_intensity_illuminant_and_off(client, monkeypatch):
    box = FakeBox()
    monkeypatch.setattr(app_module, 'get_shared_lightbox', lambda: box)

    assert client.post('/api/lightbox', json={'channel': 4, 'percent': 50}).status_code == 200
    assert ('set_illuminant', 4, 50.0) in box.calls

    assert client.post('/api/lightbox', json={'illuminant': 'D65'}).status_code == 200
    assert ('set_illuminant', 'D65', None) in box.calls

    assert client.post('/api/lightbox', json={'percent': 25}).status_code == 200
    assert ('set_illuminant', 4, 25.0) in box.calls  # bare percent targets the active channel

    assert client.post('/api/lightbox', json={'off': True}).status_code == 200
    assert ('off',) in box.calls


def test_post_when_absent_is_503(client, monkeypatch):
    monkeypatch.setattr(app_module, 'get_shared_lightbox', lambda: (_ for _ in ()).throw(LightboxError('no box')))
    assert client.post('/api/lightbox', json={'off': True}).status_code == 503
