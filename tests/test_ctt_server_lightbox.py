# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the ctt-server lightbox API routes. The lightbox accessor is
# monkeypatched, so these run without pyusb or hardware.

import pytest

from ctt.devices import LightboxError
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
        }

    def get_channel(self):
        return self._channel

    def set_intensity(self, channel, percent):
        self.calls.append(('set_intensity', channel, percent))
        self._channel, self._intensity = channel, percent

    def set_channel(self, channel):
        self.calls.append(('set_channel', channel))
        self._channel, self._intensity = channel, 100.0

    def set_illuminant(self, name, percent=None):
        self.calls.append(('set_illuminant', name, percent))
        self._channel = next(c for c, n in self.illuminants.items() if n.lower() == name.lower())

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
    assert ('set_intensity', 4, 50.0) in box.calls

    assert client.post('/api/lightbox', json={'illuminant': 'D65'}).status_code == 200
    assert ('set_illuminant', 'D65', None) in box.calls

    assert client.post('/api/lightbox', json={'off': True}).status_code == 200
    assert ('off',) in box.calls


def test_post_when_absent_is_503(client, monkeypatch):
    monkeypatch.setattr(app_module, 'get_shared_lightbox', lambda: (_ for _ in ()).throw(LightboxError('no box')))
    assert client.post('/api/lightbox', json={'off': True}).status_code == 503
