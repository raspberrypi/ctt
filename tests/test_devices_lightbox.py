# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the generic ctt.devices lightbox API and driver registry.

import pytest

from ctt.devices import Lightbox, LightboxError, get_lightbox, register_driver, registry


class FakeBox(Lightbox):
    """Minimal in-memory Lightbox for exercising the generic layer (no hardware)."""

    _ILLUM = {1: 'F12', 4: 'D65'}

    def __init__(self):
        self._channel = 1
        self._intensity = {1: 100.0, 4: 80.0}
        self.closed = False

    @classmethod
    def probe(cls, serial=None):
        return cls()

    @property
    def illuminants(self):
        return self._ILLUM

    def set_intensity(self, channel, percent):
        self._channel = channel
        self._intensity[channel] = percent

    def get_intensity(self):
        return self._intensity.get(self._channel, 0.0)

    def get_channel(self):
        return self._channel

    def set_channel(self, channel):
        self._channel = channel

    def get_default_intensity(self, channel):
        return 100.0

    def close(self):
        self.closed = True


@pytest.fixture
def clean_registry():
    """Restore the driver registry after each test that mutates it."""
    saved = list(registry.DRIVERS)
    registry._SHARED = None
    yield
    registry.DRIVERS[:] = saved
    registry._SHARED = None


def test_set_illuminant_resolves_name_case_insensitively():
    box = FakeBox()
    assert box.set_illuminant('d65') == 4
    assert box.get_channel() == 4
    box.set_illuminant('F12', 50)
    assert box.get_channel() == 1
    assert box.get_intensity() == 50


def test_set_illuminant_unknown_name_raises():
    with pytest.raises(LightboxError, match='unknown illuminant'):
        FakeBox().set_illuminant('nope')


def test_off_zeroes_active_channel():
    box = FakeBox()
    box.set_illuminant('D65', 90)
    box.off()
    assert box.get_intensity() == 0


def test_info_snapshot():
    box = FakeBox()
    box.set_illuminant('D65')
    info = box.info()
    assert info['channel'] == 4
    assert info['illuminant'] == 'D65'
    assert 'intensity' in info and 'model' in info


def test_context_manager_closes():
    box = FakeBox()
    with box as b:
        assert b is box
    assert box.closed


def test_registry_factory_returns_registered_driver(clean_registry):
    register_driver(FakeBox)
    box = get_lightbox()
    # The real LightStudioS driver probes first and finds nothing (no hardware),
    # so the factory should fall through to our FakeBox.
    assert isinstance(box, FakeBox)


def test_registry_skips_absent_driver(clean_registry):
    class Absent(Lightbox):
        @classmethod
        def probe(cls, serial=None):
            return None  # never present

        illuminants = {}

        def set_intensity(self, channel, percent): ...
        def get_intensity(self):
            return 0.0

        def get_channel(self):
            return 0

        def set_channel(self, channel): ...
        def get_default_intensity(self, channel):
            return 0.0

        def close(self): ...

    registry.DRIVERS[:] = [Absent]
    with pytest.raises(LightboxError, match='no supported lightbox'):
        get_lightbox()

    register_driver(FakeBox)
    assert isinstance(get_lightbox(), FakeBox)
