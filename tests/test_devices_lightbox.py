# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the generic devices lightbox API and driver registry.

import pytest

from devices import Lightbox, LightboxError, get_lightbox, register_lightbox_driver, registry


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

    def _set_intensity(self, channel, percent):
        self._channel = channel
        self._intensity[channel] = percent

    def _get_state(self):
        return self._channel, self._intensity.get(self._channel, 0.0)

    def _set_channel(self, channel):
        self._channel = channel

    def _get_default_intensity(self, channel):
        return 100.0

    def close(self):
        self.closed = True


@pytest.fixture
def clean_registry():
    """Restore the driver registry after each test that mutates it."""
    saved = list(registry.LIGHTBOX_DRIVERS)
    registry._SHARED = None
    yield
    registry.LIGHTBOX_DRIVERS[:] = saved
    registry._SHARED = None


def test_set_illuminant_resolves_name_case_insensitively():
    box = FakeBox()
    assert box.set_illuminant('d65') == 4
    assert box.get_state().channel == 4
    box.set_illuminant('F12', 50)
    assert box.get_state().channel == 1
    assert box.get_state().intensity == 50


def test_set_illuminant_accepts_names_and_channels():
    box = FakeBox()
    assert box.set_illuminant('d65', 40) == 4
    assert box.get_state().intensity == 40
    assert box.set_illuminant(1, 60) == 1
    assert box.set_illuminant('D65') == 4
    assert box.get_default_intensity('f12') == 100.0
    assert box.set_illuminant('4') == 4  # numeric strings are channel numbers


def test_resolve_folds_labels_and_aliases():
    class LabelledBox(FakeBox):
        @property
        def illuminant_labels(self):
            return {1: 'F12 (fluorescent)', 4: 'D65'}

        @property
        def illuminant_aliases(self):
            return {4: {'daylight'}}

    box = LabelledBox()
    assert box.set_illuminant('f12 (Fluorescent)') == 1  # label, case/punctuation folded
    assert box.set_illuminant('Daylight') == 4  # driver alias


def test_set_illuminant_unknown_name_raises():
    with pytest.raises(LightboxError, match='unknown illuminant'):
        FakeBox().set_illuminant('nope')


def test_off_zeroes_active_channel():
    box = FakeBox()
    box.set_illuminant('D65', 90)
    box.off()
    assert box.get_state().intensity == 0


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
    register_lightbox_driver(FakeBox)
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

        def _set_intensity(self, channel, percent): ...
        def _get_state(self):
            return 0, 0.0

        def _set_channel(self, channel): ...
        def _get_default_intensity(self, channel):
            return 0.0

        def close(self): ...

    registry.LIGHTBOX_DRIVERS[:] = [Absent]
    with pytest.raises(LightboxError, match='no supported lightbox'):
        get_lightbox()

    register_lightbox_driver(FakeBox)
    assert isinstance(get_lightbox(), FakeBox)
