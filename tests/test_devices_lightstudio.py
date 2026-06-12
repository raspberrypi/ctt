# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the lightSTUDIO-S driver (ctt.devices.lightstudio_s). Hardware-free:
# they bypass __init__ and drive a fake ctrl_transfer recorder, so they exercise the
# decoded protocol without pyusb or a device.

import pytest

from ctt.devices import LightboxError
from ctt.devices.lightstudio_s import CHANNEL_LABELS, CHANNEL_NAMES, CHANNEL_TEMPS, Illuminant, LightStudioS


def _bare():
    """A LightStudioS instance without USB init (for pure-logic tests)."""
    box = LightStudioS.__new__(LightStudioS)
    box._channel = 0
    return box


def test_clamp_percent():
    assert LightStudioS._clamp_percent(-5) == 0.0
    assert LightStudioS._clamp_percent(150) == 100.0
    assert LightStudioS._clamp_percent(42.5) == 42.5


def test_known_device_id():
    assert LightStudioS.VENDOR_ID == 0x20A0
    assert LightStudioS.PRODUCT_ID == 0x412D


def test_decode_permille():
    # 2-byte little-endian per-mille → percent. 0x03E8 = 1000 = 100 % (the value the
    # real device returned for channel 1's default intensity in the capture).
    assert LightStudioS._decode_permille(b'\xe8\x03') == 100.0
    assert LightStudioS._decode_permille(b'\x00\x00') == 0.0
    assert LightStudioS._decode_permille(bytes([0xF4, 0x01])) == 50.0  # 500


def test_set_intensity_encoding():
    # Reproduce the captured setIntensity transfers: channel 1 @ {0,50,100}% must map
    # to wValue 0x0103 and wIndex {0,500,1000} (frames 89/93/97 of the capture).
    calls = []
    box = _bare()
    box._dev = type('Dev', (), {'ctrl_transfer': lambda self, *a: calls.append(a)})()

    for pct, expect_index in ((0, 0), (50, 500), (100, 1000)):
        calls.clear()
        box._set_intensity_locked(1, pct)
        bm, req, w_value, w_index, data, _timeout = calls[0]
        assert (bm, req) == (0x40, 0x02)
        assert w_value == 0x0103
        assert w_index == expect_index
        assert data is None
    assert box._channel == 1


def test_validate_channel():
    box = _bare()
    box._validate_channel(1)
    box._validate_channel(8)
    with pytest.raises(LightboxError, match='invalid channel'):
        box._validate_channel(0)
    with pytest.raises(LightboxError, match='invalid channel'):
        box._validate_channel(9)


def test_illuminants_property():
    assert _bare().illuminants is CHANNEL_NAMES
    assert CHANNEL_NAMES[1] == 'F12'
    assert CHANNEL_NAMES[4] == 'D65'
    assert len(CHANNEL_NAMES) == 8


def test_illuminant_temps_property():
    # Nominal CIE CCTs are exposed; the blue-filter channel (8) is intentionally
    # absent so consumers fall back (the UI seeds 6500 K).
    box = _bare()
    assert box.illuminant_temps is CHANNEL_TEMPS
    assert CHANNEL_TEMPS[4] == 6500  # D65
    assert CHANNEL_TEMPS[3] == 5000  # D50
    assert CHANNEL_TEMPS[5] == CHANNEL_TEMPS[6] == CHANNEL_TEMPS[7] == 2856  # Illuminant A
    assert 8 not in CHANNEL_TEMPS


def test_resolve_illuminant_on_real_map():
    # The generic name→channel resolver, exercised against the driver's illuminants.
    box = _bare()
    assert box._resolve('D65') == 4
    assert box._resolve('f12') == 1
    assert box._resolve(Illuminant.HalogenBF) == 8
    assert box._resolve('halogenbf') == 8
    assert box._resolve('Halogen (10 lux)') == 5  # descriptive labels resolve too
    assert box._resolve('halogen 100') == 6
    assert box._resolve(7) == 7


def test_labels_cover_all_channels():
    assert set(CHANNEL_LABELS) == set(CHANNEL_NAMES)
    assert CHANNEL_LABELS[8] == 'Halogen + blue filter (400 lux)'


def test_unconfigured_id_raises():
    pytest.importorskip('usb')  # constructing a real device needs pyusb
    with pytest.raises(LightboxError, match='USB id is not configured'):
        LightStudioS(vendor_id=0, product_id=0)
