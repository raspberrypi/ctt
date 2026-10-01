# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# The Preview-tab PNG is a snapshot of the live preview at the selected sensor mode's
# resolution. The live main stream is only preview-sized, so it comes from a brief
# still-mode switch; the still must be held to the preview frame's exposure, gain and
# colour gains, and the user's controls re-applied afterwards (reconfiguring resets
# them). A fake Picamera2 lets this run without hardware.

import threading

import numpy as np
import pytest

from ctt_server.camera import Picamera2Camera

PREVIEW_MD = {'ExposureTime': 12345, 'AnalogueGain': 2.5, 'ColourGains': (1.8, 1.6)}


class FakePicam2:
    def __init__(self):
        self.still_configs = []
        self.switch_calls = []
        self.set_calls = []

    def capture_metadata(self):
        return dict(PREVIEW_MD)

    def create_video_configuration(self, **kwargs):
        return {'controls': {'NoiseReductionMode': 'Fast', **kwargs.get('controls', {})}}

    def create_still_configuration(self, **kwargs):
        self.still_configs.append(kwargs)
        return kwargs

    def switch_mode_and_capture_array(self, cfg, stream):
        self.switch_calls.append(stream)
        w, h = cfg['main']['size']
        return np.zeros((h, w, 3), dtype=np.uint8)

    def set_controls(self, controls):
        self.set_calls.append(dict(controls))


def _camera(fake, monkeypatch):
    """A Picamera2Camera bound to a fake picam2, bypassing the hardware __init__."""
    cam = object.__new__(Picamera2Camera)
    cam._picam2 = fake
    cam._lock = threading.Lock()
    cam._raw_size = cam.resolution = (64, 48)
    cam._raw_format = None
    cam._raw_bit_depth = 10
    cam._preview_size = (32, 24)
    cam._fps = 30.0
    cam._ev = 0.5
    cam._auto = True
    cam._manual_exposure = {}
    cam._awb_controls = {}
    monkeypatch.setattr(Picamera2Camera, '_transform', lambda self: None)
    monkeypatch.setattr(Picamera2Camera, '_frame_duration_limits', lambda self: (33333, 33333))
    return cam


def _png_size(png: bytes) -> tuple[int, int]:
    assert png[:8] == b'\x89PNG\r\n\x1a\n'  # a real PNG
    return int.from_bytes(png[16:20], 'big'), int.from_bytes(png[20:24], 'big')


def test_snapshot_is_at_the_selected_mode_resolution(monkeypatch):
    fake = FakePicam2()
    assert _png_size(_camera(fake, monkeypatch).capture_png()) == (64, 48)
    assert fake.still_configs[0]['main']['size'] == (64, 48)


def test_snapshot_still_matches_the_preview_frame(monkeypatch):
    fake = FakePicam2()
    _camera(fake, monkeypatch).capture_png()
    controls = fake.still_configs[0]['controls']
    assert controls['AeEnable'] is False
    assert controls['ExposureTime'] == 12345
    assert controls['AnalogueGain'] == 2.5
    assert controls['AwbEnable'] is False
    assert controls['ColourGains'] == (1.8, 1.6)
    assert controls['NoiseReductionMode'] == 'Fast'  # the preview's ISP controls, not the still defaults


@pytest.mark.parametrize(
    'auto, manual, awb, expected',
    [
        (True, {}, {}, {'AeEnable': True, 'AwbEnable': True, 'ExposureValue': 0.5}),
        (
            False,
            {'ExposureTime': 20000, 'AnalogueGain': 4.0},
            {'AwbEnable': False, 'ColourGains': (2.0, 1.5)},
            {
                'AeEnable': False,
                'AwbEnable': False,
                'ExposureValue': 0.5,
                'ExposureTime': 20000,
                'AnalogueGain': 4.0,
                'ColourGains': (2.0, 1.5),
            },
        ),
    ],
)
def test_user_controls_restored_after_the_snapshot(monkeypatch, auto, manual, awb, expected):
    fake = FakePicam2()
    cam = _camera(fake, monkeypatch)
    cam._auto, cam._manual_exposure, cam._awb_controls = auto, manual, awb
    cam.capture_png()
    assert fake.set_calls == [expected]
