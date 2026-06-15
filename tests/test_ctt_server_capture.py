# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# The Preview-tab snapshot is a zero-shutter-lag grab of the live preview frame:
# it reads the running main stream with NO mode switch, so the manual exposure/gain
# and the ISP denoise state survive (capture_png's still-mode switch reverts auto-
# exposure and steps the brightness). A fake Picamera2 lets this run without hardware.

import threading

import numpy as np

from ctt_server.camera import Picamera2Camera


class FakePicam2:
    def __init__(self):
        self.switch_calls = []
        self.array_calls = []

    def capture_array(self, stream):
        self.array_calls.append(stream)
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def switch_mode(self, cfg):
        self.switch_calls.append(cfg)

    def switch_mode_and_capture_array(self, cfg, stream):
        self.switch_calls.append(cfg)
        return np.zeros((4, 4, 3), dtype=np.uint8)


def _camera(fake):
    """A Picamera2Camera bound to a fake picam2, bypassing the hardware __init__."""
    cam = object.__new__(Picamera2Camera)
    cam._picam2 = fake
    cam._lock = threading.Lock()
    return cam


def test_preview_snapshot_grabs_running_stream_without_switching():
    fake = FakePicam2()
    png = _camera(fake).capture_preview_png()
    assert png[:8] == b'\x89PNG\r\n\x1a\n'  # a real PNG
    assert fake.array_calls == ['main']  # grabbed the live preview frame
    assert fake.switch_calls == []  # zero shutter lag: no mode switch
