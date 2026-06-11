# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the black level calibration (dark frame measurement).

import types

import numpy as np

from ctt.algorithms.black_level import BlackLevelCalibration, measure_dark_image
from ctt.core.camera import Camera
from ctt.core.image import Image


def _dark_image(values=(4000, 4100, 4200, 4300), order=(0, 1, 2, 3), pattern=0, name='dark_0.dng'):
    """A synthetic dark frame: constant per-channel values, 16-bit domain."""
    img = Image()
    img.name = name
    img.sigbits = 12
    img.pattern = pattern
    img.order = order
    img.channels = [np.full((50, 50), v, dtype=np.int64) for v in values]
    img.blacklevel = 256  # raw DNG metadata value
    img.blacklevel_16 = 256 << 4  # = 4096, close to the synthetic means
    img.exposure = 10000
    img.againQ8_norm = 2.0
    return img


def _camera(dark_imgs):
    cam = Camera('out.json', json={'rpi.black_level': {'black_level': 4096}})
    cam.imgs_dark = dark_imgs
    return cam


class TestMeasureDarkImage:
    def test_channel_means_identity_order(self):
        m = measure_dark_image(_dark_image())
        # order (0,1,2,3): r=ch0, gr=ch1, gb=ch2, b=ch3
        assert (m['r'], m['g'], m['b']) == (4000.0, 4150.0, 4300.0)
        assert m['black_level'] == 4150.0
        assert m['exposure'] == 10000
        assert m['gain'] == 2.0
        assert m['total_exposure'] == 20000

    def test_order_reorders_channels(self):
        # GBRG layout: spatial channels are (G, B, R, G); order maps to (R, Gr, Gb, B).
        m = measure_dark_image(_dark_image(values=(4100, 4300, 4000, 4200), order=(2, 0, 3, 1)))
        assert (m['r'], m['g'], m['b']) == (4000.0, 4150.0, 4300.0)

    def test_mono_reports_single_channel(self):
        m = measure_dark_image(_dark_image(values=(4000,) * 4, pattern=128))
        assert m['y'] == 4000.0
        assert 'r' not in m


class TestBlackLevelCalibration:
    def test_no_dark_frames_returns_none(self):
        cam = _camera([])
        assert BlackLevelCalibration(cam, None).run() is None
        assert 'black_level' not in cam.metrics

    def test_measures_and_propagates(self):
        cam = _camera([_dark_image()])
        other = types.SimpleNamespace(blacklevel_16=4096)
        cam.imgs = [other]
        result = BlackLevelCalibration(cam, None).run()
        assert result == {'black_level': 4150}
        assert cam.blacklevel_16 == 4150
        assert other.blacklevel_16 == 4150
        bl = cam.metrics['black_level']
        assert bl['source'] == 'dark'
        assert bl['black_level'] == 4150
        assert bl['frames'][0]['r'] == 4000.0
        assert not cam.metrics['warnings']

    def test_averages_multiple_frames(self):
        cam = _camera([_dark_image(values=(4100,) * 4), _dark_image(values=(4200,) * 4, name='dark_1.dng')])
        assert BlackLevelCalibration(cam, None).run() == {'black_level': 4150}
        assert len(cam.metrics['black_level']['frames']) == 2

    def test_config_override_wins(self):
        cam = _camera([_dark_image()])
        cam.blacklevel_16 = 3200
        result = BlackLevelCalibration(cam, None, blacklevel_override=3200).run()
        assert result is None
        assert cam.blacklevel_16 == 3200  # untouched
        bl = cam.metrics['black_level']
        assert bl['source'] == 'override'
        assert bl['black_level'] == 3200
        assert bl['measured'] == 4150  # still recorded for analysis

    def test_channel_spread_warns(self):
        # r/b 800 below the greens: spread 400 from the mean, past the
        # 0.5%-of-full-scale (≈328) limit.
        cam = _camera([_dark_image(values=(4000, 4800, 4800, 4000))])
        BlackLevelCalibration(cam, None).run()
        assert any('light leak' in w['message'] for w in cam.metrics['warnings'])

    def test_metadata_disagreement_warns(self):
        img = _dark_image()
        img.blacklevel = 64  # metadata says 64 << 4 = 1024, measurement ~4150
        cam = _camera([img])
        BlackLevelCalibration(cam, None).run()
        assert any('metadata' in w['message'] for w in cam.metrics['warnings'])


class TestCameraDarkClassification:
    def test_dark_file_lands_in_imgs_dark(self, tmp_path, monkeypatch):
        import ctt.core.camera as camera_mod

        for n in ('dark_0.dng', 'alsc_5000k_0.dng'):
            (tmp_path / n).write_bytes(b'DNG')
        monkeypatch.setattr(
            camera_mod,
            'load_image',
            lambda cam, address, mac_config=None, mac=True, demosaic=True: types.SimpleNamespace(),
        )
        cam = camera_mod.Camera('out.json', json={})
        cam.add_imgs(str(tmp_path) + '/', (0, 0))
        assert [i.name for i in cam.imgs_dark] == ['dark_0.dng']
        assert cam.imgs == []  # the dark frame must not be treated as Macbeth
        assert len(cam.imgs_alsc) == 1
