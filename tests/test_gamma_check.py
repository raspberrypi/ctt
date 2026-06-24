# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the gamma / tone-curve verification (diagnostic only).

import colour.models
import numpy as np

from ctt.algorithms.gamma_check import _NEUTRAL_SRGB, GammaCheck, target_oetf
from ctt.core.camera import Camera
from ctt.core.image import Image

# Scene reflectance of the six neutral patches, white 9.5 first.
REFL = colour.models.eotf_sRGB(_NEUTRAL_SRGB / 255)


def _srgb_gamma_curve(n=64):
    """A gamma_curve (flat x0,y0,x1,y1,... in 0-65535) that follows the sRGB OETF."""
    xs = np.linspace(0, 65535, n)
    ys = colour.models.eotf_inverse_sRGB(xs / 65535) * 65535
    return [int(round(v)) for pair in zip(xs, ys, strict=True) for v in pair]


def _macbeth_image(signal, bl=0, name='macbeth_5000k_0.dng'):
    """A synthetic Macbeth capture whose neutral greys (indices 3::4, white first)
    carry `signal` on every channel; all other patches are zero."""
    img = Image()
    img.name = name
    img.order = (0, 1, 2, 3)
    img.blacklevel_16 = bl
    channel = [np.zeros(16) for _ in range(24)]
    for k, gi in enumerate(range(3, 24, 4)):
        channel[gi] = np.full(16, signal[k] + bl)
    img.patches = [list(channel) for _ in range(4)]
    return img


def _camera(imgs, gamma_curve):
    cam = Camera('out.json', json={'rpi.contrast': {'gamma_curve': gamma_curve}})
    cam.imgs = imgs
    return cam


class TestTargetOetf:
    def test_unknown_name_defaults_to_srgb(self):
        assert np.isclose(target_oetf('nonsense')(np.array([1.0]))[0], 1.0)

    def test_power_law(self):
        assert np.isclose(target_oetf('power:2.0')(np.array([0.25]))[0], 0.5)


class TestGammaCheck:
    def test_linear_sensor_srgb_curve_is_near_perfect(self):
        cam = _camera([_macbeth_image(REFL * 60000)], _srgb_gamma_curve())
        assert GammaCheck(cam, None, 'sRGB').run() is None  # never writes tuning
        assert cam.json['rpi.contrast']['gamma_curve'] == _srgb_gamma_curve()  # curve untouched
        g = cam.metrics['gamma']
        assert g['target'] == 'sRGB'
        assert g['linearity']['r2'] > 0.9999
        assert g['curve']['rms_dev_8bit'] < 0.5
        assert g['tone']['rms_err_8bit'] < 1.0
        assert len(g['tone']['patches']) == 6
        # White patch is the brightest neutral, normalised to a reflectance of 1.
        assert g['tone']['patches'][0]['reflectance'] == 1.0
        assert all(0 <= p['reflectance'] <= 1 for p in g['tone']['patches'])

    def test_black_level_is_subtracted(self):
        # Same signal sitting on a pedestal must give the same result once subtracted.
        cam = _camera([_macbeth_image(REFL * 60000, bl=1000)], _srgb_gamma_curve())
        GammaCheck(cam, None, 'sRGB').run()
        assert cam.metrics['gamma']['linearity']['r2'] > 0.9999

    def test_curve_check_runs_without_images(self):
        cam = _camera([], _srgb_gamma_curve())
        GammaCheck(cam, None, 'sRGB').run()
        g = cam.metrics['gamma']
        assert 'curve' in g
        assert 'tone' not in g and 'linearity' not in g

    def test_missing_curve_skips_curve_check(self):
        cam = Camera('out.json', json={'rpi.contrast': {}})
        cam.imgs = [_macbeth_image(REFL * 60000)]
        GammaCheck(cam, None, 'sRGB').run()
        g = cam.metrics['gamma']
        assert 'curve' not in g
        assert 'linearity' in g  # linearity still runs
        assert 'tone' not in g  # but end-to-end tone needs the curve

    def test_off_target_flags_large_deviation(self):
        # The sRGB curve measured against a linear (power:1.0) target deviates a lot.
        cam = _camera([], _srgb_gamma_curve())
        GammaCheck(cam, None, 'power:1.0').run()
        assert cam.metrics['gamma']['curve']['max_dev_8bit'] > 40

    def test_nonlinear_sensor_lowers_r2(self):
        cam = _camera([_macbeth_image((REFL**1.5) * 60000)], _srgb_gamma_curve())
        GammaCheck(cam, None, 'sRGB').run()
        assert cam.metrics['gamma']['linearity']['r2'] < 0.999
