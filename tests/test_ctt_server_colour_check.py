# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the live colour-accuracy check (measured Macbeth patches vs reference).

import numpy as np

from ctt_server.colour_check import _reference, deltae_report, patch_means


def test_perfect_patches_score_zero():
    # Feeding the reference colours straight back must give ~0 delta E.
    _, m_rgb = _reference()
    out = deltae_report(m_rgb.astype(float))
    assert out['mean'] < 0.1
    assert out['colour'] < 0.1
    assert abs(out['scale'] - 1.0) < 0.02
    assert len(out['patches']) == 24
    assert not out['saturated']


def test_brightness_offset_removed_by_colour_metric():
    # Dim every patch by 0.7 in linear light: raw delta E rises (lightness is
    # wrong everywhere) but the brightness-normalised 'colour' stays ~0 and the
    # fitted scale recovers the 1/0.7 factor.
    from ctt.algorithms.ccm import degamma, gamma

    _, m_rgb = _reference()
    linear01 = degamma(m_rgb) / 65535
    dimmed = gamma(0.7 * linear01 * 255)
    out = deltae_report(dimmed)
    assert out['colour'] < 0.2
    assert out['mean'] > 2.0
    assert abs(out['scale'] - 1 / 0.7) < 0.05


def test_saturated_flag():
    _, m_rgb = _reference()
    patches = m_rgb.astype(float)
    patches[0] = [255, 255, 255]
    assert deltae_report(patches)['saturated']


def test_patch_means_samples_window():
    # Frame of solid-colour cells; means at the cell centres must match exactly.
    colours = [(200, 10, 10), (10, 200, 10), (10, 10, 200), (60, 60, 60)]
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    centres = []
    for i, c in enumerate(colours):
        x0 = i * 20
        frame[:, x0 : x0 + 20] = c
        centres.append([x0 + 10, 20])
    means = patch_means(frame, np.array(centres))
    np.testing.assert_allclose(means, colours, atol=0.01)


def test_deltae_endpoint(tmp_path, monkeypatch):
    import ctt_server.app as app_mod
    from ctt_server import sessions
    from ctt_server.app import create_app

    sessions.Workspace(tmp_path).create_project('cam')
    client = create_app(str(tmp_path)).test_client()

    # Build a synthetic 6x4 chart frame: each cell holds the reference colour in
    # detector order. The endpoint converts BGR->RGB, so store BGR here.
    _, m_rgb = _reference()
    cell = 20
    frame = np.zeros((4 * cell, 6 * cell, 3), dtype=np.uint8)
    centres = []
    for i, rgb in enumerate(m_rgb):
        r, c = divmod(i, 6)
        frame[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = rgb[::-1]  # BGR
        centres.append([c * cell + cell // 2, r * cell + cell // 2])

    class FakeCam:
        def chart_patches(self):
            return {'found': True, 'confidence': 0.9, 'centres': np.array(centres), 'frame': frame}

    monkeypatch.setattr(app_mod, 'get_shared_camera', lambda: FakeCam())
    r = client.get('/api/macbeth-deltae')
    assert r.status_code == 200
    d = r.get_json()
    assert d['found'] is True
    assert d['confidence'] == 0.9
    assert d['mean'] < 1.0  # uint8 quantisation only
    assert len(d['patches']) == 24

    # Chart not found degrades gracefully.
    monkeypatch.setattr(FakeCam, 'chart_patches', lambda self: {'found': False})
    assert client.get('/api/macbeth-deltae').get_json() == {'found': False}


def test_module_reference_matches_ccm_ordering():
    # The detector-order reorder must match ccm.py's ([i::6] reshape); spot-check
    # that patch 0 in detector order is the canonical patch 0 ('dark skin').
    from ctt.algorithms.ccm import MACBETH_RGB

    _, m_rgb = _reference()
    assert list(m_rgb[0]) == MACBETH_RGB[0]
