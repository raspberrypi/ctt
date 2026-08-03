# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the sharpen-threshold sweep analysis.
#
# The noise metric is a Gaussian high-pass residual std, so synthetic flat
# patches with known injected noise give it a ground truth without a camera:
# white noise passes the high-pass almost untouched while low-order shading
# is removed entirely.

import json

import numpy as np

from ctt_server.sharpen import (
    DEFAULT_TOLERANCE,
    GREY_SLICE,
    grey_patch_noise,
    patch_luma,
    patch_spatial_noise,
    patch_window_half,
    recommend_threshold,
    set_sharpen,
)


def noisy_patch(rng, sigma, size=64, level=128.0):
    return level + rng.normal(0.0, sigma, (size, size))


class TestPatchSpatialNoise:
    def test_scales_with_injected_noise(self):
        rng = np.random.default_rng(42)
        one = patch_spatial_noise(noisy_patch(rng, 2.0))
        two = patch_spatial_noise(noisy_patch(rng, 4.0))
        np.testing.assert_allclose(two / one, 2.0, rtol=0.1)

    def test_immune_to_illumination_shading(self):
        rng = np.random.default_rng(7)
        noise = rng.normal(0.0, 3.0, (64, 64))
        flat = 128.0 + noise
        yy, xx = np.mgrid[0:64, 0:64] / 64.0
        shaded = flat + 30.0 * xx + 20.0 * yy + 25.0 * (xx - 0.5) ** 2
        ratio = patch_spatial_noise(shaded) / patch_spatial_noise(flat)
        assert abs(ratio - 1.0) < 0.05

    def test_constant_patch_is_zero(self):
        assert patch_spatial_noise(np.full((64, 64), 100.0)) < 1e-9

    def test_tiny_patch_does_not_crash(self):
        assert patch_spatial_noise(np.full((6, 6), 100.0)) >= 0.0


class TestPatchLuma:
    def test_bgr_channel_order(self):
        # A pure-red patch in BGR order must weight by the red coefficient.
        patch = np.zeros((4, 4, 3))
        patch[..., 2] = 255.0
        np.testing.assert_allclose(patch_luma(patch), 0.299 * 255.0)

    def test_white_patch_full_luma(self):
        np.testing.assert_allclose(patch_luma(np.full((4, 4, 3), 255.0)), 255.0)


def chart_centres(origin=(60, 60), pitch=50):
    """24 patch centres in detector order on a 6x4 grid."""
    x0, y0 = origin
    return np.array([[x0 + col * pitch, y0 + row * pitch] for col in range(6) for row in range(4)], dtype=float)


class TestGreyPatchNoise:
    def test_measures_the_grey_row_only(self):
        rng = np.random.default_rng(3)
        centres = chart_centres()
        frame = np.full((300, 400, 3), 100.0)
        # Inject noise ONLY into the grey patches; the rest of the chart stays
        # clean, so any leakage from non-grey patches would read as ~0.
        half = patch_window_half(centres)
        for x, y in centres[GREY_SLICE].astype(int):
            frame[y - half : y + half + 1, x - half : x + half + 1] += rng.normal(
                0.0, 4.0, (2 * half + 1, 2 * half + 1, 3)
            )
        noise, means = grey_patch_noise([frame], centres)
        assert len(noise) == len(means) == 6
        assert all(n > 2.0 for n in noise)  # the injected noise is seen...
        np.testing.assert_allclose(means, 100.0, atol=3.0)  # ...around the flat level

    def test_averages_over_frames(self):
        rng = np.random.default_rng(11)
        centres = chart_centres()
        frames = [np.asarray(100.0 + rng.normal(0.0, 3.0, (300, 400, 3))) for _ in range(3)]
        noise_multi, _ = grey_patch_noise(frames, centres)
        noise_single, _ = grey_patch_noise(frames[:1], centres)
        assert len(noise_multi) == 6
        # Averaging the per-frame metric tightens it but must not change scale.
        np.testing.assert_allclose(noise_multi, noise_single, rtol=0.25)


class TestRecommendThreshold:
    def points(self, aggregates):
        return [{'threshold': t, 'aggregate': a} for t, a in aggregates]

    def test_picks_smallest_within_tolerance(self):
        points = self.points([(0.1, 1.4), (0.2, 1.04), (0.4, 1.01), (0.75, 1.0)])
        assert recommend_threshold(points, DEFAULT_TOLERANCE)['threshold'] == 0.2

    def test_handles_unsorted_input(self):
        points = self.points([(0.75, 1.0), (0.1, 1.4), (0.4, 1.01), (0.2, 1.04)])
        assert recommend_threshold(points, DEFAULT_TOLERANCE)['threshold'] == 0.2

    def test_exactly_at_tolerance_accepted(self):
        points = self.points([(0.1, 1.2), (0.4, 1.05)])
        assert recommend_threshold(points, 1.05)['threshold'] == 0.4

    def test_none_within_tolerance(self):
        out = recommend_threshold(self.points([(0.1, 1.5), (0.4, 1.2)]), 1.05)
        assert out['threshold'] is None
        assert 'tolerance' in out['reason']


def v2_tuning(sharpen=None):
    algorithms = [{'rpi.black_level': {'black_level': 4096}}]
    if sharpen is not None:
        algorithms.append({'rpi.sharpen': dict(sharpen)})
    algorithms.append({'rpi.ccm': {'ccms': []}})
    return {'version': 2.0, 'target': 'pisp', 'algorithms': algorithms}


class TestSetSharpen:
    def test_updates_v2_tuning(self):
        tuning = v2_tuning({'threshold': 0.75, 'limit': 0.5, 'strength': 1.0})
        set_sharpen(tuning, threshold=0.3)
        block = next(e for e in tuning['algorithms'] if 'rpi.sharpen' in e)['rpi.sharpen']
        assert block == {'threshold': 0.3, 'limit': 0.5, 'strength': 1.0}
        # Other algorithm blocks are untouched.
        assert tuning['algorithms'][0] == {'rpi.black_level': {'black_level': 4096}}

    def test_updates_legacy_tuning(self):
        tuning = {'rpi.sharpen': {'threshold': 0.75, 'limit': 0.5, 'strength': 1.0}, 'rpi.ccm': {}}
        set_sharpen(tuning, threshold=1.5)
        assert tuning['rpi.sharpen']['threshold'] == 1.5
        assert tuning['rpi.sharpen']['limit'] == 0.5

    def test_inserts_missing_block(self):
        tuning = v2_tuning()
        set_sharpen(tuning, threshold=0.2)
        block = next(e for e in tuning['algorithms'] if 'rpi.sharpen' in e)['rpi.sharpen']
        assert block == {'threshold': 0.2, 'limit': 0.5, 'strength': 1.0}

    def test_strength_zero_baseline(self):
        tuning = v2_tuning({'threshold': 0.75, 'limit': 0.5, 'strength': 1.0})
        set_sharpen(tuning, strength=0.0)
        block = next(e for e in tuning['algorithms'] if 'rpi.sharpen' in e)['rpi.sharpen']
        assert block['strength'] == 0.0
        assert block['threshold'] == 0.75  # untouched

    def test_survives_json_round_trip(self):
        tuning = set_sharpen(v2_tuning({'threshold': 0.75}), threshold=0.4)
        assert json.loads(json.dumps(tuning))['algorithms'][1]['rpi.sharpen']['threshold'] == 0.4


# --- endpoints ---


def _client(tmp_path):
    from ctt_server import sessions
    from ctt_server.app import create_app

    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('cam')
    return create_app(str(tmp_path)).test_client(), proj


def test_sharpen_page_renders(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get('/projects/cam/sharpen')
    assert r.status_code == 200
    assert b'sharpenApp' in r.data
    assert client.get('/projects/nope/sharpen').status_code == 404


def test_sharpen_data_empty(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get('/projects/cam/sharpen/data')
    assert r.status_code == 200
    data = r.get_json()
    assert data['results'] is None
    assert data['running'] is False
    assert data['can_apply'] is False


def test_sharpen_data_returns_stored_results(tmp_path):
    import ctt_server.sharpen as sharpen_mod

    client, proj = _client(tmp_path)
    stored = {'version': 1, 'points': [], 'recommended': {'threshold': 0.2}}
    path = sharpen_mod.results_path(proj)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(stored))
    data = client.get('/projects/cam/sharpen/data').get_json()
    assert data['results']['recommended']['threshold'] == 0.2


def test_sharpen_apply_requires_generated_tuning(tmp_path, monkeypatch):
    import ctt_server.app as app_mod

    client, _ = _client(tmp_path)
    monkeypatch.setattr(app_mod, 'platform_target', lambda: 'pisp')
    r = client.post('/projects/cam/sharpen/apply', json={'threshold': 0.2})
    assert r.status_code == 400
    assert 'run CTT' in r.get_json()['error']


def test_sharpen_apply_validates_threshold(tmp_path, monkeypatch):
    import ctt_server.app as app_mod

    client, _ = _client(tmp_path)
    monkeypatch.setattr(app_mod, 'platform_target', lambda: 'pisp')
    assert client.post('/projects/cam/sharpen/apply', json={}).status_code == 400
    assert client.post('/projects/cam/sharpen/apply', json={'threshold': 'x'}).status_code == 400
    assert client.post('/projects/cam/sharpen/apply', json={'threshold': -1}).status_code == 400
    assert client.post('/projects/cam/sharpen/apply', json={'threshold': 99}).status_code == 400


def test_sharpen_apply_writes_tuning_and_records(tmp_path, monkeypatch):
    import ctt_server.app as app_mod
    import ctt_server.sharpen as sharpen_mod

    client, proj = _client(tmp_path)
    monkeypatch.setattr(app_mod, 'platform_target', lambda: 'pisp')
    proj.output_dir.mkdir(parents=True)
    tuning_path = proj.output_dir / 'cam_pisp.json'
    tuning_path.write_text(json.dumps(v2_tuning({'threshold': 0.75, 'limit': 0.5, 'strength': 1.0})))
    results_file = sharpen_mod.results_path(proj)
    results_file.parent.mkdir(parents=True)
    results_file.write_text(json.dumps({'version': 1, 'applied': None}))

    r = client.post('/projects/cam/sharpen/apply', json={'threshold': 0.2})
    assert r.status_code == 200
    assert r.get_json()['file'] == 'cam_pisp.json'
    written = json.loads(tuning_path.read_text())
    block = next(e for e in written['algorithms'] if 'rpi.sharpen' in e)['rpi.sharpen']
    assert block['threshold'] == 0.2
    assert block['limit'] == 0.5  # untouched
    assert json.loads(results_file.read_text())['applied']['threshold'] == 0.2


def test_sharpen_stream_refuses_while_running(tmp_path, monkeypatch):
    import ctt_server.app as app_mod
    import ctt_server.sharpen as sharpen_mod

    client, _ = _client(tmp_path)
    monkeypatch.setattr(app_mod, 'get_shared_camera', lambda: object())
    assert sharpen_mod._sharpen_lock.acquire(blocking=False)
    try:
        r = client.get('/projects/cam/sharpen/sweep/stream')
        assert b'already running' in r.data
    finally:
        sharpen_mod._sharpen_lock.release()


def test_sharpen_stream_invalid_settings_streams_error(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get('/projects/cam/sharpen/sweep/stream?gain=notanumber')
    assert b'invalid sweep settings' in r.data
