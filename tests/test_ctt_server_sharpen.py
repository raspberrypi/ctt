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
    DEFAULT_OVERSHOOT_CAP,
    DEFAULT_PEAK_CAP,
    DEFAULT_TOLERANCE,
    GREY_SLICE,
    REFERENCE_LIMIT,
    classify_edges,
    grey_patch_noise,
    patch_luma,
    patch_spatial_noise,
    patch_window_half,
    recommend_limit,
    recommend_strength,
    recommend_threshold,
    select_texture_tiles,
    set_sharpen,
    strength_point_tunings,
    texture_gain,
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


class TestSelectTextureTiles:
    def scene(self):
        """Flat noise (1 LSB) with one faint-texture tile, one strong-texture
        tile and one high-contrast edge tile, all grid-aligned."""
        rng = np.random.default_rng(9)
        luma = 128.0 + rng.normal(0.0, 1.0, (256, 320))
        luma[64:128, 64:128] += rng.normal(0.0, 4.0, (64, 64))  # faint texture: in band
        luma[192:256, 0:64] += rng.normal(0.0, 25.0, (64, 64))  # strong texture: above band
        luma[0:64, 128:160] = 60.0  # edge tile: a hard step...
        luma[0:64, 160:192] = 190.0  # ...excluded by its low-frequency swing
        return luma

    def test_selects_faint_texture_only(self):
        tiles = select_texture_tiles(self.scene(), noise_floor=1.0)
        assert [(t['x'], t['y']) for t in tiles] == [(64, 64)]
        assert 2.0 <= tiles[0]['energy'] <= 10.0

    def test_noise_floor_fallback_from_flat_tiles(self):
        # Without a grey-patch noise floor the flattest tiles stand in for it.
        tiles = select_texture_tiles(self.scene())
        assert [(t['x'], t['y']) for t in tiles] == [(64, 64)]

    def test_max_tiles_prefers_strongest(self):
        rng = np.random.default_rng(3)
        luma = 128.0 + rng.normal(0.0, 0.5, (128, 320))
        for i, sigma in enumerate((3.0, 5.0, 4.0)):
            luma[:64, i * 64 : (i + 1) * 64] += rng.normal(0.0, sigma, (64, 64))
        tiles = select_texture_tiles(luma, noise_floor=0.5, max_tiles=2)
        # The two strongest in-band tiles win: best signal-to-noise ratios.
        assert [(t['x'], t['y']) for t in tiles] == [(64, 0), (128, 0)]

    def test_flat_frame_returns_empty(self):
        assert select_texture_tiles(np.full((256, 256), 128.0), noise_floor=1.0) == []

    def test_frame_smaller_than_a_tile_returns_empty(self):
        assert select_texture_tiles(np.full((32, 32), 128.0), noise_floor=1.0) == []


class TestTextureGain:
    def frames(self):
        """A baseline with one texture tile, and the same scene with the
        texture amplified 1.5x (what stronger sharpening does to it)."""
        rng = np.random.default_rng(4)
        base = 128.0 + rng.normal(0.0, 0.2, (128, 128))
        tex = rng.normal(0.0, 4.0, (64, 64))
        baseline = base.copy()
        baseline[:64, :64] += tex
        sharpened = base.copy()
        sharpened[:64, :64] += 1.5 * tex
        return baseline, sharpened

    def test_gain_tracks_texture_amplification(self):
        baseline, sharpened = self.frames()
        tiles = select_texture_tiles(baseline, noise_floor=0.2)
        assert [(t['x'], t['y']) for t in tiles] == [(0, 0)]
        assert abs(texture_gain(baseline, tiles) - 1.0) < 0.05
        gain = texture_gain(sharpened, tiles)
        assert 1.4 < gain < 1.6

    def test_no_tiles_gives_none(self):
        assert texture_gain(np.full((128, 128), 128.0), []) is None


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


class TestRecommendStrength:
    def points(self, rows):
        return [
            {'strength': s, 'overshoot': o, 'undershoot': u, 'mtf_peak': p, 'acutance_gain': 1.0, 'mtf50_boost': 1.0}
            for s, o, u, p in rows
        ]

    def test_picks_largest_within_both_caps(self):
        points = self.points([(0.5, 0.02, 0.03, 1.05), (1.0, 0.05, 0.08, 1.15), (2.0, 0.2, 0.3, 1.5)])
        assert recommend_strength(points, 0.10, 1.25)['strength'] == 1.0
        # The calibrated default cap (5%) is stricter and drops to 0.5.
        assert recommend_strength(points, DEFAULT_OVERSHOOT_CAP, DEFAULT_PEAK_CAP)['strength'] == 0.5

    def test_peak_cap_alone_rejects(self):
        # Halo fine, but the MTF peak betrays over-crisping.
        points = self.points([(0.5, 0.02, 0.02, 1.1), (1.0, 0.04, 0.04, 1.4)])
        assert recommend_strength(points, 0.10, 1.25)['strength'] == 0.5

    def test_worse_halo_side_gates(self):
        # Undershoot (the stronger negative gain) breaches the cap alone.
        points = self.points([(1.0, 0.05, 0.15, 1.1)])
        assert recommend_strength(points, 0.10, 1.25)['strength'] is None

    def test_exactly_at_caps_accepted(self):
        points = self.points([(1.0, 0.10, 0.05, 1.25)])
        assert recommend_strength(points, 0.10, 1.25)['strength'] == 1.0

    def test_unmeasured_points_skipped(self):
        points = self.points([(0.5, 0.02, 0.02, 1.05)]) + [
            {'strength': 1.0, 'overshoot': None, 'undershoot': None, 'mtf_peak': None, 'acutance_gain': None}
        ]
        assert recommend_strength(points, 0.10, 1.25)['strength'] == 0.5

    def test_none_within_caps(self):
        out = recommend_strength(self.points([(0.5, 0.2, 0.2, 1.5)]), 0.10, 1.25)
        assert out['strength'] is None
        assert 'cap' in out['reason']

    def tex_points(self, rows):
        return [
            {
                'strength': s,
                'overshoot': 0.02,
                'undershoot': 0.02,
                'mtf_peak': 1.1,
                'acutance_gain': a,
                'texture_gain': t,
                'mtf50_boost': 1.0,
            }
            for s, a, t in rows
        ]

    def test_texture_floor_prunes_soft_texture_pick(self):
        # The observed failure mode: a weak strength wins on (edge) acutance
        # but renders faint texture visibly softer than a stronger candidate.
        points = self.tex_points([(0.25, 1.3, 1.05), (0.5, 1.2, 1.25), (1.0, 1.1, 1.3)])
        out = recommend_strength(points, 0.10, 1.25)
        assert out['strength'] == 0.5  # 0.25's texture is >5% below the best (1.3)
        assert out['texture_gain'] == 1.25

    def test_texture_within_slack_keeps_acutance_choice(self):
        points = self.tex_points([(0.25, 1.3, 1.28), (0.5, 1.2, 1.29), (1.0, 1.1, 1.3)])
        assert recommend_strength(points, 0.10, 1.25)['strength'] == 0.25

    def test_capped_point_does_not_set_texture_floor(self):
        # A point over the halo/peak caps must not raise the floor for the rest.
        points = self.tex_points([(0.5, 1.2, 1.1), (1.0, 1.15, 1.12)])
        points += [
            {
                'strength': 2.0,
                'overshoot': 0.3,
                'undershoot': 0.3,
                'mtf_peak': 1.5,
                'acutance_gain': 1.4,
                'texture_gain': 2.0,
                'mtf50_boost': 1.2,
            }
        ]
        assert recommend_strength(points, 0.10, 1.25)['strength'] == 0.5

    def test_without_texture_measurements_acutance_rules(self):
        # No texture tiles in the scene: the floor disengages entirely.
        points = self.points([(0.5, 0.02, 0.03, 1.05), (1.0, 0.05, 0.08, 1.15)])
        out = recommend_strength(points, 0.10, 1.25)
        assert out['strength'] == 1.0
        assert out['texture_gain'] is None

    def test_non_monotonic_gain_picks_highest(self):
        # The threshold gate and limit clipping can make sharpness FALL with
        # strength; the recommendation must follow the measured acutance
        # gain, not assume more strength = sharper.
        points = [
            {'strength': 0.25, 'overshoot': 0.02, 'undershoot': 0.03, 'mtf_peak': 1.05, 'acutance_gain': 1.3},
            {'strength': 1.0, 'overshoot': 0.04, 'undershoot': 0.05, 'mtf_peak': 1.06, 'acutance_gain': 1.15},
            {'strength': 2.0, 'overshoot': 0.05, 'undershoot': 0.06, 'mtf_peak': 1.06, 'acutance_gain': 1.08},
        ]
        out = recommend_strength(points, 0.10, 1.25)
        assert out['strength'] == 0.25
        assert out['acutance_gain'] == 1.3


class TestRecommendLimit:
    def test_picks_largest_within_cap(self):
        points = [
            {'limit': 0.25, 'overshoot': 0.03, 'undershoot': 0.04},
            {'limit': 0.5, 'overshoot': 0.08, 'undershoot': 0.09},
            {'limit': 1.0, 'overshoot': 0.2, 'undershoot': 0.25},
        ]
        assert recommend_limit(points, 0.10)['limit'] == 0.5

    def test_none_within_cap(self):
        out = recommend_limit([{'limit': 0.25, 'overshoot': 0.3, 'undershoot': 0.3}], 0.10)
        assert out['limit'] is None
        assert 'cap' in out['reason']

    def test_disengaged_limit_not_recommended(self):
        # At a weak strength the delta cap never engages: halos flat across
        # the candidates means every limit passes trivially — recommend none.
        points = [
            {'limit': 0.125, 'overshoot': 0.001, 'undershoot': 0.0},
            {'limit': 0.5, 'overshoot': 0.008, 'undershoot': 0.009},
            {'limit': 2.0, 'overshoot': 0.024, 'undershoot': 0.025},
        ]
        out = recommend_limit(points, 0.10)
        assert out['limit'] == 2.0  # 2.4% spread: the limit IS engaging
        flat = [
            {'limit': 0.125, 'overshoot': 0.001, 'undershoot': 0.0},
            {'limit': 2.0, 'overshoot': 0.006, 'undershoot': 0.005},
        ]
        out = recommend_limit(flat, 0.10)
        assert out['limit'] is None
        assert 'never engages' in out['reason']


class TestClassifyEdges:
    def edge(self, low, high):
        return {'edge_low': low, 'edge_high': high}

    def test_classification_boundaries(self):
        edges = [
            self.edge(10.0, 240.0),  # step 230 -> high
            self.edge(60.0, 180.0),  # step 120 -> moderate
            self.edge(120.0, 135.0),  # step 15 -> too shallow
        ]
        assert classify_edges(edges) == ['high', 'moderate', None]

    def test_missing_plateaus_unclassified(self):
        assert classify_edges([{'edge_low': None, 'edge_high': None}]) == [None]


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

    def test_limit_and_combinations(self):
        tuning = v2_tuning({'threshold': 0.75, 'limit': 0.5, 'strength': 1.0})
        set_sharpen(tuning, strength=1.25, limit=0.25)
        block = next(e for e in tuning['algorithms'] if 'rpi.sharpen' in e)['rpi.sharpen']
        assert block == {'threshold': 0.75, 'limit': 0.25, 'strength': 1.25}

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
    # A v1 (pre-sections) file must migrate on read into the threshold section.
    stored = {'version': 1, 'points': [], 'recommended': {'threshold': 0.2}}
    path = sharpen_mod.results_path(proj)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(stored))
    data = client.get('/projects/cam/sharpen/data').get_json()
    assert data['results']['version'] == 2
    assert data['results']['threshold']['recommended']['threshold'] == 0.2
    assert data['results']['strength'] is None
    # Migration is read-time only: the disk file stays v1.
    assert json.loads(path.read_text())['version'] == 1


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
    # The applied record lands inside the (v1-migrated) threshold section.
    assert json.loads(results_file.read_text())['threshold']['applied']['threshold'] == 0.2


def test_sharpen_apply_strength_and_limit(tmp_path, monkeypatch):
    import ctt_server.app as app_mod
    import ctt_server.sharpen as sharpen_mod

    client, proj = _client(tmp_path)
    monkeypatch.setattr(app_mod, 'platform_target', lambda: 'pisp')
    proj.output_dir.mkdir(parents=True)
    tuning_path = proj.output_dir / 'cam_pisp.json'
    tuning_path.write_text(json.dumps(v2_tuning({'threshold': 0.75, 'limit': 0.5, 'strength': 1.0})))
    results_file = sharpen_mod.results_path(proj)
    results_file.parent.mkdir(parents=True)
    seeded = {'version': 2, 'threshold': {'applied': None}, 'strength': {'applied': None}}
    results_file.write_text(json.dumps(seeded))

    r = client.post('/projects/cam/sharpen/apply', json={'strength': 1.25, 'limit': 0.25})
    assert r.status_code == 200
    block = next(e for e in json.loads(tuning_path.read_text())['algorithms'] if 'rpi.sharpen' in e)['rpi.sharpen']
    assert block == {'threshold': 0.75, 'limit': 0.25, 'strength': 1.25}
    stored = json.loads(results_file.read_text())
    assert stored['strength']['applied'] == {
        'strength': 1.25,
        'limit': 0.25,
        'at': stored['strength']['applied']['at'],
    }
    assert stored['threshold']['applied'] is None  # untouched

    # Out-of-range and empty bodies are rejected.
    assert client.post('/projects/cam/sharpen/apply', json={}).status_code == 400
    assert client.post('/projects/cam/sharpen/apply', json={'strength': 99}).status_code == 400
    assert client.post('/projects/cam/sharpen/apply', json={'limit': -1}).status_code == 400


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


def test_strength_stream_refuses_while_running(tmp_path, monkeypatch):
    import ctt_server.app as app_mod
    import ctt_server.sharpen as sharpen_mod

    client, _ = _client(tmp_path)
    monkeypatch.setattr(app_mod, 'get_shared_camera', lambda: object())
    assert sharpen_mod._sharpen_lock.acquire(blocking=False)
    try:
        r = client.get('/projects/cam/sharpen/strength/stream')
        assert b'already running' in r.data
    finally:
        sharpen_mod._sharpen_lock.release()


def test_strength_stream_invalid_settings_streams_error(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get('/projects/cam/sharpen/strength/stream?strengths=a,b')
    assert b'invalid sweep settings' in r.data


def test_merge_results_preserves_other_section(tmp_path):
    import ctt_server.sharpen as sharpen_mod

    _client_unused, proj = _client(tmp_path)
    sharpen_mod._merge_results(proj, 'threshold', {'recommended': {'threshold': 0.75}})
    sharpen_mod._merge_results(proj, 'strength', {'recommended': {'strength': 1.0}})
    stored = json.loads(sharpen_mod.results_path(proj).read_text())
    assert stored['version'] == 2
    assert stored['threshold']['recommended']['threshold'] == 0.75  # survived the strength write
    assert stored['strength']['recommended']['strength'] == 1.0
    # And the reverse direction.
    sharpen_mod._merge_results(proj, 'threshold', {'recommended': {'threshold': 0.4}})
    stored = json.loads(sharpen_mod.results_path(proj).read_text())
    assert stored['strength']['recommended']['strength'] == 1.0


def test_strength_point_tunings_pin_the_reference_limit():
    points = strength_point_tunings([0.5, 1.0])
    # Baseline first, then the candidates in order — and every point carries
    # the explicit reference limit, so the base tuning's limit never leaks in.
    assert points[0]['strength'] == 0.0
    assert [p['strength'] for p in points[1:]] == [0.5, 1.0]
    assert all(p['limit'] == REFERENCE_LIMIT for p in points)


def test_strength_point_tunings_override_base_limit(tmp_path):
    import ctt_server.sharpen as sharpen_mod

    # A base tuning arriving with a tight limit must not steer the phase:
    # the written point files carry the reference limit instead.
    base = tmp_path / 'base.json'
    base.write_text(json.dumps(v2_tuning({'threshold': 0.75, 'limit': 0.125, 'strength': 1.0})))
    paths = sharpen_mod._write_point_tunings(base, tmp_path / 'tmp', strength_point_tunings([0.5]), prefix='s')
    for path in paths:
        block = json.loads(path.read_text())['algorithms'][1]['rpi.sharpen']
        assert block['limit'] == REFERENCE_LIMIT


def test_write_point_tunings_unique_files_and_kwargs(tmp_path):
    import ctt_server.sharpen as sharpen_mod

    base = tmp_path / 'base.json'
    base.write_text(json.dumps(v2_tuning({'threshold': 0.75, 'limit': 0.5, 'strength': 1.0})))
    tmp_dir = tmp_path / 'tmp'
    points = [{'strength': 0.0}, {'strength': 1.5, 'limit': 0.25}]
    paths = sharpen_mod._write_point_tunings(base, tmp_dir, points, prefix='s')
    assert [p.name for p in paths] == ['s_0.json', 's_1.json']
    first = json.loads(paths[0].read_text())['algorithms'][1]['rpi.sharpen']
    second = json.loads(paths[1].read_text())['algorithms'][1]['rpi.sharpen']
    assert first['strength'] == 0.0 and first['threshold'] == 0.75
    assert second == {'threshold': 0.75, 'limit': 0.25, 'strength': 1.5}
