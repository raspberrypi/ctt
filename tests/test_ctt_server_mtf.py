# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the slanted-edge MTF (e-SFR) measurement.
#
# A synthetic edge with a Gaussian blur of known sigma has the analytic MTF
# exp(-2 * pi^2 * sigma^2 * f^2), so MTF50 = sqrt(ln 2 / 2) / (pi * sigma).
# That gives the maths a ground truth without any camera or DNG fixtures.

import numpy as np
from scipy.special import erf

from ctt_server.mtf import _PLANE_STEP, analyse_edge


def synthetic_edge(h=120, w=120, angle_deg=5.0, sigma=1.0, lo=0.1, hi=0.9, transpose=False):
    """A Gaussian-blurred step edge slanted angle_deg off vertical."""
    slope = np.tan(np.radians(angle_deg))
    cc, rr = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    dist = cc - (w / 2 + slope * (rr - h / 2))
    img = lo + (hi - lo) * 0.5 * (1 + erf(dist / (sigma * np.sqrt(2))))
    return img.T if transpose else img


def analytic_mtf50(sigma):
    # In plane cycles/px; analyse_edge reports sensor cycles/px (plane / _PLANE_STEP).
    return np.sqrt(np.log(2) / 2) / (np.pi * sigma) / _PLANE_STEP


class TestAnalyseEdge:
    def test_gaussian_edge_matches_analytic_mtf50(self):
        for sigma in (0.8, 1.5):
            out = analyse_edge(synthetic_edge(sigma=sigma))
            assert out['ok'], out
            np.testing.assert_allclose(out['mtf50'], analytic_mtf50(sigma), rtol=0.07)

    def test_horizontal_edge_is_transposed(self):
        out = analyse_edge(synthetic_edge(sigma=1.0, transpose=True))
        assert out['ok'], out
        np.testing.assert_allclose(out['mtf50'], analytic_mtf50(1.0), rtol=0.07)

    def test_sharper_edge_higher_mtf50(self):
        sharp = analyse_edge(synthetic_edge(sigma=0.7))
        soft = analyse_edge(synthetic_edge(sigma=2.5))
        assert sharp['ok'] and soft['ok']
        assert sharp['mtf50'] > soft['mtf50']

    def test_edge_angle_reported(self):
        out = analyse_edge(synthetic_edge(angle_deg=7.0))
        assert out['ok']
        assert abs(abs(out['angle_deg']) - 7.0) < 1.0

    def test_flat_roi_fails(self):
        out = analyse_edge(np.full((100, 100), 0.5))
        assert not out['ok']
        assert 'contrast' in out['reason']

    def test_square_edge_rejected(self):
        # A perfectly vertical edge can't be supersampled; it must be flagged.
        out = analyse_edge(synthetic_edge(angle_deg=0.0))
        assert not out['ok']
        assert 'angle' in out['reason']

    def test_tiny_roi_fails(self):
        out = analyse_edge(np.zeros((8, 8)))
        assert not out['ok']


class TestMeasureRois:
    def test_rois_cropped_from_green_plane(self, monkeypatch):
        import ctt_server.mtf as mtf_mod

        # Plane with an edge in the top-left 60x60 region only (sensor 120x120).
        plane = np.full((200, 200), 0.5)
        plane[:60, :60] = synthetic_edge(60, 60, sigma=1.0)
        monkeypatch.setattr(mtf_mod, 'green_plane', lambda path: plane)

        rois = [
            {'x': 0, 'y': 0, 'w': 120, 'h': 120},  # sensor coords -> plane top-left 60x60
            {'x': 240, 'y': 240, 'w': 120, 'h': 120},  # flat region: should fail cleanly
        ]
        out = mtf_mod.measure_rois('fake.dng', rois)
        assert out[0]['ok'] and out[0]['mtf50'] is not None
        assert not out[1]['ok']  # flat region fails cleanly (wherever snap leaves it)
        assert out[0]['w'] == 120  # ROI size echoed back


# --- endpoints ---


def _client(tmp_path):
    from ctt_server import sessions
    from ctt_server.app import create_app

    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('cam')
    return create_app(str(tmp_path)).test_client(), proj


def test_mtf_preview_and_measure_404_without_capture(tmp_path):
    client, _ = _client(tmp_path)
    assert client.get('/projects/cam/mtf/preview').status_code == 404
    r = client.post('/projects/cam/mtf/measure', json={'rois': [{'x': 0, 'y': 0, 'w': 100, 'h': 100}]})
    assert r.status_code == 404


def test_mtf_measure_validates_rois(tmp_path, monkeypatch):
    client, proj = _client(tmp_path)
    (proj.path / 'mtf').mkdir()
    (proj.path / 'mtf' / 'chart.dng').write_bytes(b'FAKEDNG')

    r = client.post('/projects/cam/mtf/measure', json={'rois': []})
    assert r.status_code == 400
    r = client.post('/projects/cam/mtf/measure', json={'rois': [{'x': 1}]})
    assert r.status_code == 400


def test_mtf_measure_returns_results(tmp_path, monkeypatch):
    import ctt_server.app as app_mod

    client, proj = _client(tmp_path)
    (proj.path / 'mtf').mkdir()
    (proj.path / 'mtf' / 'chart.dng').write_bytes(b'FAKEDNG')
    (proj.path / 'mtf' / 'chart.jpg').write_bytes(b'JPG')

    fake = [{'x': 0, 'y': 0, 'w': 100, 'h': 100, 'ok': True, 'mtf50': 0.21, 'angle_deg': 5.0, 'curve': []}]
    monkeypatch.setattr(app_mod.mtf, 'measure_rois', lambda path, rois: fake)

    assert client.get('/projects/cam/mtf/preview').status_code == 200
    r = client.post('/projects/cam/mtf/measure', json={'rois': [{'x': 0, 'y': 0, 'w': 100, 'h': 100}]})
    assert r.status_code == 200
    assert r.get_json()['rois'][0]['mtf50'] == 0.21


def test_mtf_measure_bad_dng_500(tmp_path):
    # A junk DNG must produce a clean 500, not a crash.
    client, proj = _client(tmp_path)
    (proj.path / 'mtf').mkdir()
    (proj.path / 'mtf' / 'chart.dng').write_bytes(b'NOTAREALDNG')
    r = client.post('/projects/cam/mtf/measure', json={'rois': [{'x': 0, 'y': 0, 'w': 100, 'h': 100}]})
    assert r.status_code == 500


def test_mtf_page_renders(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get('/projects/cam/mtf')
    assert r.status_code == 200
    assert b'mtfApp' in r.data
    assert client.get('/projects/nope/mtf').status_code == 404


# --- validity check + snapping ---


def test_whole_square_in_roi_rejected():
    # A complete dark band (both tilted edges inside the ROI) is not a single
    # edge: the ESF is non-monotonic and must be rejected, not given a bogus
    # MTF50. (Two superposed opposite slanted edges 20 plane px apart.)
    rising = synthetic_edge(h=120, w=120, sigma=1.0)  # edge at column 60
    falling = 1.0 - np.roll(rising, 20, axis=1)  # opposite edge at column 80
    band = np.minimum(rising, falling)  # dark outside, bright band inside
    out = analyse_edge(band)
    assert not out['ok']
    assert 'single clean edge' in out['reason']


def test_zone_plate_rejected():
    # Concentric rings (a zone plate) have strong gradients but no clean edge.
    cc, rr = np.meshgrid(np.arange(120, dtype=float), np.arange(120, dtype=float))
    radius = np.hypot(cc - 60, rr - 60)
    img = 0.5 + 0.4 * np.cos(radius**2 / 18)
    out = analyse_edge(img)
    assert not out['ok']


def test_snap_centres_edge_within_box(monkeypatch):
    import ctt_server.mtf as mtf_mod

    # Edge at plane x=150, inside the box but near its right border (outside
    # the +/-16 px ESF span of the box centre). Snapping must re-centre the
    # box on the edge — but only ever within the user's original footprint.
    plane = synthetic_edge(h=300, w=300, sigma=1.0)  # edge at column ~150
    monkeypatch.setattr(mtf_mod, 'green_plane', lambda path: plane)

    roi = {'x': 200, 'y': 100, 'w': 120, 'h': 120}  # plane x=100..160; edge at ~150
    out = mtf_mod.measure_rois('fake.dng', [roi])[0]
    assert out['ok'], out
    assert out['x'] > roi['x']  # moved right to centre the edge
    np.testing.assert_allclose(out['mtf50'], analytic_mtf50(1.0), rtol=0.1)


def test_failed_roi_keeps_original_position(monkeypatch):
    import ctt_server.mtf as mtf_mod

    # A box on featureless content fails — and must stay exactly where the
    # user put it (no jumping about on repeated Measure presses).
    plane = np.full((300, 300), 0.5)
    monkeypatch.setattr(mtf_mod, 'green_plane', lambda path: plane)

    roi = {'x': 200, 'y': 100, 'w': 120, 'h': 120}
    out = mtf_mod.measure_rois('fake.dng', [roi])[0]
    assert not out['ok']
    assert out['x'] == roi['x'] and out['y'] == roi['y']


def test_edge_spanning_partial_roi_height():
    # A bar that ends inside the ROI: the flat rows beyond its end must not
    # contaminate the ESF (regression — found on a real ISO 12233 capture).
    img = np.full((120, 120), 0.9)
    img[:70, :] = synthetic_edge(h=70, w=120, sigma=1.0)  # edge only in the top 70 rows
    out = analyse_edge(img)
    assert out['ok'], out
    np.testing.assert_allclose(out['mtf50'], analytic_mtf50(1.0), rtol=0.1)


# --- auto-detect ---


def test_auto_detect_finds_and_spreads_edges(monkeypatch):
    import ctt_server.mtf as mtf_mod

    # Two clean tilted edges in different zones of an otherwise-flat plane.
    plane = np.full((400, 600), 0.5)
    plane[20:140, 30:150] = synthetic_edge(120, 120, sigma=1.0)  # top-left, vertical edge
    plane[260:380, 450:570] = synthetic_edge(120, 120, sigma=1.0, transpose=True)  # bottom-right, horizontal
    monkeypatch.setattr(mtf_mod, 'green_plane', lambda path: plane)

    out = mtf_mod.auto_detect('fake.dng', max_regions=9)
    assert len(out) >= 2
    zones = {r['zone'] for r in out}
    assert any('top-left' in z for z in zones)
    assert any('bottom-right' in z for z in zones)
    for r in out:
        assert r['ok'] and r['mtf50'] is not None
    # No two same-shape boxes overlap on both axes (deduped); differently
    # shaped boxes may neighbour each other (they measure different edges).
    for i, a in enumerate(out):
        for b in out[i + 1 :]:
            if (a['w'], a['h']) == (b['w'], b['h']):
                assert abs(a['x'] - b['x']) >= a['w'] or abs(a['y'] - b['y']) >= a['h']


def test_auto_detect_flat_plane_empty(monkeypatch):
    import ctt_server.mtf as mtf_mod

    monkeypatch.setattr(mtf_mod, 'green_plane', lambda path: np.full((400, 600), 0.5))
    assert mtf_mod.auto_detect('fake.dng') == []


def test_auto_detect_respects_max_regions(monkeypatch):
    import ctt_server.mtf as mtf_mod

    # Many parallel edges: result must be capped.
    plane = np.full((400, 600), 0.1)
    for x0 in range(50, 550, 100):
        plane[:, x0 : x0 + 50] = synthetic_edge(400, 50, sigma=1.0, lo=0.1, hi=0.9)
    monkeypatch.setattr(mtf_mod, 'green_plane', lambda path: plane)
    assert len(mtf_mod.auto_detect('fake.dng', max_regions=4)) <= 4


def test_mtf_auto_endpoint(tmp_path, monkeypatch):
    import ctt_server.app as app_mod

    client, proj = _client(tmp_path)
    assert client.post('/projects/cam/mtf/auto').status_code == 404  # no capture yet
    (proj.path / 'mtf').mkdir()
    (proj.path / 'mtf' / 'chart.dng').write_bytes(b'FAKEDNG')
    fake = [{'x': 0, 'y': 0, 'w': 64, 'h': 128, 'ok': True, 'mtf50': 0.2, 'zone': 'centre', 'curve': []}]
    monkeypatch.setattr(app_mod.mtf, 'auto_detect', lambda path: fake)
    r = client.post('/projects/cam/mtf/auto')
    assert r.status_code == 200
    assert r.get_json()['rois'][0]['zone'] == 'centre'
