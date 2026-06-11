# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for burst capture: indexed filenames, in-CTT averaging of burst
# groups, and single-frame noise statistics.

import types

import numpy as np

from ctt.core.camera import Camera, burst_group_key, get_col_lux
from ctt_server import naming, sessions


class TestBurstNaming:
    def test_macbeth_indexed_name(self):
        name = naming.build_filename('macbeth', 5000, lux=800, label='d65', index=2)
        assert name == 'd65_5000k_800l_2.dng'
        # CTT's parser still reads the tags through the index suffix.
        assert get_col_lux(name) == (5000, 800)
        assert naming.parse_filename(name)[0] == 'macbeth'

    def test_macbeth_unindexed_default(self):
        assert naming.build_filename('macbeth', 5000, lux=800, label='d65') == 'd65_5000k_800l.dng'

    def test_next_index_macbeth(self):
        existing = ['d65_5000k_800l_0.dng', 'd65_5000k_800l_1.dng', 'd65_3000k_800l_0.dng']
        assert naming.next_index(existing, 'macbeth', 5000, lux=800, label='d65') == 2
        assert naming.next_index(existing, 'macbeth', 3000, lux=800, label='d65') == 1
        assert naming.next_index(existing, 'macbeth', 5000, lux=500, label='d65') == 0

    def test_group_key(self):
        assert burst_group_key('d65_5000k_800l_2.dng') == 'd65_5000k_800l.dng'
        assert burst_group_key('d65_5000k_800l.dng') == 'd65_5000k_800l.dng'
        # ALSC-style names are not Macbeth groups (no lux tag before the index).
        assert burst_group_key('alsc_5000k_1.dng') == 'alsc_5000k_1.dng'


class TestBurstCaptureStorage:
    def test_add_capture_indexed_burst(self, tmp_path):
        ws = sessions.Workspace(tmp_path)
        proj = ws.create_project('cam')
        names = [proj.add_capture(b'D', 'macbeth', 5000, lux=800, indexed=True).filename for _ in range(3)]
        assert names == ['cam_5000k_800l_0.dng', 'cam_5000k_800l_1.dng', 'cam_5000k_800l_2.dng']
        # A plain capture keeps the index-free overwrite name.
        assert proj.add_capture(b'D', 'macbeth', 5000, lux=800).filename == 'cam_5000k_800l.dng'

    def test_capture_endpoint_burst(self, tmp_path, monkeypatch):
        import ctt_server.app as app_mod
        from ctt_server.app import create_app

        ws = sessions.Workspace(tmp_path)
        ws.create_project('cam')
        client = create_app(str(tmp_path)).test_client()

        class FakeCam:
            def capture_burst(self, frames, quality=95):
                return [(b'DNG', b'JPG', {'exposure': 1})] * frames

        monkeypatch.setattr(app_mod, 'get_shared_camera', lambda: FakeCam())
        r = client.post(
            '/projects/cam/capture',
            json={'image_type': 'macbeth', 'colour_temp': 5000, 'lux': 800, 'frames': 3},
        )
        assert r.status_code == 200
        d = r.get_json()
        assert [c['filename'] for c in d['added']] == [
            'cam_5000k_800l_0.dng',
            'cam_5000k_800l_1.dng',
            'cam_5000k_800l_2.dng',
        ]
        assert d['counts']['macbeth'] == 3

        # Single capture: index-free name, single-entry list.
        r = client.post('/projects/cam/capture', json={'image_type': 'macbeth', 'colour_temp': 3000, 'lux': 500})
        assert [c['filename'] for c in r.get_json()['added']] == ['cam_3000k_500l.dng']


class TestGroupAveraging:
    def _stub_loader(self, monkeypatch, values):
        """dng_load_image stub: each call returns 64x64 channels of the next value."""
        import ctt.core.image_loader as loader_mod
        from ctt.core.image import Image

        calls = iter(values)

        def fake_load(cam, im_str, demosaic=True):
            img = Image()
            value = next(calls)
            img.channels = [np.full((64, 64), value, dtype=np.float64) for _ in range(4)]
            img.sigbits = 12
            img.blacklevel_16 = 0
            img.name = im_str.split('/')[-1]
            return img

        monkeypatch.setattr(loader_mod, 'dng_load_image', fake_load)
        # Chart detection stub: 6x4 grid of centres, corners spanning half the width.
        centres = np.array([[10 + c * 8, 10 + r * 8] for r in range(4) for c in range(6)], dtype=float)
        corners = np.array([[10, 10], [50, 10], [50, 40], [10, 40]], dtype=float)
        monkeypatch.setattr(loader_mod, 'find_macbeth', lambda cam, chan, cfg: (([corners], [centres]), 0.9))
        return loader_mod

    def test_group_averages_channels_and_keeps_single_patches(self, tmp_path, monkeypatch):
        loader_mod = self._stub_loader(monkeypatch, [1000.0, 3000.0])
        cam = Camera('out.json', json={})
        img = loader_mod.load_image_group(cam, ['/x/a_5000k_800l_0.dng', '/x/a_5000k_800l_1.dng'], (0, 0))
        assert img is not None
        assert img.frames_averaged == 2
        # Averaged channels feed the colour algorithms...
        assert float(np.mean(img.patches[0][0])) == 2000.0
        # ...while the noise calibration gets true single-frame data.
        assert float(np.mean(img.patches_single[0][0])) == 1000.0

    def test_single_member_group_is_plain_load(self, tmp_path, monkeypatch):
        loader_mod = self._stub_loader(monkeypatch, [3000.0])
        cam = Camera('out.json', json={})
        img = loader_mod.load_image_group(cam, ['/x/a_5000k_800l.dng'], (0, 0))
        assert img.frames_averaged == 1
        assert img.patches_single is None


class TestAddImgsGrouping:
    def test_macbeth_groups_processed_once(self, tmp_path, monkeypatch):
        import ctt.core.camera as camera_mod

        for f in ('d65_5000k_800l_0.dng', 'd65_5000k_800l_1.dng', 'e27_3000k_500l.dng', 'alsc_5000k_0.dng'):
            (tmp_path / f).write_bytes(b'DNG')

        group_calls = []

        def fake_group(cam, paths, mac_config):
            group_calls.append([p.split('/')[-1] for p in paths])
            return types.SimpleNamespace(macbeth_confidence=None, blacklevel_16=0, name='')

        monkeypatch.setattr(camera_mod, 'load_image_group', fake_group)
        monkeypatch.setattr(
            camera_mod,
            'load_image',
            lambda cam, a, mac_config=None, mac=True: types.SimpleNamespace(blacklevel_16=0, name=''),
        )

        cam = camera_mod.Camera('out.json', json={})
        cam.add_imgs(str(tmp_path) + '/', (0, 0))
        assert group_calls == [['d65_5000k_800l_0.dng', 'd65_5000k_800l_1.dng'], ['e27_3000k_500l.dng']]
        assert len(cam.imgs) == 2
        assert cam.imgs[0].name == 'd65_5000k_800l.dng'  # group name, not the indexed member
        assert len(cam.imgs_alsc) == 1


class TestNoiseUsesSingleFrame:
    def test_patches_single_preferred(self, monkeypatch):
        from ctt.algorithms.noise import noise

        cam = Camera('out.json', json={})
        rng = np.random.default_rng(1)
        means = np.linspace(2000, 40000, 24)
        flat = [[np.full(100, m) for m in means]] * 4  # zero spatial noise
        noisy = [[m + rng.normal(0, np.sqrt(m), 100) for m in means]] * 4

        img = types.SimpleNamespace(name='x.dng', patches=flat, patches_single=noisy, blacklevel_16=0, againQ8_norm=1.0)
        slope_single = noise(cam, img)[0]
        img.patches_single = None
        slope_avg = noise(cam, img)[0]
        assert slope_single > 0.5  # fitted the noisy single-frame data
        assert abs(slope_avg) < 1e-6  # averaged (flat) data has no spatial noise
        assert 'Using single-frame patches' in str(cam.log)
