# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the ctt-server dark-frame capture flow and black level endpoint.

import pytest

from ctt_server import sessions


def test_dark_captures_are_indexed(tmp_path):
    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('cam')
    names = [proj.add_capture(b'D', 'dark').filename for _ in range(3)]
    assert names == ['dark_0.dng', 'dark_1.dng', 'dark_2.dng']
    assert proj.counts()['dark'] == 3


def test_dark_upload_is_tagged(tmp_path):
    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('cam')
    cap = proj.import_capture('dark_5.dng', b'D')
    assert (cap.image_type, cap.colour_temp, cap.lux, cap.label) == ('dark', None, None, None)


def test_capture_endpoint_dark_needs_no_colour_temp(tmp_path, monkeypatch):
    import ctt_server.app as app_mod
    from ctt_server.app import create_app

    ws = sessions.Workspace(tmp_path)
    ws.create_project('cam')
    client = create_app(str(tmp_path)).test_client()

    class FakeCam:
        def capture_burst(self, frames, quality=95):
            return [(b'DNG', b'JPG', {'exposure': 1})] * frames

    monkeypatch.setattr(app_mod, 'get_shared_camera', lambda: FakeCam())
    r = client.post('/projects/cam/capture', json={'image_type': 'dark', 'frames': 1})
    assert r.status_code == 200
    d = r.get_json()
    assert [c['filename'] for c in d['added']] == ['dark_0.dng']
    assert d['counts']['dark'] == 1
    # A second capture gets the next index rather than overwriting.
    r = client.post('/projects/cam/capture', json={'image_type': 'dark', 'frames': 1})
    assert [c['filename'] for c in r.get_json()['added']] == ['dark_1.dng']


@pytest.fixture
def blacklevel_client(tmp_path, monkeypatch):
    import ctt.algorithms.black_level as bl_mod
    from ctt_server.app import create_app

    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('cam')
    # The endpoint measures with measure_dark_dng; stub out the DNG read.
    monkeypatch.setattr(
        bl_mod,
        'measure_dark_dng',
        lambda path: {
            'name': path.split('/')[-1],
            'black_level': 4150.0,
            'r': 4000.0,
            'g': 4150.0,
            'b': 4300.0,
            'std': 10.0,
            'exposure': 10000,
            'gain': 2.0,
            'total_exposure': 20000,
        },
    )
    return create_app(str(tmp_path)).test_client(), proj


def test_blacklevel_endpoint_no_darks(blacklevel_client):
    client, _ = blacklevel_client
    d = client.get('/projects/cam/blacklevel').get_json()
    assert d == {'black_level': None, 'frames': []}


def test_blacklevel_endpoint_measures_darks(blacklevel_client):
    client, proj = blacklevel_client
    proj.add_capture(b'D', 'dark')
    proj.add_capture(b'D', 'dark')
    d = client.get('/projects/cam/blacklevel').get_json()
    assert d['black_level'] == 4150
    assert [f['name'] for f in d['frames']] == ['dark_0.dng', 'dark_1.dng']


def test_blacklevel_endpoint_skips_excluded(blacklevel_client):
    client, proj = blacklevel_client
    proj.add_capture(b'D', 'dark')
    proj.set_excluded('dark_0.dng', True)
    d = client.get('/projects/cam/blacklevel').get_json()
    assert d == {'black_level': None, 'frames': []}
