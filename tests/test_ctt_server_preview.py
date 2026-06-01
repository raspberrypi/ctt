# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the Results-page live-preview-test routes. The camera reload and platform
# detection are monkeypatched, so these run without picamera2 or hardware.

import pytest

from ctt_server import app as app_module
from ctt_server import sessions


class FakeCam:
    model = 'FakeCam'


@pytest.fixture
def client(tmp_path):
    return app_module.create_app(tmp_path).test_client()


def _project_with_tuning(tmp_path, name, target=None):
    proj = sessions.Workspace(tmp_path).create_project(name)
    if target:
        proj.output_dir.mkdir(parents=True, exist_ok=True)
        (proj.output_dir / f'{name}_{target}.json').write_text('{}')
    return proj


def test_preview_test_loads_platform_tuning(client, tmp_path, monkeypatch):
    _project_with_tuning(tmp_path, 'imx', 'pisp')
    calls = {}
    monkeypatch.setattr(app_module, 'platform_target', lambda: 'pisp')
    monkeypatch.setattr(
        app_module,
        'reload_shared_camera',
        lambda tuning_file=None, preview_max_width=1280: (
            calls.update(tuning=tuning_file, width=preview_max_width) or FakeCam()
        ),
    )
    r = client.post('/projects/imx/preview-test')
    data = r.get_json()
    assert r.status_code == 200
    assert data['target'] == 'pisp'
    assert data['tuning'] == 'imx_pisp.json'
    assert calls['tuning'].endswith('imx_pisp.json')
    assert calls['width'] == 1920  # bigger/sharper stream than the capture preview


def test_preview_test_400_when_platform_tuning_missing(client, tmp_path, monkeypatch):
    _project_with_tuning(tmp_path, 'imxb', 'vc4')  # only a vc4 tuning exists
    monkeypatch.setattr(app_module, 'platform_target', lambda: 'pisp')  # but this Pi runs PiSP
    monkeypatch.setattr(app_module, 'reload_shared_camera', lambda **k: FakeCam())
    r = client.post('/projects/imxb/preview-test')
    assert r.status_code == 400
    assert 'PISP' in r.get_json()['error']


def test_preview_default_restores_default(client, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        app_module,
        'reload_shared_camera',
        lambda tuning_file=None, preview_max_width=1280: calls.update(tuning=tuning_file) or FakeCam(),
    )
    r = client.post('/api/preview-default')
    assert r.status_code == 200
    assert calls['tuning'] is None
