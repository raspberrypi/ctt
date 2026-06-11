# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for custom (hand-edited) tuning files: the Tuning-tab editor endpoints,
# the diff, the custom preview-test kind and the discard-on-rerun behaviour.

import json

import pytest

from ctt_server import app as app_module
from ctt_server import ctt_runner, sessions

ORIGINAL = '{\n    "version": 2.0,\n    "black_level": 4096\n}\n'
EDITED = '{\n    "version": 2.0,\n    "black_level": 3200\n}\n'


class FakeCam:
    model = 'FakeCam'


@pytest.fixture
def client(tmp_path):
    return app_module.create_app(tmp_path).test_client()


def _project_with_tuning(tmp_path, name='cam', target='pisp'):
    proj = sessions.Workspace(tmp_path).create_project(name)
    proj.output_dir.mkdir(parents=True, exist_ok=True)
    (proj.output_dir / f'{name}_{target}.json').write_text(ORIGINAL)
    return proj


def test_tuning_data_original_only(client, tmp_path):
    _project_with_tuning(tmp_path)
    d = client.get('/projects/cam/tuning-data/pisp').get_json()
    assert d == {'original': ORIGINAL, 'custom': None, 'diff': None}


def test_tuning_data_404s(client, tmp_path):
    _project_with_tuning(tmp_path)
    assert client.get('/projects/cam/tuning-data/vc4').status_code == 404  # no vc4 file
    assert client.get('/projects/cam/tuning-data/bogus').status_code == 404  # invalid target


def test_save_custom_and_diff(client, tmp_path):
    proj = _project_with_tuning(tmp_path)
    r = client.post('/projects/cam/tuning/custom/pisp', json={'json': EDITED})
    assert r.status_code == 200
    d = r.get_json()
    assert d['custom'] == EDITED
    assert (proj.output_dir / 'cam_pisp_custom.json').read_text() == EDITED
    assert '-    "black_level": 4096' in d['diff']
    assert '+    "black_level": 3200' in d['diff']
    assert '--- cam_pisp.json' in d['diff']
    # The state endpoint reports the same thing on a fresh fetch.
    assert client.get('/projects/cam/tuning-data/pisp').get_json() == d


def test_save_rejects_invalid_json(client, tmp_path):
    proj = _project_with_tuning(tmp_path)
    r = client.post('/projects/cam/tuning/custom/pisp', json={'json': '{not json'})
    assert r.status_code == 400
    assert 'Invalid JSON' in r.get_json()['error']
    assert not (proj.output_dir / 'cam_pisp_custom.json').exists()


def test_revert_deletes_custom(client, tmp_path):
    proj = _project_with_tuning(tmp_path)
    client.post('/projects/cam/tuning/custom/pisp', json={'json': EDITED})
    r = client.post('/projects/cam/tuning/custom/pisp/delete')
    assert r.status_code == 200
    assert r.get_json()['custom'] is None
    assert not (proj.output_dir / 'cam_pisp_custom.json').exists()


def test_download_custom(client, tmp_path):
    _project_with_tuning(tmp_path)
    assert client.get('/projects/cam/download/custom/pisp').status_code == 404  # none yet
    client.post('/projects/cam/tuning/custom/pisp', json={'json': EDITED})
    r = client.get('/projects/cam/download/custom/pisp')
    assert r.status_code == 200
    assert r.data.decode() == EDITED


def test_preview_test_custom_kind(client, tmp_path, monkeypatch):
    _project_with_tuning(tmp_path)
    calls = {}
    monkeypatch.setattr(app_module, 'platform_target', lambda: 'pisp')
    monkeypatch.setattr(
        app_module,
        'reload_shared_camera',
        lambda tuning_file=None, preview_max_width=1280, validate=False: (
            calls.update(tuning=tuning_file, validate=validate) or FakeCam()
        ),
    )
    # No custom file yet: a clear 400, and the camera is not reloaded.
    r = client.post('/projects/cam/preview-test', json={'kind': 'custom'})
    assert r.status_code == 400
    assert 'Tuning tab' in r.get_json()['error']

    client.post('/projects/cam/tuning/custom/pisp', json={'json': EDITED})
    r = client.post('/projects/cam/preview-test', json={'kind': 'custom'})
    d = r.get_json()
    assert r.status_code == 200
    assert (d['kind'], d['tuning']) == ('custom', 'cam_pisp_custom.json')
    assert calls['tuning'].endswith('cam_pisp_custom.json')
    assert calls['validate'] is True  # hand edits are canary-tested before loading

    # The default kind still loads the generated original, without the canary.
    r = client.post('/projects/cam/preview-test')
    assert r.get_json()['tuning'] == 'cam_pisp.json'
    assert calls['validate'] is False


def test_validate_tuning_file_reports_libcamera_error(monkeypatch):
    from ctt_server import camera as camera_mod

    class FakeProc:
        returncode = 1
        stdout = ''
        stderr = 'Traceback ...\nERROR IPA module failed to load tuning\nRuntimeError: boom'

    monkeypatch.setattr(camera_mod.subprocess, 'run', lambda *a, **kw: FakeProc())
    with pytest.raises(camera_mod.CameraError, match='IPA module failed'):
        camera_mod.validate_tuning_file('/tmp/x_custom.json')


def test_validate_tuning_file_passes_on_success(monkeypatch):
    from ctt_server import camera as camera_mod

    class FakeProc:
        returncode = 0
        stdout = ''
        stderr = ''

    monkeypatch.setattr(camera_mod.subprocess, 'run', lambda *a, **kw: FakeProc())
    camera_mod.validate_tuning_file('/tmp/x_custom.json')  # must not raise


def test_run_stream_discards_custom(tmp_path, monkeypatch):
    import ctt.core.runner as runner_mod

    proj = _project_with_tuning(tmp_path)
    custom = proj.output_dir / 'cam_pisp_custom.json'
    custom.write_text(EDITED)
    other = proj.output_dir / 'cam_vc4_custom.json'
    other.write_text(EDITED)
    monkeypatch.setattr(runner_mod, 'run_ctt_targets', lambda *a, **kw: None)

    lines = list(ctt_runner.run_ctt_stream(proj, ['pisp'], 'full', {}))
    assert not custom.exists()  # the run target's edits are discarded...
    assert other.exists()  # ...but other targets' edits are kept
    assert any('Discarded custom tuning edits for pisp' in line for line in lines)


def test_run_stream_keeps_custom_on_failure(tmp_path, monkeypatch):
    import ctt.core.runner as runner_mod

    proj = _project_with_tuning(tmp_path)
    custom = proj.output_dir / 'cam_pisp_custom.json'
    custom.write_text(EDITED)

    def boom(*a, **kw):
        raise RuntimeError('run failed')

    monkeypatch.setattr(runner_mod, 'run_ctt_targets', boom)
    lines = list(ctt_runner.run_ctt_stream(proj, ['pisp'], 'full', {}))
    assert custom.exists()  # a failed run must not throw the edits away
    assert lines[-1] == 'CTT_EXIT 1'


def test_save_roundtrips_arbitrary_valid_json(client, tmp_path):
    _project_with_tuning(tmp_path)
    text = json.dumps({'algorithms': [{'rpi.black_level': {'black_level': 3200}}]}, indent=4)
    r = client.post('/projects/cam/tuning/custom/pisp', json={'json': text})
    assert r.status_code == 200
    assert r.get_json()['custom'] == text
