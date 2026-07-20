# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for Workspace.rename_project: the directory move plus the capture,
# output-file and reference rewrites. Hardware-free (dummy image bytes).

import json

import pytest

from ctt_server.sessions import Workspace


def _populate(ws: Workspace, name: str):
    """A project with Macbeth + ALSC + dark captures and a full output/ set."""
    proj = ws.create_project(name)
    proj.add_capture(b'dng', 'macbeth', 5535, lux=1085)  # label defaults to project name
    proj.add_capture(b'dng', 'alsc', 3500)  # generic prefix, no project name
    proj.add_capture(b'dng', 'dark')
    out = proj.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{name}_pisp.json').write_text('{"version": 2.0}')  # tuning: must not be rewritten
    (out / f'{name}_pisp.log').write_text(
        f'Output /ws/{name}/output/{name}_pisp.json\nLoading imx662avgtest_5535k_1085l.dng\n'
        if name == 'imx662_avg_test'
        else f'Output {name}_pisp.json\n'
    )
    (out / f'{name}_pisp_metrics.json').write_text(json.dumps({'source': f'{name}_ref'}))
    return proj


def test_rename_moves_dir_and_reprefixes_everything(tmp_path):
    ws = Workspace(tmp_path)
    _populate(ws, 'imx662_avg_test')
    tuning = ws.root / 'imx662_avg_test' / 'output' / 'imx662_avg_test_pisp.json'
    tuning_mtime = tuning.stat().st_mtime

    proj = ws.rename_project('imx662_avg_test', 'imx662')

    assert proj.name == 'imx662'
    assert (ws.root / 'imx662').is_dir()
    assert not (ws.root / 'imx662_avg_test').exists()

    # Every capture entry still resolves to a file on disk.
    for c in proj.captures:
        assert (proj.path / c.filename).exists(), c.filename
    macbeth = next(c for c in proj.captures if c.image_type == 'macbeth')
    assert macbeth.filename == 'imx662_5535k_1085l.dng'  # prefix rewritten
    assert macbeth.label == 'imx662'
    # ALSC/dark carry no project prefix, so they are untouched.
    assert {c.filename for c in proj.captures if c.image_type != 'macbeth'} == {'alsc_3500k_0.dng', 'dark_0.dng'}

    # Output files reprefixed.
    names = {p.name for p in proj.output_dir.iterdir()}
    assert names == {'imx662_pisp.json', 'imx662_pisp.log', 'imx662_pisp_metrics.json'}

    # project.json is self-consistent and carries the new name.
    saved = json.loads(proj.sidecar.read_text())
    assert saved['name'] == 'imx662'

    # Tuning JSON was moved, not rewritten: mtime preserved, content intact.
    moved = proj.output_dir / 'imx662_pisp.json'
    assert moved.stat().st_mtime == tuning_mtime
    assert moved.read_text() == '{"version": 2.0}'

    # Logs/metrics no longer mention either old spelling; new name present.
    log = (proj.output_dir / 'imx662_pisp.log').read_text()
    assert 'imx662_avg_test' not in log and 'imx662avgtest' not in log
    assert 'imx662_5535k_1085l.dng' in log and '/ws/imx662/output/imx662_pisp.json' in log
    metrics = (proj.output_dir / 'imx662_pisp_metrics.json').read_text()
    assert 'imx662_avg_test' not in metrics and metrics.strip() == '{"source": "imx662_ref"}'


def test_rename_onto_existing_raises(tmp_path):
    ws = Workspace(tmp_path)
    _populate(ws, 'imx662_avg_test')
    ws.create_project('imx662')
    with pytest.raises(FileExistsError):
        ws.rename_project('imx662_avg_test', 'imx662')
    assert (ws.root / 'imx662_avg_test').is_dir()  # source untouched on failure


def test_rename_invalid_name_raises(tmp_path):
    ws = Workspace(tmp_path)
    _populate(ws, 'imx662_avg_test')
    with pytest.raises(ValueError):
        ws.rename_project('imx662_avg_test', '   ')


def test_rename_missing_source_raises(tmp_path):
    ws = Workspace(tmp_path)
    with pytest.raises(FileNotFoundError):
        ws.rename_project('nope', 'whatever')


def test_rename_same_name_is_noop(tmp_path):
    ws = Workspace(tmp_path)
    _populate(ws, 'imx662')
    proj = ws.rename_project('imx662', 'imx662')
    assert proj.name == 'imx662'
    assert (proj.path / 'imx662_5535k_1085l.dng').exists()


# --- route wiring ----------------------------------------------------------


def _client(tmp_path):
    from ctt_server.app import create_app

    return create_app(str(tmp_path)).test_client()


def test_rename_route_redirects_to_new_project(tmp_path):
    ws = Workspace(tmp_path)
    _populate(ws, 'imx662_avg_test')
    client = _client(tmp_path)

    resp = client.post('/projects/imx662_avg_test/rename', data={'new_name': 'imx662'})
    assert resp.status_code == 302
    assert resp.headers['Location'].endswith('/projects/imx662')
    assert client.get('/projects/imx662').status_code == 200
    assert client.get('/projects/imx662_avg_test').status_code == 404


def test_rename_route_shows_error_when_target_exists(tmp_path):
    ws = Workspace(tmp_path)
    _populate(ws, 'imx662_avg_test')
    ws.create_project('imx662')
    client = _client(tmp_path)

    resp = client.post('/projects/imx662_avg_test/rename', data={'new_name': 'imx662'})
    assert resp.status_code == 200  # re-rendered projects page, not a redirect
    assert b'already exists' in resp.data
    assert (ws.root / 'imx662_avg_test').is_dir()  # source untouched
