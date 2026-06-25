# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for custom (hand-edited) tuning variants: the Tuning-tab editor
# endpoints, the diff, the slug-named variant files + manifest (create/edit/
# copy/delete, migration, reconcile, staleness), the System (built-in) tuning
# view, the custom preview-test kind and the keep-on-rerun behaviour.

import json
import os

import pytest

from ctt_server import app as app_module
from ctt_server import ctt_runner, sessions

ORIGINAL = '{\n    "version": 2.0,\n    "black_level": 4096\n}\n'
EDITED = '{\n    "version": 2.0,\n    "black_level": 3200\n}\n'
SYSTEM = '{\n    "version": 2.0,\n    "black_level": 2048\n}\n'


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


def _create(client, label, text=EDITED, target='pisp'):
    return client.post(f'/projects/cam/tuning/custom/{target}', json={'label': label, 'json': text})


# --- reading -------------------------------------------------------------


def test_tuning_data_original_only(client, tmp_path):
    _project_with_tuning(tmp_path)
    d = client.get('/projects/cam/tuning-data/pisp').get_json()
    assert d == {
        'original': ORIGINAL,
        'custom': None,
        'diff': None,
        'variants': [],
        'selected': None,
        'existing': None,
    }


def test_tuning_data_404s(client, tmp_path):
    _project_with_tuning(tmp_path)
    assert client.get('/projects/cam/tuning-data/vc4').status_code == 404  # no vc4 file
    assert client.get('/projects/cam/tuning-data/bogus').status_code == 404  # invalid target


def test_existing_tuning_from_metrics(client, tmp_path):
    proj = _project_with_tuning(tmp_path)
    system = tmp_path / 'imx_system.json'
    system.write_text(SYSTEM)
    (proj.output_dir / 'cam_pisp_metrics.json').write_text(json.dumps({'default_tuning_path': str(system)}))
    assert client.get('/projects/cam/tuning-data/pisp').get_json()['existing'] == SYSTEM


# --- create / edit / copy / delete --------------------------------------


def test_create_variant_and_diff(client, tmp_path):
    proj = _project_with_tuning(tmp_path)
    d = _create(client, 'Warm tweak').get_json()
    assert d['custom'] == EDITED
    assert d['selected'] == 'warm-tweak'
    assert d['variants'] == [{'id': 'warm-tweak', 'label': 'Warm tweak', 'stale': False}]
    assert (proj.output_dir / 'cam_pisp_custom_warm-tweak.json').read_text() == EDITED
    assert '-    "black_level": 4096' in d['diff']
    assert '+    "black_level": 3200' in d['diff']
    assert '--- cam_pisp.json' in d['diff']


def test_create_requires_label(client, tmp_path):
    _project_with_tuning(tmp_path)
    r = client.post('/projects/cam/tuning/custom/pisp', json={'label': '  ', 'json': EDITED})
    assert r.status_code == 400
    assert 'label' in r.get_json()['error'].lower()


def test_create_rejects_invalid_json(client, tmp_path):
    proj = _project_with_tuning(tmp_path)
    r = _create(client, 'Bad', text='{not json')
    assert r.status_code == 400
    assert 'Invalid JSON' in r.get_json()['error']
    assert not (proj.output_dir / 'cam_pisp_custom_bad.json').exists()


def test_duplicate_label_gets_distinct_slug(client, tmp_path):
    proj = _project_with_tuning(tmp_path)
    assert _create(client, 'My tweak').get_json()['selected'] == 'my-tweak'
    assert _create(client, 'My tweak').get_json()['selected'] == 'my-tweak-2'
    assert (proj.output_dir / 'cam_pisp_custom_my-tweak.json').exists()
    assert (proj.output_dir / 'cam_pisp_custom_my-tweak-2.json').exists()


def test_index_file_is_not_a_variant(client, tmp_path):
    proj = _project_with_tuning(tmp_path)
    _create(client, 'One')
    assert (proj.output_dir / 'cam_pisp_custom_index.json').exists()  # the manifest
    variants = client.get('/projects/cam/tuning-data/pisp').get_json()['variants']
    assert [v['id'] for v in variants] == ['one']  # 'index' is not surfaced


def test_edit_variant_in_place(client, tmp_path):
    proj = _project_with_tuning(tmp_path)
    _create(client, 'One')
    other = '{\n    "version": 2.0,\n    "black_level": 1000\n}\n'
    r = client.post('/projects/cam/tuning/custom/pisp/one', json={'json': other})
    assert r.status_code == 200
    d = r.get_json()
    assert d['custom'] == other and d['selected'] == 'one'
    assert (proj.output_dir / 'cam_pisp_custom_one.json').read_text() == other
    assert [v['id'] for v in d['variants']] == ['one']  # no new file


def test_edit_missing_variant_404s(client, tmp_path):
    _project_with_tuning(tmp_path)
    assert client.post('/projects/cam/tuning/custom/pisp/nope', json={'json': EDITED}).status_code == 404


def test_copy_variant(client, tmp_path):
    proj = _project_with_tuning(tmp_path)
    _create(client, 'Original tweak')
    r = client.post('/projects/cam/tuning/custom/pisp/original-tweak/copy', json={'label': 'A copy'})
    assert r.status_code == 200
    d = r.get_json()
    assert d['selected'] == 'a-copy'
    assert [v['label'] for v in d['variants']] == ['Original tweak', 'A copy']
    assert (proj.output_dir / 'cam_pisp_custom_a-copy.json').read_text() == EDITED


def test_delete_variant(client, tmp_path):
    proj = _project_with_tuning(tmp_path)
    _create(client, 'One')
    _create(client, 'Two')
    r = client.post('/projects/cam/tuning/custom/pisp/one/delete')
    assert r.status_code == 200
    d = r.get_json()
    assert [v['id'] for v in d['variants']] == ['two']
    assert d['selected'] == 'two'  # falls back to the surviving variant
    assert not (proj.output_dir / 'cam_pisp_custom_one.json').exists()


def test_invalid_slug_rejected(client, tmp_path):
    _project_with_tuning(tmp_path)
    _create(client, 'One')
    assert client.post('/projects/cam/tuning/custom/pisp/Bad-Caps', json={'json': EDITED}).status_code == 404
    assert client.post('/projects/cam/tuning/custom/pisp/has_underscore/delete').status_code == 404
    assert client.post('/projects/cam/tuning/custom/pisp/index/delete').status_code == 404  # reserved


def test_too_many_variants(client, tmp_path):
    _project_with_tuning(tmp_path)
    for i in range(20):  # MAX_VARIANTS
        assert _create(client, f'V{i}').status_code == 200
    r = _create(client, 'one too many')
    assert r.status_code == 400
    assert 'Too many' in r.get_json()['error']


# --- migration & reconcile ----------------------------------------------


def test_legacy_custom_migrated(client, tmp_path):
    proj = _project_with_tuning(tmp_path)
    (proj.output_dir / 'cam_pisp_custom.json').write_text(EDITED)
    d = client.get('/projects/cam/tuning-data/pisp').get_json()
    assert d['variants'] == [{'id': 'custom', 'label': 'Custom', 'stale': False}]
    assert d['custom'] == EDITED
    assert not (proj.output_dir / 'cam_pisp_custom.json').exists()  # renamed
    assert (proj.output_dir / 'cam_pisp_custom_custom.json').read_text() == EDITED


def test_integer_scheme_migrated_to_slugs(client, tmp_path):
    proj = _project_with_tuning(tmp_path)
    # Files + manifest as written by the earlier integer-id scheme.
    (proj.output_dir / 'cam_pisp_custom_1.json').write_text(EDITED)
    (proj.output_dir / 'cam_pisp_custom_index.json').write_text(
        json.dumps([{'id': 1, 'label': 'Warm demo', 'base_mtime': None}])
    )
    d = client.get('/projects/cam/tuning-data/pisp').get_json()
    assert d['variants'] == [{'id': 'warm-demo', 'label': 'Warm demo', 'stale': False}]
    assert not (proj.output_dir / 'cam_pisp_custom_1.json').exists()  # renamed to the slug
    assert (proj.output_dir / 'cam_pisp_custom_warm-demo.json').read_text() == EDITED


def test_orphan_file_surfaces(client, tmp_path):
    proj = _project_with_tuning(tmp_path)
    (proj.output_dir / 'cam_pisp_custom_hand-made.json').write_text(EDITED)  # no manifest entry
    d = client.get('/projects/cam/tuning-data/pisp').get_json()
    assert d['variants'] == [{'id': 'hand-made', 'label': 'hand made', 'stale': False}]


def test_manifest_entry_with_missing_file_dropped(client, tmp_path):
    proj = _project_with_tuning(tmp_path)
    _create(client, 'One')
    _create(client, 'Two')
    (proj.output_dir / 'cam_pisp_custom_one.json').unlink()  # delete out-of-band
    d = client.get('/projects/cam/tuning-data/pisp').get_json()
    assert [v['id'] for v in d['variants']] == ['two']


# --- staleness -----------------------------------------------------------


def test_variant_goes_stale_when_original_regenerated(client, tmp_path):
    proj = _project_with_tuning(tmp_path)
    _create(client, 'One')
    assert client.get('/projects/cam/tuning-data/pisp').get_json()['variants'][0]['stale'] is False
    # Regenerate the original (newer mtime) — the variant's diff base is now stale.
    gen = proj.output_dir / 'cam_pisp.json'
    os.utime(gen, (gen.stat().st_atime + 100, gen.stat().st_mtime + 100))
    assert client.get('/projects/cam/tuning-data/pisp').get_json()['variants'][0]['stale'] is True


# --- download ------------------------------------------------------------


def test_download_variant(client, tmp_path):
    _project_with_tuning(tmp_path)
    assert client.get('/projects/cam/download/custom/pisp').status_code == 404  # none yet
    _create(client, 'One')
    r = client.get('/projects/cam/download/custom/pisp?variant=one')
    assert r.status_code == 200 and r.data.decode() == EDITED
    assert client.get('/projects/cam/download/custom/pisp?variant=nope').status_code == 404


# --- live preview --------------------------------------------------------


def test_preview_test_custom_kind(client, tmp_path, monkeypatch):
    _project_with_tuning(tmp_path)
    calls = {}
    monkeypatch.setattr(app_module, 'platform_target', lambda: 'pisp')
    monkeypatch.setattr(
        app_module,
        'reload_shared_camera',
        lambda tuning_file=None, preview_max_width=1280: calls.update(tuning=tuning_file) or FakeCam(),
    )
    # No variants yet: a clear 400, and the camera is not reloaded.
    r = client.post('/projects/cam/preview-test', json={'kind': 'custom'})
    assert r.status_code == 400
    assert 'Tuning tab' in r.get_json()['error']

    # Two variants: selecting the *second* must load it, not fall back to the first.
    _create(client, 'AI denoise')
    _create(client, 'HW denoise')
    r = client.post('/projects/cam/preview-test', json={'kind': 'custom', 'variant': 'hw-denoise'})
    d = r.get_json()
    assert r.status_code == 200
    assert (d['kind'], d['tuning'], d['variant'], d['label']) == (
        'custom',
        'cam_pisp_custom_hw-denoise.json',
        'hw-denoise',
        'HW denoise',
    )
    assert calls['tuning'].endswith('cam_pisp_custom_hw-denoise.json')

    # An unknown/blank variant falls back to the first; the default kind loads the original.
    assert client.post('/projects/cam/preview-test', json={'kind': 'custom'}).get_json()['variant'] == 'ai-denoise'
    assert client.post('/projects/cam/preview-test').get_json()['tuning'] == 'cam_pisp.json'


# --- runner --------------------------------------------------------------


def test_run_stream_keeps_variants(tmp_path, monkeypatch):
    import ctt.core.runner as runner_mod

    proj = _project_with_tuning(tmp_path)
    variant = proj.output_dir / 'cam_pisp_custom_demo.json'
    variant.write_text(EDITED)
    monkeypatch.setattr(runner_mod, 'run_ctt_targets', lambda *a, **kw: None)

    lines = list(ctt_runner.run_ctt_stream(proj, ['pisp'], 'full', {}))
    assert variant.exists()  # variants persist across a successful re-run
    assert not any('Discarded' in line for line in lines)
    assert lines[-1] == 'CTT_EXIT 0'


def test_save_roundtrips_arbitrary_valid_json(client, tmp_path):
    _project_with_tuning(tmp_path)
    text = json.dumps({'algorithms': [{'rpi.black_level': {'black_level': 3200}}]}, indent=4)
    r = _create(client, 'Json test', text=text)
    assert r.status_code == 200
    assert r.get_json()['custom'] == text
