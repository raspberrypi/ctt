# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for ctt_server session model and in-process CTT execution.

import json

import pytest

from ctt_server import ctt_runner, naming, sessions


def test_session_add_and_count(tmp_path):
    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('imx-test')
    proj.add_capture(b'FAKEDNG', 'alsc', 5000)
    proj.add_capture(b'FAKEDNG', 'macbeth', 5858, lux=1344, label='d65')
    assert proj.counts() == {'macbeth': 1, 'alsc': 1, 'cac': 0}

    # Filenames are CTT-correct on disk.
    names = sorted(p.name for p in proj.path.glob('*.dng'))
    assert names == ['alsc_5000k_0.dng', 'd65_5858k_1344l.dng']

    # Reloading the project from disk restores capture metadata.
    reloaded = ws.get_project('imx-test')
    assert reloaded.counts() == {'macbeth': 1, 'alsc': 1, 'cac': 0}


def test_recapture_same_name_overwrites_not_duplicates(tmp_path):
    # Re-capturing a macbeth image at the same colour temp + lux reuses the
    # filename, so the metadata entry is replaced in place, not duplicated.
    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('imx-test')
    first = proj.add_capture(b'FIRST', 'macbeth', 5000, lux=1000)
    second = proj.add_capture(b'SECOND', 'macbeth', 5000, lux=1000)

    assert first.filename == second.filename  # same colour temp + lux ⇒ same name
    assert proj.counts()['macbeth'] == 1
    assert [c.filename for c in proj.captures] == [second.filename]
    # On-disk DNG reflects the latest capture.
    assert (proj.path / second.filename).read_bytes() == b'SECOND'


def test_load_heals_duplicate_filenames(tmp_path):
    # A project.json written by an older build may contain duplicate filenames;
    # loading collapses them to a single entry.
    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('imx-test')
    dup = {'filename': 'imx-test_5000k_1000l.dng', 'image_type': 'macbeth', 'colour_temp': 5000, 'lux': 1000}
    proj.sidecar.write_text(json.dumps({'captures': [dup, dict(dup)]}))

    reloaded = ws.get_project('imx-test')
    assert reloaded.counts()['macbeth'] == 1


def test_import_capture_auto_tags(tmp_path):
    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('imx-test')
    mac = proj.import_capture('d65_5000k_800l.dng', b'MAC')
    alsc = proj.import_capture('alsc_3000k_0.dng', b'ALSC')

    assert (mac.image_type, mac.colour_temp, mac.lux, mac.label) == ('macbeth', 5000, 800, 'd65')
    assert (alsc.image_type, alsc.colour_temp, alsc.lux) == ('alsc', 3000, None)
    # Stored under the original (auto-tagged) filename, bytes preserved.
    assert (proj.path / 'd65_5000k_800l.dng').read_bytes() == b'MAC'
    assert proj.counts() == {'macbeth': 1, 'alsc': 1, 'cac': 0}


def test_import_capture_normalises_extension_case(tmp_path):
    proj = sessions.Workspace(tmp_path).create_project('p')
    cap = proj.import_capture('alsc_5000k_1.DNG', b'X')
    assert cap.filename == 'alsc_5000k_1.dng'


def test_import_capture_rejects_untagged(tmp_path):
    proj = sessions.Workspace(tmp_path).create_project('p')
    with pytest.raises(naming.NamingError):
        proj.import_capture('random.dng', b'X')
    assert proj.counts()['macbeth'] == 0


def test_upload_route_auto_tags_and_skips(tmp_path):
    from io import BytesIO

    from ctt_server.app import create_app

    sessions.Workspace(tmp_path).create_project('cam')
    client = create_app(str(tmp_path)).test_client()
    data = {
        'files': [
            (BytesIO(b'MAC'), 'd65_5000k_800l.dng'),
            (BytesIO(b'BAD'), 'oops.dng'),
        ]
    }
    r = client.post('/projects/cam/upload', data=data, content_type='multipart/form-data')
    assert r.status_code == 200
    body = r.get_json()
    assert [a['filename'] for a in body['added']] == ['d65_5000k_800l.dng']
    assert body['added'][0]['colour_temp'] == 5000
    assert [s['filename'] for s in body['skipped']] == ['oops.dng']
    assert body['counts']['macbeth'] == 1


def test_session_delete_capture(tmp_path):
    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('p')
    proj.add_capture(b'X', 'alsc', 3000)
    proj.delete_capture('alsc_3000k_0.dng')
    assert proj.counts()['alsc'] == 0
    assert not list(proj.path.glob('*.dng'))


def test_add_capture_writes_and_omits_jpeg(tmp_path):
    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('p')
    proj.add_capture(b'DNG', 'alsc', 5000, jpeg_bytes=b'JPG')
    assert (proj.path / 'alsc_5000k_0.jpg').read_bytes() == b'JPG'
    assert proj.has_saved_jpeg('alsc_5000k_0.dng')
    # No JPEG given -> no sidecar written.
    proj.add_capture(b'DNG', 'alsc', 6000)
    assert not (proj.path / 'alsc_6000k_1.jpg').exists()
    assert not proj.has_saved_jpeg('alsc_6000k_1.dng')


def test_recapture_overwrites_jpeg(tmp_path):
    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('p')
    proj.add_capture(b'D1', 'macbeth', 5000, lux=1000, jpeg_bytes=b'J1')
    cap = proj.add_capture(b'D2', 'macbeth', 5000, lux=1000, jpeg_bytes=b'J2')
    assert (proj.path / cap.filename).with_suffix('.jpg').read_bytes() == b'J2'
    assert len(list(proj.path.glob('*.jpg'))) == 1


def test_delete_capture_removes_jpeg(tmp_path):
    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('p')
    proj.add_capture(b'X', 'alsc', 3000, jpeg_bytes=b'JPG')
    proj.delete_capture('alsc_3000k_0.dng')
    assert not list(proj.path.glob('*.jpg'))
    assert not list(proj.path.glob('*.dng'))


def test_serialise_captures_jpeg_source(tmp_path):
    from ctt_server.app import _RAWPY, _serialise_captures

    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('p')
    proj.add_capture(b'DNG', 'alsc', 5000, jpeg_bytes=b'JPG')  # has a saved JPEG
    proj.import_capture('alsc_3000k_0.dng', b'X')  # DNG only
    by_name = {c['filename']: c for c in _serialise_captures(proj)}
    assert by_name['alsc_5000k_0.dng']['jpeg'] == 'saved'
    assert by_name['alsc_3000k_0.dng']['jpeg'] == ('dng' if _RAWPY else None)


def test_capture_jpeg_route(tmp_path):
    from ctt_server.app import _RAWPY, create_app

    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('cam')
    proj.add_capture(b'DNG', 'alsc', 5000, jpeg_bytes=b'JPGDATA')
    client = create_app(str(tmp_path)).test_client()

    # Saved sibling JPEG is served verbatim.
    r = client.get('/projects/cam/captures/alsc_5000k_0.dng/jpeg')
    assert r.status_code == 200
    assert r.mimetype == 'image/jpeg'
    assert r.data == b'JPGDATA'

    # A non-.dng name is rejected.
    assert client.get('/projects/cam/captures/foo.txt/jpeg').status_code == 404

    # DNG-only capture: 404 without rawpy; with rawpy the develop fails on junk bytes -> 500.
    proj.import_capture('alsc_3000k_0.dng', b'NOTAREALDNG')
    r = client.get('/projects/cam/captures/alsc_3000k_0.dng/jpeg')
    assert r.status_code == (500 if _RAWPY else 404)


def test_output_files_reports_mtime(tmp_path):
    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('cam')
    proj.output_dir.mkdir(parents=True, exist_ok=True)
    (proj.output_dir / 'cam_pisp.json').write_text('{}')
    out = ctt_runner.output_files(proj, ['pisp', 'vc4'])
    assert out['pisp']['json'] is not None
    assert isinstance(out['pisp']['mtime'], float)  # present for the existing target
    assert out['vc4']['json'] is None and out['vc4']['mtime'] is None  # absent target


def test_build_config_shape():
    cfg = ctt_runner.build_config({'awb': {'greyworld': True}, 'alsc': {'luminance_strength': 0.5}, 'blacklevel': 64})
    assert cfg['awb']['greyworld'] == 1
    assert cfg['alsc']['luminance_strength'] == 0.5
    assert cfg['blacklevel'] == 64
    assert cfg['macbeth']['show'] == 0  # never pop interactive windows
    assert cfg['ccm']['matrix_selection'] == 'average'  # default


def test_build_config_ccm_options():
    cfg = ctt_runner.build_config({'ccm': {'matrix_selection': 'patches', 'test_patches': ['1', '5', '9']}})
    assert cfg['ccm']['matrix_selection'] == 'patches'
    assert cfg['ccm']['test_patches'] == [1, 5, 9]
    # An unknown selection falls back to the safe default.
    assert ctt_runner.build_config({'ccm': {'matrix_selection': 'bogus'}})['ccm']['matrix_selection'] == 'average'


def test_run_stream_rejects_bad_target(tmp_path):
    # Early validation: no CTT import, no thread — just a clear error + exit code.
    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('p')
    lines = list(ctt_runner.run_ctt_stream(proj, ['bogus'], 'full', {}))
    assert any('no valid target' in line for line in lines)
    assert lines[-1] == 'CTT_EXIT 2'


def test_run_stream_in_process_empty_project(tmp_path):
    # Runs CTT in-process (no subprocess) on a project with no DNGs: CTT logs that
    # no usable images were found, which we capture via the logging handler, and
    # the stream ends cleanly. Exercises the full in-process capture path.
    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('empty')
    lines = list(ctt_runner.run_ctt_stream(proj, ['pisp'], 'full', {}))
    assert any('Loading images from' in line for line in lines)
    assert any('No usable' in line for line in lines)
    assert lines[-1] == 'CTT_EXIT 0'
