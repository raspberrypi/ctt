# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for ctt_server session model and in-process CTT execution.

import json

from ctt_server import ctt_runner, sessions


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


def test_session_delete_capture(tmp_path):
    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('p')
    proj.add_capture(b'X', 'alsc', 3000)
    proj.delete_capture('alsc_3000k_0.dng')
    assert proj.counts()['alsc'] == 0
    assert not list(proj.path.glob('*.dng'))


def test_build_config_shape():
    cfg = ctt_runner.build_config({'awb': {'greyworld': True}, 'alsc': {'luminance_strength': 0.5}, 'blacklevel': 64})
    assert cfg['awb']['greyworld'] == 1
    assert cfg['alsc']['luminance_strength'] == 0.5
    assert cfg['blacklevel'] == 64
    assert cfg['macbeth']['show'] == 0  # never pop interactive windows


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
