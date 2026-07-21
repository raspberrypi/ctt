# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the characterisation routes and orchestrator: SSE analysis stream,
# results.json persistence, staleness and locking. DNG loading is stubbed with
# synthetic FrameSets, so these run without real captures.

import json
from pathlib import Path

import numpy as np
import pytest

from ctt.characterisation import FrameSet
from ctt.characterisation.discover import CaptureGroup
from ctt_server import characterise, sessions

BLACK_16 = 3840.0


def _synth_frameset(n_frames, signal_e, dn_per_e=1.5, read_noise_dn=15.0, exposure_us=10_000, gain=1.0, seed=0):
    rng = np.random.default_rng(seed)
    frames = [
        np.clip(
            BLACK_16 + dn_per_e * rng.poisson(signal_e, (48, 48)) + rng.normal(0, read_noise_dn, (48, 48)), 0, 65535
        )
        for _ in range(n_frames)
    ]
    return FrameSet(frames=frames, exposure_us=exposure_us, gain=gain, blacklevel_16=BLACK_16, sigbits=16, channel='Gr')


def _group(project_dir: Path, label, kind, n, exposure_us, gain=1.0, colour_temp=None):
    paths = []
    for i in range(n):
        p = project_dir / f'{label}_{i}.dng'
        p.write_bytes(b'DNG')
        paths.append(p)
    return CaptureGroup(
        label=label,
        kind=kind,
        paths=paths,
        exposure_us=exposure_us,
        gain=gain,
        width=1920,
        height=1080,
        sigbits=16,
        blacklevel=3840,
        colour_temp=colour_temp,
    )


@pytest.fixture
def project(tmp_path):
    ws = sessions.Workspace(tmp_path)
    proj = ws.create_project('imx662')
    proj.path.mkdir(parents=True, exist_ok=True)
    return proj


@pytest.fixture
def client(tmp_path):
    from ctt_server.app import create_app

    return create_app(str(tmp_path)).test_client()


@pytest.fixture
def stubbed_pipeline(project, monkeypatch):
    """Stub discovery + DNG loading with synthetic groups and framesets."""
    groups = [
        _group(project.path, 'dark', 'dark', 8, 33333),
        _group(project.path, 'alsc_3500k', 'flat', 8, 133, gain=1.07, colour_temp=3500),
        _group(project.path, 'alsc_6000k', 'flat', 5, 250, gain=1.07, colour_temp=6000),
    ]
    monkeypatch.setattr(characterise, 'scan_project', lambda directory, excluded=frozenset(): groups)

    def fake_dark_analysis(group, roi_fraction):
        fs = _synth_frameset(len(group.paths), 0.0, exposure_us=group.exposure_us, seed=1)
        return {
            'available': True,
            'label': group.label,
            'pedestal': {'black_level_16': 3843.0, 'metadata_black_level_16': BLACK_16, 'frames': []},
            'read_noise_dn': 15.2,
            'read_noise_e': None,
            'dsnu_dn': 2.1,
            'dsnu_e': None,
            'exposure_us': group.exposure_us,
            'gain': group.gain,
            'n_frames': len(group.paths),
            'channel': 'Gr',
        }, fs

    monkeypatch.setattr(characterise, '_dark_analysis', fake_dark_analysis)

    def fake_frameset(paths, roi_fraction=0.5):
        signal = 4000.0 if '3500' in paths[0] else 9000.0
        return _synth_frameset(len(paths), signal, gain=1.07, exposure_us=133, seed=len(paths))

    monkeypatch.setattr(characterise, 'frameset_from_dngs', fake_frameset)
    return groups


def _stream_lines(client, name='imx662'):
    r = client.get(f'/projects/{name}/characterisation/analyse/stream')
    assert r.status_code == 200
    return [json.loads(line[6:]) for line in r.data.decode().splitlines() if line.startswith('data: ')]


def test_data_endpoint_empty_project(client, project, monkeypatch):
    monkeypatch.setattr(characterise, 'scan_project', lambda directory, excluded=frozenset(): [])
    d = client.get('/projects/imx662/characterisation/data').get_json()
    assert d['results'] is None and d['groups'] == [] and d['has_results'] is False and d['stale'] is False


def test_analyse_stream_persists_results(client, project, stubbed_pipeline):
    lines = _stream_lines(client)
    assert lines[-1] == 'CHAR_EXIT 0'
    assert any('Found 3 capture groups' in ln for ln in lines)
    assert any('black level 3843' in ln for ln in lines)

    results = json.loads(characterise.results_path(project).read_text())
    assert results['version'] == 1
    assert results['dark']['available'] and results['dark']['read_noise_dn'] == 15.2
    # Two flat points at one gain, non-clipped: an (unreliable) 2-point fit.
    assert len(results['ptc']['points']) == 2
    assert len(results['ptc']['fits']) == 1 and not results['ptc']['fits'][0]['reliable']
    assert 'run the live sweep' in results['ptc']['unavailable_reason']
    # e- values stay null without a reliable conversion gain.
    assert results['dark']['read_noise_e'] is None
    assert results['prnu']['available'] and len(results['prnu']['groups']) == 2


def test_data_reports_staleness_after_new_capture(client, project, stubbed_pipeline):
    _stream_lines(client)
    d = client.get('/projects/imx662/characterisation/data').get_json()
    assert d['has_results'] and d['stale'] is False

    # A new/changed capture invalidates the recorded input fingerprint.
    characterise._scan_cache.clear()
    import os

    dng = project.path / 'dark_0.dng'
    os.utime(dng, (dng.stat().st_atime, dng.stat().st_mtime + 10))
    d = client.get('/projects/imx662/characterisation/data').get_json()
    assert d['stale'] is True


def test_analyse_refuses_while_calibration_runs(client, project, monkeypatch):
    from ctt_server import ctt_runner

    monkeypatch.setattr(ctt_runner, 'is_running', lambda: True)
    lines = _stream_lines(client)
    assert lines == ['ERROR: a calibration is running; wait for it to finish', 'CHAR_EXIT 2']


def test_analyse_refuses_concurrent_analysis(client, project):
    assert characterise._char_lock.acquire(blocking=False)
    try:
        lines = _stream_lines(client)
        assert lines == ['ERROR: a characterisation analysis is already running', 'CHAR_EXIT 2']
    finally:
        characterise._char_lock.release()


def test_analyse_stream_no_captures(client, project, monkeypatch):
    monkeypatch.setattr(characterise, 'scan_project', lambda directory, excluded=frozenset(): [])
    lines = _stream_lines(client)
    assert lines[-1] == 'CHAR_EXIT 1'
    assert any('no usable captures' in ln for ln in lines)


def test_page_renders(client, project):
    r = client.get('/projects/imx662/characterisation')
    assert r.status_code == 200
    assert b'characterisationApp' in r.data


def test_carry_live_sweep_preserves_sweep_metrics():
    # A fresh offline analysis must not drop a prior live sweep's derived metrics.
    prev = {
        'ptc': {
            'points': [
                {
                    'mean_dn': 100.0,
                    'var_dn2': 50.0,
                    'exposure_us': 100,
                    'gain': 1.0,
                    'n_frames': 8,
                    'clipped': False,
                    'label': 's',
                    'source': 'live',
                },
                {
                    'mean_dn': 5000.0,
                    'var_dn2': 2500.0,
                    'exposure_us': 5000,
                    'gain': 1.0,
                    'n_frames': 8,
                    'clipped': False,
                    'label': 's',
                    'source': 'live',
                },
            ],
            'fits': [{'gain': 1.0, 'reliable': True}],
        },
        'linearity': {'available': True, 'r2': 0.999},
        'full_well': {'available': True, 'full_well_e': 12000},
        'dynamic_range': {'available': True, 'db': 60.0},
    }
    results = {'ptc': {'points': [], 'fits': [], 'unavailable_reason': 'needs a sweep'}, 'dark': {'available': False}}
    characterise._carry_live_sweep(results, prev)
    assert results['linearity'] == prev['linearity']
    assert results['full_well'] == prev['full_well']
    assert results['dynamic_range'] == prev['dynamic_range']
    assert [p['source'] for p in results['ptc']['points']] == ['live', 'live']  # live points re-added


def test_carry_live_sweep_noop_without_previous():
    results = {'ptc': {'points': [], 'fits': []}, 'dark': {'available': False}}
    characterise._carry_live_sweep(results, None)
    assert results['ptc']['points'] == [] and 'linearity' not in results
