# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the gap-driven auto-characterise cycle (apps/ctt_server/auto_characterise.py)
# and its routes. Hardware-free: the generator runs against scripted fakes, and the
# characterise sub-streams (quick_scan / read_results / sweep_stream / analyse_stream) are
# monkeypatched so no pixel data or camera is needed.

import json

import pytest

from ctt_server import app as app_module
from ctt_server import auto_characterise, characterise, sessions
from ctt_server.auto_capture import StabiliseConfig
from ctt_server.camera import CameraError
from devices import LightboxError, LightmeterError, Measurement

FAST = StabiliseConfig(min_wait_s=0.0, sample_interval_s=0.0, timeout_s=1.0)


class ScriptedMeter:
    def __init__(self, script=((1000.0, 5000.0),)):
        self.script = list(script)

    def measure(self):
        entry = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if entry is None:
            raise LightmeterError('scripted failure')
        return Measurement(illuminance_lux=entry[0], cct=entry[1])


class FakeBox:
    illuminants = {4: 'D65', 7: 'Halogen400'}

    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def set_illuminant(self, illuminant, percent=None):
        self.calls.append(('set_illuminant', illuminant, percent))
        if illuminant in self.fail_on:
            raise LightboxError(f'lamp {illuminant} failed')

    def off(self):
        self.calls.append(('off',))

    def info(self):
        return {'channel': 7, 'illuminant_temps': {7: 2856}}


class FakeCam:
    def __init__(self, fail=False):
        self.fail = fail
        self.controls = {'exposure': 5000, 'gain': 1.0, 'auto_exposure': True, 'fps': 30, 'exposure_min': 100}
        self.bursts = []

    def get_controls(self):
        return dict(self.controls)

    def set_controls(self, c):
        self.controls.update({k: v for k, v in c.items() if v is not None})
        return dict(self.controls)

    def capture_burst(self, frames, quality=95):
        if self.fail:
            raise CameraError('camera gone')
        gain = self.controls.get('gain')
        self.bursts.append((frames, gain))
        return [(b'DNG', b'JPG', {'exposure': self.controls.get('exposure'), 'gain': gain})] * frames


@pytest.fixture
def project(tmp_path):
    return sessions.Workspace(tmp_path).create_project('cam')


@pytest.fixture(autouse=True)
def stub_characterise(monkeypatch):
    """Default: empty project, no results; sweep/analyse succeed and record calls."""
    calls = {'sweep': [], 'analyse': 0}

    monkeypatch.setattr(characterise, 'quick_scan', lambda proj: {'groups': [], 'has_results': False, 'stale': False})
    monkeypatch.setattr(characterise, 'read_results', lambda proj: None)
    monkeypatch.setattr(characterise, 'flat_exposure', lambda cam, gain, roi_fraction=0.5, target=0.5: 1000)

    def fake_sweep(proj, cam, gains, points_per_gain=10, frames=8):
        calls['sweep'].append(list(gains))
        yield f'sweep {gains}'
        yield 'CHAR_EXIT 0'

    def fake_analyse(proj):
        calls['analyse'] += 1
        yield 'analysed'
        yield 'CHAR_EXIT 0'

    monkeypatch.setattr(characterise, 'sweep_stream', fake_sweep)
    monkeypatch.setattr(characterise, 'analyse_stream', fake_analyse)
    return calls


def drain(gen):
    return list(gen)


def events(stream, kind):
    return [e for e in stream if e['event'] == kind]


def run(project, box, meter, cam, **kw):
    kw.setdefault('cfg', FAST)
    return drain(auto_characterise.run_auto_characterise_stream(project, cam, box, meter, **kw))


# --- gap analysis --------------------------------------------------------------
def test_gaps_reuse_and_capture(monkeypatch, project):
    monkeypatch.setattr(
        characterise,
        'quick_scan',
        lambda proj: {'groups': [{'kind': 'dark', 'gain': 1.0}], 'has_results': True, 'stale': False},
    )
    monkeypatch.setattr(characterise, 'read_results', lambda proj: {'ptc': {'fits': [{'gain': 1.0, 'reliable': True}]}})
    plan = auto_characterise._gaps(project, [1.0, 4.0], 0.05)
    g1, g4 = plan
    assert g1 == {'gain': 1.0, 'darks': 'reuse', 'flats': 'capture', 'sweep': 'reuse'}
    assert g4 == {'gain': 4.0, 'darks': 'capture', 'flats': 'capture', 'sweep': 'sweep'}


def test_gaps_stale_results_force_resweep(monkeypatch, project):
    monkeypatch.setattr(characterise, 'quick_scan', lambda proj: {'groups': [], 'has_results': True, 'stale': True})
    monkeypatch.setattr(characterise, 'read_results', lambda proj: {'ptc': {'fits': [{'gain': 1.0, 'reliable': True}]}})
    (g1,) = auto_characterise._gaps(project, [1.0], 0.05)
    assert g1['sweep'] == 'sweep'  # stale invalidates the reused fit


# --- full cycle ----------------------------------------------------------------
def _run_with_lenscap(gen):
    """Drain a generator, releasing the lens-cap pause when it waits."""
    stream = []
    for event in gen:
        stream.append(event)
        if event['event'] == 'waiting_user':
            auto_characterise.request_continue()
    return stream


def test_full_cycle_captures_all_when_project_empty(project, stub_characterise):
    box, meter, cam = FakeBox(), ScriptedMeter(), FakeCam()
    gen = auto_characterise.run_auto_characterise_stream(
        project, cam, box, meter, [1.0], 'Halogen400', intensity=80.0, cfg=FAST
    )
    stream = _run_with_lenscap(gen)
    done = stream[-1]
    assert done['event'] == 'done' and done['ok'] is True
    assert stub_characterise['sweep'] == [[1.0]] and stub_characterise['analyse'] == 1
    assert ('set_illuminant', 'Halogen400', 80.0) in box.calls and ('off',) in box.calls
    kinds = {c.image_type for c in project.captures}
    assert kinds == {'alsc', 'dark'}
    assert not auto_characterise.is_running()


def test_reused_inputs_are_not_recaptured(project, monkeypatch, stub_characterise):
    # Darks + a reliable sweep already exist at gain 1.0; only the flat is missing.
    monkeypatch.setattr(
        characterise,
        'quick_scan',
        lambda proj: {'groups': [{'kind': 'dark', 'gain': 1.0}], 'has_results': True, 'stale': False},
    )
    monkeypatch.setattr(characterise, 'read_results', lambda proj: {'ptc': {'fits': [{'gain': 1.0, 'reliable': True}]}})
    box, cam = FakeBox(), FakeCam()
    stream = _run_with_lenscap(
        auto_characterise.run_auto_characterise_stream(project, cam, box, ScriptedMeter(), [1.0], 'D65', cfg=FAST)
    )
    assert stub_characterise['sweep'] == []  # sweep reused
    assert ('off',) not in box.calls  # no dark phase, box left as-is
    assert {c.image_type for c in project.captures} == {'alsc'}  # only the flat captured, no darks
    assert 'dark g1' in stream[-1]['reused'] and 'sweep g1' in stream[-1]['reused']


def test_operating_point_captures_dark_at_requested_gain(project, monkeypatch, stub_characterise):
    # Darks exist at gain 1.0; requesting gain 4.0 must capture a fresh dark at 4.0.
    monkeypatch.setattr(
        characterise,
        'quick_scan',
        lambda proj: {'groups': [{'kind': 'dark', 'gain': 1.0}], 'has_results': False, 'stale': False},
    )
    cam = FakeCam()
    _run_with_lenscap(
        auto_characterise.run_auto_characterise_stream(
            project, cam, FakeBox(), ScriptedMeter(), [4.0], 'D65', include_flats=False, include_sweep=False, cfg=FAST
        )
    )
    assert cam.bursts and all(g == 4.0 for _, g in cam.bursts)  # captured at the requested gain


def test_sweep_filtered_to_missing_gains(project, monkeypatch, stub_characterise):
    # gain 1.0 already has a reliable fit; requesting [1.0, 4.0] sweeps only 4.0.
    monkeypatch.setattr(characterise, 'read_results', lambda proj: {'ptc': {'fits': [{'gain': 1.0, 'reliable': True}]}})
    monkeypatch.setattr(characterise, 'quick_scan', lambda proj: {'groups': [], 'has_results': True, 'stale': False})
    run(
        project,
        FakeBox(),
        ScriptedMeter(),
        FakeCam(),
        gains=[1.0, 4.0],
        lamp='D65',
        include_darks=False,
        include_flats=False,
    )
    assert stub_characterise['sweep'] == [[4.0]]


def test_sweep_skipped_when_all_reused(project, monkeypatch, stub_characterise):
    monkeypatch.setattr(characterise, 'read_results', lambda proj: {'ptc': {'fits': [{'gain': 1.0, 'reliable': True}]}})
    monkeypatch.setattr(characterise, 'quick_scan', lambda proj: {'groups': [], 'has_results': True, 'stale': False})
    stream = run(
        project,
        FakeBox(),
        ScriptedMeter(),
        FakeCam(),
        gains=[1.0],
        lamp='D65',
        include_darks=False,
        include_flats=False,
    )
    assert stub_characterise['sweep'] == [] and stub_characterise['analyse'] == 0  # nothing to do
    assert stream[-1]['ok'] is True


# --- control ------------------------------------------------------------------
def test_cancel_turns_box_off_and_releases(project, stub_characterise):
    box = FakeBox()
    gen = auto_characterise.run_auto_characterise_stream(
        project, FakeCam(), box, ScriptedMeter(), [1.0], 'D65', include_darks=False, include_flats=False, cfg=FAST
    )
    stream = []
    for event in gen:
        stream.append(event)
        if event['event'] == 'plan':
            auto_characterise.request_cancel()
    assert stream[-1]['event'] == 'done' and stream[-1]['cancelled'] is True
    assert ('off',) in box.calls and not auto_characterise.is_running()


def test_camera_failure_is_fatal(project, stub_characterise):
    box = FakeBox()
    stream = _run_with_lenscap(
        auto_characterise.run_auto_characterise_stream(
            project,
            FakeCam(fail=True),
            box,
            ScriptedMeter(),
            [1.0],
            'D65',
            include_sweep=False,
            include_darks=False,
            cfg=FAST,
        )
    )
    assert stream[-1]['event'] == 'error' and 'camera failure' in stream[-1]['error']
    assert not auto_characterise.is_running() and ('off',) in box.calls


def test_second_cycle_rejected_while_running(project, stub_characterise):
    gen = auto_characterise.run_auto_characterise_stream(
        project, FakeCam(), FakeBox(), ScriptedMeter(), [1.0], 'D65', include_darks=False, include_flats=False, cfg=FAST
    )
    next(gen)  # holds the lock
    second = run(project, FakeBox(), ScriptedMeter(), FakeCam(), gains=[1.0], lamp='D65')
    assert second[0]['event'] == 'error' and 'already running' in second[0]['error']
    gen.close()


def test_continue_and_cancel_when_idle(stub_characterise):
    assert auto_characterise.request_continue() is False
    assert auto_characterise.request_cancel() is False


# --- routes -------------------------------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    sessions.Workspace(tmp_path).create_project('cam')
    monkeypatch.setattr(app_module, 'get_shared_camera', lambda: FakeCam())
    monkeypatch.setattr(app_module, 'get_shared_lightbox', lambda: FakeBox())
    monkeypatch.setattr(app_module, 'get_shared_lightmeter', lambda: ScriptedMeter())
    monkeypatch.setattr(
        app_module.auto_characterise,
        'run_auto_characterise_stream',
        lambda *a, **kw: iter([{'event': 'start'}, {'event': 'done', 'ok': True}]),
    )
    return app_module.create_app(tmp_path).test_client()


def sse_events(response):
    return [json.loads(line[len('data: ') :]) for line in response.get_data(as_text=True).splitlines() if line]


def test_stream_route_emits_sse(client):
    r = client.get('/projects/cam/auto-characterise/stream?gains=1,4&lamp=D65&intensity=80')
    assert r.status_code == 200 and r.mimetype == 'text/event-stream'
    stream = sse_events(r)
    assert stream[0]['event'] == 'start' and stream[-1]['event'] == 'done'


def test_stream_route_requires_lamp(client):
    stream = sse_events(client.get('/projects/cam/auto-characterise/stream?gains=1'))
    assert stream[0]['event'] == 'error' and 'lamp' in stream[0]['error'].lower()


def test_stream_route_reports_missing_devices(client, monkeypatch):
    monkeypatch.setattr(app_module, 'get_shared_lightmeter', lambda: (_ for _ in ()).throw(LightmeterError('no meter')))
    stream = sse_events(client.get('/projects/cam/auto-characterise/stream?gains=1&lamp=D65'))
    assert stream[0]['event'] == 'error' and 'meter' in stream[0]['error'].lower()


def test_continue_cancel_routes_409_when_idle(client):
    assert client.post('/projects/cam/auto-characterise/continue').status_code == 409
    assert client.post('/projects/cam/auto-characterise/cancel').status_code == 409
