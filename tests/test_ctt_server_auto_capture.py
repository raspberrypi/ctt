# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the auto-capture cycle (apps/ctt_server/auto_capture.py) and its routes.
# Hardware-free: the generator is driven directly against scripted fakes, and the
# routes are exercised with the shared-device accessors monkeypatched.

import json
import re

import pytest

from ctt_server import app as app_module
from ctt_server import auto_capture, sessions
from ctt_server.auto_capture import AdjustConfig, LampStep, StabiliseConfig, parse_lamps, run_auto_capture_stream
from ctt_server.camera import CameraError
from devices import LightboxError, LightmeterError, Measurement

# Instant stabilisation for tests: no settle wait, no inter-sample pacing, 1 s timeout.
FAST = StabiliseConfig(min_wait_s=0.0, sample_interval_s=0.0, timeout_s=1.0)


class ScriptedMeter:
    """measure() pops scripted (lux, cct) readings, repeating the last one.

    An entry of None raises LightmeterError instead.
    """

    def __init__(self, script):
        self.script = list(script)

    def measure(self):
        entry = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if entry is None:
            raise LightmeterError('scripted failure')
        lux, cct = entry
        return Measurement(illuminance_lux=lux, cct=cct, in_range=lux >= 1.0)


def steady_meter(lux=1000.0, cct=5000.0):
    return ScriptedMeter([(lux, cct)])


class FakeBox:
    """Records illuminant switches; optionally fails on given illuminants."""

    illuminants = {1: 'F12', 4: 'D65'}

    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def set_illuminant(self, illuminant, percent=None):
        self.calls.append(('set_illuminant', illuminant, percent))
        if illuminant in self.fail_on:
            raise LightboxError(f'lamp {illuminant} failed')

    def off(self):
        self.calls.append(('off',))


class FakeCam:
    def __init__(self, fail=False):
        self.fail = fail
        self.bursts = []

    def capture_burst(self, frames, quality=95):
        if self.fail:
            raise CameraError('camera gone')
        self.bursts.append(frames)
        return [(b'DNG', b'JPG', {'exposure': 1})] * frames


class InspectCam(FakeCam):
    """FakeCam with a scripted detect_chart()/histogram() for the adjust loop.

    Each chart dict may set 'found'/'saturated'/'confidence'; the last entry repeats.
    """

    def __init__(self, charts, clip=0.0, fail=False):
        super().__init__(fail=fail)
        self.charts = list(charts)
        self.clip = clip
        self.inspects = 0

    def detect_chart(self):
        self.inspects += 1
        c = self.charts.pop(0) if len(self.charts) > 1 else self.charts[0]
        return {
            'found': c.get('found', True),
            'saturated': c.get('saturated', False),
            'confidence': c.get('confidence', 0.9),
            'corners': None,
            'small': False,
        }

    def histogram(self, bins=64):
        return {'clipping': {'r': self.clip, 'g': self.clip, 'b': self.clip}, 'bins': bins}


@pytest.fixture
def project(tmp_path):
    return sessions.Workspace(tmp_path).create_project('cam')


def drain(gen):
    return list(gen)


def events(stream, kind):
    return [e for e in stream if e['event'] == kind]


# --- lamp parsing --------------------------------------------------------------
def test_parse_lamps():
    assert parse_lamps('D65:100,F12:80.5') == [LampStep('D65', 100.0), LampStep('F12', 80.5)]
    assert parse_lamps('D65') == [LampStep('D65', None)]  # no intensity -> lamp default
    with pytest.raises(ValueError, match='no lamps'):
        parse_lamps('')
    with pytest.raises(ValueError, match='0-100'):
        parse_lamps('D65:150')
    with pytest.raises(ValueError, match='malformed'):
        parse_lamps(':50')


# --- generator: happy path ------------------------------------------------------
def test_cycle_captures_each_lamp_with_measured_tags(project):
    meter = ScriptedMeter([(900.0, 4800.0), (990.0, 4990.0), (1000.0, 5000.0), (1001.0, 5001.0), (999.0, 4999.0)])
    box, cam = FakeBox(), FakeCam()
    stream = drain(
        run_auto_capture_stream(
            project, cam, box, meter, [LampStep('F12', 80.0), LampStep('D65', 100.0)], frames=2, cfg=FAST
        )
    )
    kinds = [e['event'] for e in stream]
    assert kinds[0] == 'start' and kinds[-1] == 'done'
    assert kinds.count('lamp_start') == kinds.count('lamp_done') == 2
    assert kinds.count('stable') == 2 and 'stabilise_timeout' not in kinds
    done = stream[-1]
    assert done['ok'] is True and done['captured'] == 2 and done['failed'] == []
    # Lightbox switched per lamp with the requested intensity; left lit at the end.
    assert ('set_illuminant', 'F12', 80.0) in box.calls and ('set_illuminant', 'D65', 100.0) in box.calls
    assert ('off',) not in box.calls
    # Captures are tagged from the measured (rounded) reading, with it recorded.
    caps = project.captures
    assert len(caps) == 4 and all(c.image_type == 'macbeth' for c in caps)
    assert all(c.colour_temp == round(c.lightmeter['cct']) for c in caps)
    assert all(c.lux == round(c.lightmeter['illuminance_lux']) for c in caps)
    assert all(re.search(r'_\d+\.dng$', c.filename) for c in caps)  # always indexed


def test_captured_events_reuse_manual_response_shape(project):
    stream = drain(
        run_auto_capture_stream(project, FakeCam(), FakeBox(), steady_meter(), [LampStep('D65')], frames=1, cfg=FAST)
    )
    (captured,) = events(stream, 'captured')
    entry = captured['added'][0]
    assert set(entry) == {'filename', 'image_type', 'colour_temp', 'lux', 'label', 'valid', 'jpeg', 'lightmeter'}
    assert captured['counts']['macbeth'] == 1


# --- generator: stabilisation ----------------------------------------------------
def test_timeout_proceeds_with_warning(project):
    # Alternating readings never satisfy the window; the 1 s timeout still captures.
    meter = ScriptedMeter([(100.0, 3000.0), (200.0, 6000.0)] * 50)
    stream = drain(
        run_auto_capture_stream(
            project,
            FakeCam(),
            FakeBox(),
            meter,
            [LampStep('D65')],
            frames=1,
            cfg=StabiliseConfig(min_wait_s=0.0, sample_interval_s=0.0, timeout_s=0.0),
        )
    )
    assert events(stream, 'stabilise_timeout')
    done = stream[-1]
    assert done['ok'] is True and done['warnings'] and 'did not stabilise' in done['warnings'][0]
    assert len(project.captures) == 1


def test_persistent_meter_failure_fails_the_lamp_and_continues(project):
    # First lamp: meter always fails; second lamp: steady readings.
    fail_then_ok = ScriptedMeter([None, None, None, (1000.0, 5000.0)])
    box = FakeBox()
    stream = drain(
        run_auto_capture_stream(
            project, FakeCam(), box, fail_then_ok, [LampStep('F12'), LampStep('D65')], frames=1, cfg=FAST
        )
    )
    (lamp_error,) = events(stream, 'lamp_error')
    assert lamp_error['illuminant'] == 'F12'
    done = stream[-1]
    assert done['failed'] == ['F12'] and done['ok'] is False and done['captured'] == 1


def test_under_range_readings_fail_the_lamp(project):
    # A persistently under-range (below-limit) lamp fails with a clear message and is
    # never tagged/captured.
    dark_then_ok = ScriptedMeter([(-100.0, 0.0), (-100.0, 0.0), (-100.0, 0.0), (1000.0, 5000.0)])
    stream = drain(
        run_auto_capture_stream(
            project, FakeCam(), FakeBox(), dark_then_ok, [LampStep('F12'), LampStep('D65')], frames=1, cfg=FAST
        )
    )
    (lamp_error,) = events(stream, 'lamp_error')
    assert lamp_error['illuminant'] == 'F12' and 'range' in lamp_error['error'].lower()
    assert stream[-1]['failed'] == ['F12'] and stream[-1]['captured'] == 1


# --- generator: chart-aware light adjustment ---------------------------------------
def _set_percents(box):
    return [c[2] for c in box.calls if c[0] == 'set_illuminant']


def test_adjust_lowers_intensity_when_chart_saturated(project):
    cam = InspectCam([{'saturated': True}, {'saturated': True}, {'found': True, 'saturated': False}])
    box = FakeBox()
    stream = drain(
        run_auto_capture_stream(project, cam, box, steady_meter(), [LampStep('D65', 100.0)], frames=1, cfg=FAST)
    )
    adjusts = events(stream, 'adjust')
    assert [a['reason'] for a in adjusts] == ['saturated', 'saturated']
    assert _set_percents(box) == [100.0, 70.0, 49.0]  # stepped down by 0.7 each time
    assert stream[-1]['ok'] is True and len(project.captures) == 1


def test_adjust_lowers_when_frame_clipped_and_chart_hidden(project):
    # Chart not found but the frame is heavily clipped -> over-exposed, so step down.
    cam = InspectCam([{'found': False}, {'found': True, 'saturated': False}], clip=0.5)
    box = FakeBox()
    stream = drain(
        run_auto_capture_stream(project, cam, box, steady_meter(), [LampStep('D65', 100.0)], frames=1, cfg=FAST)
    )
    assert events(stream, 'adjust')[0]['reason'] == 'saturated'
    assert _set_percents(box) == [100.0, 70.0]
    assert len(project.captures) == 1


def test_adjust_raises_then_fails_when_chart_never_found(project):
    cam = InspectCam([{'found': False}], clip=0.0)  # never found, not clipped -> keep raising
    box = FakeBox()
    stream = drain(
        run_auto_capture_stream(project, cam, box, steady_meter(), [LampStep('D50', 50.0)], frames=1, cfg=FAST)
    )
    assert all(a['reason'] == 'not_found' for a in events(stream, 'adjust'))
    assert _set_percents(box)[0] == 50.0 and _set_percents(box)[-1] == 100.0  # raised to the ceiling
    assert events(stream, 'lamp_error')[0]['illuminant'] == 'D50'
    assert stream[-1]['failed'] == ['D50'] and len(project.captures) == 0  # chartless frame not saved


def test_adjust_can_be_disabled(project):
    cam = InspectCam([{'saturated': True}])  # would trigger adjustment if inspected
    stream = drain(
        run_auto_capture_stream(
            project,
            cam,
            FakeBox(),
            steady_meter(),
            [LampStep('D65', 100.0)],
            frames=1,
            cfg=FAST,
            adjust=AdjustConfig(enabled=False),
        )
    )
    assert not events(stream, 'frame_check') and not events(stream, 'adjust')
    assert cam.inspects == 0 and len(project.captures) == 1


# --- generator: failure policy ----------------------------------------------------
def test_lamp_switch_failure_continues_with_remaining(project):
    box = FakeBox(fail_on={'F12'})
    stream = drain(
        run_auto_capture_stream(
            project, FakeCam(), box, steady_meter(), [LampStep('F12'), LampStep('D65')], frames=1, cfg=FAST
        )
    )
    assert events(stream, 'lamp_error')[0]['illuminant'] == 'F12'
    assert stream[-1]['failed'] == ['F12'] and stream[-1]['captured'] == 1
    assert len(project.captures) == 1


def test_camera_failure_is_fatal_and_releases(project):
    box = FakeBox()
    stream = drain(
        run_auto_capture_stream(project, FakeCam(fail=True), box, steady_meter(), [LampStep('D65')], cfg=FAST)
    )
    assert stream[-1]['event'] == 'error' and 'camera failure' in stream[-1]['error']
    assert not auto_capture.is_running()
    assert ('off',) in box.calls  # lamps not left burning after a fatal error


# --- generator: cancel -------------------------------------------------------------
def test_cancel_between_lamps_stops_and_turns_off(project):
    box = FakeBox()
    gen = run_auto_capture_stream(
        project,
        FakeCam(),
        box,
        steady_meter(),
        [LampStep('F12'), LampStep('D65')],
        frames=1,
        include_darks=True,
        cfg=FAST,
    )
    stream = []
    for event in gen:
        stream.append(event)
        if event['event'] == 'lamp_done':  # cancel after the first lamp completes
            assert auto_capture.request_cancel()
    kinds = [e['event'] for e in stream]
    assert 'cancelled' in kinds
    assert kinds.count('lamp_start') == 1  # second lamp never started
    assert 'waiting_user' not in kinds  # cancel skips the dark phase
    assert stream[-1]['event'] == 'done' and stream[-1]['cancelled'] is True
    assert ('off',) in box.calls
    assert not auto_capture.is_running()


def test_client_disconnect_turns_off_and_releases(project):
    box = FakeBox()
    gen = run_auto_capture_stream(project, FakeCam(), box, steady_meter(), [LampStep('D65')], cfg=FAST)
    next(gen)  # start event; the cycle now holds the lock
    assert auto_capture.is_running()
    gen.close()  # GeneratorExit, as when the SSE client goes away
    assert not auto_capture.is_running()
    assert ('off',) in box.calls


# --- generator: darks and the lens-cap pause ----------------------------------------
def test_darks_wait_for_continue_then_capture_untagged(project):
    box = FakeBox()
    gen = run_auto_capture_stream(
        project, FakeCam(), box, steady_meter(), [LampStep('D65')], frames=2, include_darks=True, cfg=FAST
    )
    stream = []
    for event in gen:
        stream.append(event)
        if event['event'] == 'waiting_user':
            # The generator is suspended before proceed.wait(); releasing it here
            # exercises the real continue path without a second thread.
            assert auto_capture.request_continue()
    assert ('off',) in box.calls  # lightbox off before the dark frames
    darks = [c for c in project.captures if c.image_type == 'dark']
    assert len(darks) == 2
    assert all(c.colour_temp is None and c.lux is None and c.lightmeter is None for c in darks)
    assert stream[-1]['event'] == 'done' and stream[-1]['ok'] is True


def test_continue_when_nothing_waiting_is_rejected():
    assert auto_capture.request_continue() is False
    assert auto_capture.request_cancel() is False


# --- generator: single-run lock ------------------------------------------------------
def test_second_cycle_rejected_while_running(project):
    gen = run_auto_capture_stream(project, FakeCam(), FakeBox(), steady_meter(), [LampStep('D65')], cfg=FAST)
    next(gen)  # holds the lock
    second = drain(run_auto_capture_stream(project, FakeCam(), FakeBox(), steady_meter(), [LampStep('D65')], cfg=FAST))
    assert second[0]['event'] == 'error' and 'already running' in second[0]['error']
    gen.close()
    assert not auto_capture.is_running()


# --- routes ---------------------------------------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    sessions.Workspace(tmp_path).create_project('cam')
    monkeypatch.setattr(app_module, 'get_shared_camera', lambda: FakeCam())
    monkeypatch.setattr(app_module, 'get_shared_lightbox', lambda: FakeBox())
    monkeypatch.setattr(app_module, 'get_shared_lightmeter', lambda: steady_meter())
    monkeypatch.setattr(
        auto_capture,
        'run_auto_capture_stream',
        lambda *a, **kw: iter([{'event': 'start'}, {'event': 'done', 'ok': True}]),
    )
    return app_module.create_app(tmp_path).test_client()


def sse_events(response):
    return [json.loads(line[len('data: ') :]) for line in response.get_data(as_text=True).splitlines() if line]


def test_stream_route_emits_sse(client):
    r = client.get('/projects/cam/auto-capture/stream?lamps=D65:100&frames=2&darks=1')
    assert r.status_code == 200
    assert r.mimetype == 'text/event-stream'
    stream = sse_events(r)
    assert stream[0]['event'] == 'start' and stream[-1]['event'] == 'done'


def test_stream_route_passes_parameters(tmp_path, monkeypatch, client):
    seen = {}

    def fake_stream(project, camera, lightbox, lightmeter, lamps, frames, include_darks, **kw):
        seen.update(lamps=lamps, frames=frames, darks=include_darks)
        yield {'event': 'done', 'ok': True}

    monkeypatch.setattr(auto_capture, 'run_auto_capture_stream', fake_stream)
    client.get('/projects/cam/auto-capture/stream?lamps=D65:100,F12&frames=99&darks=1')
    assert seen['lamps'] == [LampStep('D65', 100.0), LampStep('F12', None)]
    assert seen['frames'] == 16  # clamped to the burst limit
    assert seen['darks'] is True


def test_stream_route_reports_missing_devices_in_stream(client, monkeypatch):
    def _raise():
        raise LightboxError('no lightbox')

    monkeypatch.setattr(app_module, 'get_shared_lightbox', _raise)
    r = client.get('/projects/cam/auto-capture/stream?lamps=D65')
    assert r.status_code == 200  # EventSource cannot read 4xx bodies
    stream = sse_events(r)
    assert stream[0]['event'] == 'error' and 'lightbox' in stream[0]['error'].lower()
    assert stream[-1]['event'] == 'done' and stream[-1]['ok'] is False


def test_stream_route_rejects_bad_lamps_in_stream(client):
    stream = sse_events(client.get('/projects/cam/auto-capture/stream?lamps=D65:200'))
    assert stream[0]['event'] == 'error'


def test_continue_and_cancel_409_when_idle(client):
    assert client.post('/projects/cam/auto-capture/continue').status_code == 409
    assert client.post('/projects/cam/auto-capture/cancel').status_code == 409
