# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Automated capture cycle over a lightbox and a light meter.
#
# run_auto_capture_stream() is a generator that switches the lightbox through a list
# of lamps, waits for the light meter to stabilise after each switch, captures a
# Macbeth burst per lamp tagged from the measured reading, and optionally finishes
# with dark frames (lightbox off, paused until the user confirms the lens cap).
# Each step yields a structured event dict, streamed to the browser as SSE by the
# route in app.py.
#
# Like ctt_runner, at most one cycle runs at a time (module lock). Unlike ctt_runner
# no worker thread is needed: every step is under our control, so a plain generator
# suffices — and a client disconnect raises GeneratorExit at the next yield, whose
# `finally` turns the lightbox off and releases the lock.

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from devices import LightboxError, LightmeterError

from .camera import CameraError

if TYPE_CHECKING:
    from devices.lightmeter import Measurement

    from .sessions import Capture, Project

logger = logging.getLogger(__name__)

_run_lock = threading.Lock()
_control: AutoControl | None = None  # non-None only while a cycle holds _run_lock


@dataclass(frozen=True)
class LampStep:
    """One lamp in the cycle: which illuminant, and at what intensity (None = the
    lamp's stored default)."""

    illuminant: str
    percent: float | None = None


@dataclass(frozen=True)
class StabiliseConfig:
    """Tunables for the wait-until-the-meter-settles loop."""

    min_wait_s: float = 5.0  # always wait this long after switching, before reading
    window: int = 10  # consecutive readings that must agree
    sample_interval_s: float = 2.0  # fixed cadence between readings (0.5 Hz), meter-latency independent
    lux_tol: float = 0.02  # window lux spread <= 2 % of the window mean
    cct_tol_k: float = 40.0  # window CCT spread <= 40 K
    timeout_s: float = 120.0  # then proceed with the last reading, flagged as a warning
    max_read_failures: int = 3  # consecutive measure() failures -> lamp error


@dataclass(frozen=True)
class AdjustConfig:
    """Tunables for adjusting the lamp so the Macbeth chart is framed and unclipped.

    Before each lamp's burst the live frame is inspected: if the chart is over-exposed
    the lamp is stepped down, if it can't be found the lamp is stepped up (in case the
    scene is simply too dark), each change followed by a fresh meter stabilisation.
    """

    enabled: bool = True
    max_adjust: int = 4  # intensity changes per lamp before giving up
    min_confidence: float = 0.0  # chart-detection confidence floor (0 = just require found)
    clip_tol: float = 0.02  # frame clipping above this (chart absent) reads as over-exposed
    down_factor: float = 0.7  # intensity multiplier when the chart is saturated
    up_factor: float = 1.4  # intensity multiplier when the chart can't be found
    min_percent: float = 1.0
    max_percent: float = 100.0


@dataclass
class LampOutcome:
    """The result of processing one lamp, for the main loop to turn into events."""

    added: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    error: str | None = None
    fatal: bool = False
    cancelled: bool = False


@dataclass
class AutoControl:
    """Cross-thread controls for the running cycle (single instance under _run_lock)."""

    cancel: threading.Event = field(default_factory=threading.Event)
    proceed: threading.Event = field(default_factory=threading.Event)
    waiting: bool = False  # blocked on the lens-cap prompt


class StabiliseResult(NamedTuple):
    reading: Measurement | None  # None: cancelled or the meter kept failing
    timed_out: bool
    error: str | None = None  # why no reading (meter failure / under range), for the caller


def is_running() -> bool:
    return _run_lock.locked()


def request_cancel() -> bool:
    """Signal the running cycle to stop; True if one was running."""
    control = _control
    if control is None:
        return False
    control.cancel.set()
    control.proceed.set()  # unblock a lens-cap wait so the cancel is seen promptly
    return True


def request_continue() -> bool:
    """Release a cycle paused at the lens-cap prompt; True if one was waiting."""
    control = _control
    if control is None or not control.waiting:
        return False
    control.proceed.set()
    return True


def parse_lamps(spec: str) -> list[LampStep]:
    """Parse a lamps query string: 'D65:100,F12:80' (intensity optional per lamp).

    Raises ValueError for an empty list, a malformed pair or an out-of-range
    intensity. Illuminant names are validated later by Lightbox.set_illuminant.
    """
    lamps = []
    for part in filter(None, (p.strip() for p in spec.split(','))):
        name, sep, pct = part.partition(':')
        percent = None
        if sep:
            percent = float(pct)
            if not 0 <= percent <= 100:
                raise ValueError(f'intensity for {name!r} must be 0-100, got {percent}')
        if not name:
            raise ValueError(f'malformed lamp entry {part!r}')
        lamps.append(LampStep(illuminant=name, percent=percent))
    if not lamps:
        raise ValueError('no lamps selected')
    return lamps


def save_burst(
    project: Project,
    shots,
    image_type: str,
    colour_temp: int | None,
    *,
    lux: int | None = None,
    reading: dict | None = None,
) -> list[Capture]:
    """Save a burst's shots as indexed captures, all carrying the same reading.

    indexed=True always: repeated cycles must append _<n> files, never overwrite.
    """
    return [
        project.add_capture(
            dng, image_type, colour_temp, lux=lux, controls=meta, jpeg_bytes=jpg, indexed=True, lightmeter=reading
        )
        for dng, jpg, meta in shots
    ]


def capture_entry(cap: Capture) -> dict:
    """A capture in the shape the manual capture endpoint returns in 'added'."""
    return {
        'filename': cap.filename,
        'image_type': cap.image_type,
        'colour_temp': cap.colour_temp,
        'lux': cap.lux,
        'label': cap.label,
        'valid': True,
        'jpeg': 'saved',
        'lightmeter': cap.lightmeter,
    }


def _stable_suffix(window: list, cfg: StabiliseConfig) -> int:
    """Length of the longest run of trailing readings that agree within tolerance."""
    for n in range(len(window), 1, -1):
        tail = window[-n:]
        lux = [m.illuminance_lux for m in tail]
        cct = [m.cct for m in tail]
        spread = max(lux) - min(lux)
        mean = sum(lux) / n
        lux_ok = spread == 0 or (mean > 0 and spread / mean <= cfg.lux_tol)
        if lux_ok and max(cct) - min(cct) <= cfg.cct_tol_k:
            return n
    return min(len(window), 1)


def _stabilise(lightmeter, illuminant: str, cfg: StabiliseConfig, cancel: threading.Event) -> Iterator[dict]:
    """Poll the meter until consecutive readings agree; yields a progress event per
    sample and returns a StabiliseResult (sub-generator, consumed with `yield from`).

    On timeout the last reading is still returned (flagged timed_out): a slowly
    drifting lamp gives a usable tag, and the full reading is recorded with the
    capture for later scrutiny. Persistent meter failures or a cancel return None.
    """
    if cancel.wait(cfg.min_wait_s):
        return StabiliseResult(None, False)
    start = time.monotonic()
    window: list[Measurement] = []
    failures = 0
    last_error = 'no usable light-meter reading'
    while not cancel.is_set():
        tick = time.monotonic()
        try:
            reading = lightmeter.measure()
        except LightmeterError as err:
            failures += 1
            last_error = 'light-meter read failed'
            logger.warning('auto-capture: meter reading failed (%d/%d): %s', failures, cfg.max_read_failures, err)
            if failures >= cfg.max_read_failures:
                return StabiliseResult(None, False, last_error)
            if cancel.wait(max(0.0, cfg.sample_interval_s - (time.monotonic() - tick))):
                return StabiliseResult(None, False)
            continue
        # An out-of-range reading is not a measurement; treat it like a read failure
        # (a persistently dark scene fails the step with a clear "add light" message).
        if not reading.in_range:
            failures += 1
            last_error = 'light is below the meter’s measurable range — add more light'
            yield {
                'event': 'reading',
                'illuminant': illuminant,
                'lux': reading.illuminance_lux,
                'cct': reading.cct,
                'stable_count': 0,
                'needed': cfg.window,
                'elapsed_s': round(time.monotonic() - start, 1),
                'in_range': False,
            }
            if failures >= cfg.max_read_failures:
                return StabiliseResult(None, False, last_error)
            if cancel.wait(max(0.0, cfg.sample_interval_s - (time.monotonic() - tick))):
                return StabiliseResult(None, False)
            continue
        failures = 0
        window.append(reading)
        if len(window) > cfg.window:
            window.pop(0)
        stable = _stable_suffix(window, cfg)
        elapsed = time.monotonic() - start
        yield {
            'event': 'reading',
            'illuminant': illuminant,
            'lux': reading.illuminance_lux,
            'cct': reading.cct,
            'stable_count': stable,
            'needed': cfg.window,
            'elapsed_s': round(elapsed, 1),
            'in_range': True,
        }
        if len(window) == cfg.window and stable == cfg.window:
            yield {
                'event': 'stable',
                'illuminant': illuminant,
                'lux': reading.illuminance_lux,
                'cct': reading.cct,
                'elapsed_s': round(elapsed, 1),
            }
            return StabiliseResult(reading, False)
        if elapsed > cfg.timeout_s:
            yield {
                'event': 'stabilise_timeout',
                'illuminant': illuminant,
                'lux': reading.illuminance_lux,
                'cct': reading.cct,
                'elapsed_s': round(elapsed, 1),
            }
            return StabiliseResult(reading, True)
        # Hold a fixed cadence between samples (independent of meter latency), staying
        # cancellable during the wait.
        if cancel.wait(max(0.0, cfg.sample_interval_s - (time.monotonic() - tick))):
            return StabiliseResult(None, False)
    return StabiliseResult(None, False)


def _inspect_frame(camera) -> dict | None:
    """Best-effort look at the live frame: is the Macbeth chart present and unclipped?

    Returns {found, saturated, confidence, clip_max}, or None when the detector or
    histogram is unavailable — in which case the caller captures without adjusting
    rather than blocking the cycle on a flaky check.
    """
    try:
        chart = camera.detect_chart()
    except Exception:
        return None
    clip_max = 0.0
    if not chart.get('found'):
        # The chart being absent can mean the whole frame is blown out; use the global
        # clipping to tell over-exposure (step down) from too-dark/mis-framed (step up).
        try:
            hist = camera.histogram()
            clip_max = max(hist.get('clipping', {}).values(), default=0.0)
        except Exception:
            clip_max = 0.0
    return {
        'found': bool(chart.get('found')),
        'saturated': bool(chart.get('saturated')),
        'confidence': chart.get('confidence'),
        'clip_max': clip_max,
    }


def _capture_lamp(project, camera, lightbox, lightmeter, lamp, frames, cfg, adjust, control) -> Iterator[dict]:
    """Switch to a lamp, stabilise, optionally adjust its intensity for the chart, and
    capture a Macbeth burst. Yields progress events; returns a LampOutcome.
    """
    percent = lamp.percent
    try:
        lightbox.set_illuminant(lamp.illuminant, percent)
    except LightboxError as err:
        return LampOutcome(error=str(err))

    warnings: list[str] = []
    reading_obj = None
    note = None
    # Adjustment needs a numeric starting intensity to scale; a lamp left at its stored
    # default (percent None) is captured without adjustment.
    can_adjust = adjust.enabled and percent is not None
    for attempt in range(adjust.max_adjust + 1):
        result = yield from _stabilise(lightmeter, lamp.illuminant, cfg, control.cancel)
        if control.cancel.is_set():
            return LampOutcome(cancelled=True)
        if result.reading is None:
            return LampOutcome(error=result.error or 'no usable light-meter reading')
        reading_obj = result.reading
        if result.timed_out:
            warnings.append(f'{lamp.illuminant}: meter did not stabilise within {cfg.timeout_s:.0f} s')
        if not can_adjust:
            break
        check = _inspect_frame(camera)
        if check is None:
            break  # no detector available: capture as-is
        yield {
            'event': 'frame_check',
            'illuminant': lamp.illuminant,
            'found': check['found'],
            'saturated': check['saturated'],
            'confidence': check['confidence'],
        }
        if check['found'] and not check['saturated'] and (check['confidence'] or 0) >= adjust.min_confidence:
            break  # framed and unclipped: good to capture
        if attempt == adjust.max_adjust:
            note = 'chart not found' if not check['found'] else 'chart still saturated'
            break
        # Over-exposed -> step down; chart absent but not clipped -> step up (maybe dark).
        if check['saturated'] or (not check['found'] and check['clip_max'] > adjust.clip_tol):
            new_percent = max(adjust.min_percent, percent * adjust.down_factor)
            reason = 'saturated'
        elif not check['found']:
            new_percent = min(adjust.max_percent, percent * adjust.up_factor)
            reason = 'not_found'
        else:  # found but low confidence: a framing problem the light can't fix
            note = 'low-confidence chart'
            break
        if abs(new_percent - percent) < 0.5:  # already at the limit; can't move further
            note = 'chart not found' if not check['found'] else 'chart still saturated'
            break
        percent = new_percent
        yield {'event': 'adjust', 'illuminant': lamp.illuminant, 'percent': round(percent, 1), 'reason': reason}
        try:
            lightbox.set_illuminant(lamp.illuminant, percent)
        except LightboxError as err:
            return LampOutcome(error=str(err))
        # loop: re-stabilise at the new intensity before re-checking

    # A chart that never appeared cannot serve as a Macbeth calibration frame; skip it.
    if note == 'chart not found':
        return LampOutcome(error='Macbeth chart not found (even after adjusting the light)')
    if note:
        warnings.append(f'{lamp.illuminant}: {note}')

    reading = reading_obj.to_dict()
    reading.pop('spectrum_1nm', None)  # keep the sidecar compact, as manual captures do
    yield {'event': 'capturing', 'illuminant': lamp.illuminant, 'frames': frames}
    try:
        shots = camera.capture_burst(frames)
    except CameraError as err:
        return LampOutcome(fatal=True, error=f'camera failure: {err}')
    caps = save_burst(
        project,
        shots,
        'macbeth',
        round(reading_obj.cct),
        lux=round(reading_obj.illuminance_lux),
        reading=reading,
    )
    return LampOutcome(added=[capture_entry(c) for c in caps], warnings=warnings)


def run_auto_capture_stream(
    project: Project,
    camera,
    lightbox,
    lightmeter,
    lamps: list[LampStep],
    frames: int = 3,
    include_darks: bool = False,
    cfg: StabiliseConfig | None = None,
    adjust: AdjustConfig | None = None,
) -> Iterator[dict]:
    """Run the auto-capture cycle, yielding structured progress events.

    Per-lamp failures are reported and the cycle continues with the remaining lamps
    (each lamp is an independent calibration point); only a camera failure aborts.
    Exactly one terminal event is emitted: 'done', or 'error' for a fatal failure.
    On cancel, fatal error or client disconnect the lightbox is switched off; a
    normally completed cycle without darks leaves the last lamp lit.
    """
    global _control
    cfg = cfg or StabiliseConfig()
    adjust = adjust or AdjustConfig()
    if not _run_lock.acquire(blocking=False):
        yield {'event': 'error', 'error': 'an auto-capture cycle is already running'}
        return
    control = _control = AutoControl()
    captured_lamps = 0
    failed: list[str] = []
    warnings: list[str] = []
    fatal = normal_finish = False
    try:
        yield {
            'event': 'start',
            'lamps': [{'illuminant': lamp.illuminant, 'percent': lamp.percent} for lamp in lamps],
            'frames': frames,
            'include_darks': include_darks,
        }
        for index, lamp in enumerate(lamps, start=1):
            if control.cancel.is_set():
                break
            yield {
                'event': 'lamp_start',
                'illuminant': lamp.illuminant,
                'percent': lamp.percent,
                'index': index,
                'total': len(lamps),
            }
            outcome = yield from _capture_lamp(
                project, camera, lightbox, lightmeter, lamp, frames, cfg, adjust, control
            )
            if outcome.cancelled:
                break
            if outcome.fatal:
                fatal = True
                yield {'event': 'error', 'error': outcome.error}
                break
            if outcome.error:
                failed.append(lamp.illuminant)
                yield {'event': 'lamp_error', 'illuminant': lamp.illuminant, 'error': outcome.error}
                continue
            warnings.extend(outcome.warnings)
            captured_lamps += 1
            yield {
                'event': 'captured',
                'illuminant': lamp.illuminant,
                'added': outcome.added,
                'counts': project.counts(),
            }
            yield {'event': 'lamp_done', 'illuminant': lamp.illuminant, 'index': index, 'total': len(lamps)}

        if include_darks and not fatal and not control.cancel.is_set():
            with contextlib.suppress(LightboxError):
                lightbox.off()
            control.proceed.clear()
            control.waiting = True
            try:
                while not control.cancel.is_set():
                    # Re-emitted periodically: keeps the stream alive through proxies and
                    # bounds disconnect detection while we sit at the prompt.
                    yield {'event': 'waiting_user', 'message': 'Fit the lens cap, then press Continue.'}
                    if control.proceed.wait(15):
                        break
            finally:
                control.waiting = False
            if not control.cancel.is_set():
                yield {'event': 'capturing', 'illuminant': None, 'frames': frames}
                try:
                    shots = camera.capture_burst(frames)
                except CameraError as err:
                    fatal = True
                    yield {'event': 'error', 'error': f'camera failure: {err}'}
                else:
                    caps = save_burst(project, shots, 'dark', None)
                    yield {
                        'event': 'captured',
                        'illuminant': None,
                        'added': [capture_entry(c) for c in caps],
                        'counts': project.counts(),
                    }

        if not fatal:
            cancelled = control.cancel.is_set()
            if cancelled:
                yield {'event': 'cancelled'}
            yield {
                'event': 'done',
                'ok': not cancelled and not failed,
                'captured': captured_lamps,
                'failed': failed,
                'warnings': warnings,
                'cancelled': cancelled,
            }
            normal_finish = not cancelled
    finally:
        # Reached on completion, cancel, fatal error, or a client disconnect
        # (GeneratorExit). Never leave lamps burning unattended in the failure cases;
        # a normal finish keeps the last lamp lit for the user.
        if not normal_finish:
            with contextlib.suppress(Exception):
                lightbox.off()
        _control = None
        _run_lock.release()
