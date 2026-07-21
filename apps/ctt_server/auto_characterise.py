# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Gap-driven automated sensor characterisation over a lightbox and a light meter.
#
# run_auto_characterise_stream() completes a characterisation with the least work: it
# inventories what the project already holds (characterise.quick_scan + results.json),
# then only captures or sweeps what is missing, matched by operating point (gain). The
# lightbox provides steady illumination for the sweep and flat-field captures; the light
# meter gates each on stabilisation (reusing auto_capture._stabilise). Dark frames are
# captured last, after a lens-cap pause.
#
# Like the other long-running flows it yields structured event dicts for the browser to
# render; the reused characterise.sweep_stream / analyse_stream emit plain lines, which
# are wrapped as 'log' events. At most one cycle runs at a time (module lock), and it
# refuses to start alongside a calibration or another characterisation.

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING

from devices import LightboxError

from . import characterise, ctt_runner
from .auto_capture import AutoControl, StabiliseConfig, _stabilise, save_burst
from .camera import CameraError

if TYPE_CHECKING:
    from .sessions import Project

logger = logging.getLogger(__name__)

_run_lock = threading.Lock()
_control: AutoControl | None = None


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


def _gaps(project: Project, gains: list[float], tol: float) -> list[dict]:
    """Per requested gain, decide whether each input is present (reuse) or missing.

    Darks/flats are matched to an existing capture group at the same gain; the sweep is
    considered present when results.json holds a reliable PTC fit at that gain and is not
    stale (new captures since the last analysis invalidate it).
    """
    scan = characterise.quick_scan(project)
    groups = scan['groups']
    stale = scan['stale']
    results = characterise.read_results(project)
    fits = results['ptc']['fits'] if results else []

    def has_group(kind: str, gain: float) -> bool:
        return any(g['kind'] == kind and abs(g['gain'] - gain) <= tol for g in groups)

    def has_sweep(gain: float) -> bool:
        return not stale and any(abs(f['gain'] - gain) <= tol and f.get('reliable') for f in fits)

    return [
        {
            'gain': gain,
            'darks': 'reuse' if has_group('dark', gain) else 'capture',
            'flats': 'reuse' if has_group('flat', gain) else 'capture',
            'sweep': 'reuse' if has_sweep(gain) else 'sweep',
        }
        for gain in gains
    ]


def _wrap_char_stream(lines: Iterator[str]) -> Iterator[dict]:
    """Forward a CHAR_EXIT-terminated line stream as 'log' events; return success."""
    ok = False
    for line in lines:
        if line.startswith('CHAR_EXIT '):
            ok = line.strip().endswith('0')
            continue
        yield {'event': 'log', 'line': line}
    return ok


def _lamp_cct(lightbox, lamp, reading) -> int | None:
    """Colour temperature to tag a flat with: the measured reading if available, else
    the lamp's nominal CCT from the lightbox."""
    if reading is not None:
        return round(reading.cct)
    with contextlib.suppress(Exception):
        info = lightbox.info()
        channel = info['channel']
        temps = info.get('illuminant_temps', {})
        return temps.get(channel) or temps.get(str(channel))
    return None


def _prep_dark(camera, gain: float) -> None:
    """Manual, shortest exposure at the requested gain — standard for read noise/DSNU."""
    exp_min = camera.get_controls().get('exposure_min') or 100
    camera.set_controls({'auto_exposure': False, 'exposure': int(exp_min), 'gain': float(gain), 'fps': 0, 'awb': False})


def _prep_flat(camera, gain: float, roi_fraction: float = 0.5) -> None:
    """Manual gain at ~50 % of the DN swing: probe saturation, then expose to half of it,
    so the flat is well-exposed and unclipped for PRNU."""
    exp_min = camera.get_controls().get('exposure_min') or 100
    camera.set_controls({'auto_exposure': False, 'gain': float(gain), 'exposure': int(exp_min), 'fps': 0, 'awb': False})
    sat_exposure, _ = characterise._find_saturation(camera, float(gain), roi_fraction)
    camera.set_controls(
        {
            'auto_exposure': False,
            'gain': float(gain),
            'exposure': max(sat_exposure // 2, int(exp_min)),
            'fps': 0,
            'awb': False,
        }
    )


def run_auto_characterise_stream(
    project: Project,
    camera,
    lightbox,
    lightmeter,
    gains: list[float],
    lamp: str,
    intensity: float | None = None,
    include_darks: bool = True,
    include_flats: bool = True,
    include_sweep: bool = True,
    dark_frames: int = 16,
    flat_frames: int = 16,
    sweep_points: int = 10,
    sweep_frames: int = 8,
    cfg: StabiliseConfig | None = None,
    gain_tol: float = 0.05,
) -> Iterator[dict]:
    """Run the gap-driven auto-characterise cycle, yielding structured progress events.

    Lit steps (sweep, flats) come first while the lamp is on; dark frames are captured
    last after the box is switched off and the user has fitted the lens cap. Existing
    inputs at a matching gain are reused. A camera failure aborts; other per-step failures
    are reported and the cycle continues. Exactly one terminal event: 'done' or 'error'.
    """
    global _control
    cfg = cfg or StabiliseConfig()
    gains = sorted({round(float(g), 4) for g in gains}) or [1.0]

    if ctt_runner.is_running() or characterise._char_lock.locked():
        yield {'event': 'error', 'error': 'a calibration or characterisation is already running'}
        return
    if not _run_lock.acquire(blocking=False):
        yield {'event': 'error', 'error': 'an auto-characterise cycle is already running'}
        return
    control = _control = AutoControl()
    prev = None
    reused: list[str] = []
    captured: list[str] = []
    swept: list[float] = []
    warnings: list[str] = []
    fatal = normal_finish = False
    try:
        plan = _gaps(project, gains, gain_tol)
        yield {'event': 'start', 'gains': gains, 'lamp': lamp}
        yield {'event': 'plan', 'gains': plan}

        missing_sweep = [p['gain'] for p in plan if p['sweep'] == 'sweep'] if include_sweep else []
        missing_flats = [p['gain'] for p in plan if p['flats'] == 'capture'] if include_flats else []
        missing_darks = [p['gain'] for p in plan if p['darks'] == 'capture'] if include_darks else []
        for p in plan:
            if include_sweep and p['sweep'] == 'reuse':
                reused.append(f'sweep g{p["gain"]:g}')
            if include_flats and p['flats'] == 'reuse':
                reused.append(f'flat g{p["gain"]:g}')
            if include_darks and p['darks'] == 'reuse':
                reused.append(f'dark g{p["gain"]:g}')

        prev = camera.get_controls()

        # --- lit steps: sweep then flats (lamp on, meter-gated) ----------------
        reading = None
        if missing_sweep or missing_flats:
            yield {'event': 'phase', 'name': 'illuminating', 'lamp': lamp}
            try:
                lightbox.set_illuminant(lamp, intensity)
            except LightboxError as err:
                fatal = True
                yield {'event': 'error', 'error': f'lightbox: {err}'}
            if not fatal:
                result = yield from _stabilise(lightmeter, lamp, cfg, control.cancel)
                reading = result.reading

        if missing_sweep and not fatal and not control.cancel.is_set():
            yield {'event': 'phase', 'name': 'sweep', 'gains': missing_sweep}
            ok = yield from _wrap_char_stream(
                characterise.sweep_stream(
                    project, camera, missing_sweep, points_per_gain=sweep_points, frames=sweep_frames
                )
            )
            if ok:
                swept.extend(missing_sweep)
            else:
                warnings.append('sweep did not complete cleanly')

        if missing_flats and not fatal and not control.cancel.is_set():
            cct = _lamp_cct(lightbox, lamp, reading)
            for gain in missing_flats:
                if control.cancel.is_set():
                    break
                yield {'event': 'phase', 'name': 'flat', 'gain': gain}
                try:
                    _prep_flat(camera, gain)
                    shots = camera.capture_burst(flat_frames)
                except CameraError as err:
                    fatal = True
                    yield {'event': 'error', 'error': f'camera failure: {err}'}
                    break
                caps = save_burst(project, shots, 'alsc', cct)
                captured.append(f'flat g{gain:g}')
                yield {'event': 'captured', 'kind': 'flat', 'gain': gain, 'added': [c.filename for c in caps]}

        # --- dark frames last: box off, lens-cap pause -------------------------
        if missing_darks and not fatal and not control.cancel.is_set():
            with contextlib.suppress(LightboxError):
                lightbox.off()
            control.proceed.clear()
            control.waiting = True
            try:
                while not control.cancel.is_set():
                    yield {'event': 'waiting_user', 'message': 'Fit the lens cap, then press Continue.'}
                    if control.proceed.wait(15):
                        break
            finally:
                control.waiting = False
            for gain in missing_darks:
                if control.cancel.is_set():
                    break
                yield {'event': 'phase', 'name': 'dark', 'gain': gain}
                try:
                    _prep_dark(camera, gain)
                    shots = camera.capture_burst(dark_frames)
                except CameraError as err:
                    fatal = True
                    yield {'event': 'error', 'error': f'camera failure: {err}'}
                    break
                caps = save_burst(project, shots, 'dark', None)
                captured.append(f'dark g{gain:g}')
                yield {'event': 'captured', 'kind': 'dark', 'gain': gain, 'added': [c.filename for c in caps]}

        # --- merge newly-captured darks/flats into results (the sweep self-merges) ---
        if captured and not fatal and not control.cancel.is_set():
            yield {'event': 'phase', 'name': 'analyse'}
            yield from _wrap_char_stream(characterise.analyse_stream(project))

        if not fatal:
            cancelled = control.cancel.is_set()
            if cancelled:
                yield {'event': 'cancelled'}
            yield {
                'event': 'done',
                'ok': not cancelled,
                'reused': reused,
                'captured': captured,
                'swept': swept,
                'warnings': warnings,
                'cancelled': cancelled,
            }
            normal_finish = not cancelled
    finally:
        if prev is not None:
            with contextlib.suppress(Exception):
                camera.set_controls(
                    {
                        'auto_exposure': prev.get('auto_exposure', True),
                        'exposure': prev.get('exposure'),
                        'gain': prev.get('gain'),
                        'fps': prev.get('fps', 30),
                        'awb': True,
                    }
                )
        if not normal_finish:
            with contextlib.suppress(Exception):
                lightbox.off()
        _control = None
        _run_lock.release()
