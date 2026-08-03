# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Empirical ISP sharpen tuning: threshold, strength and limit.
#
# The rpi.sharpen scalars shape the sharpening block three ways: 'threshold'
# scales its noise gate (filter responses below it are treated as noise and
# left alone), 'strength' scales the gain applied to responses that pass the
# gate, and 'limit' caps the delta any pixel may receive (halo height on
# strong edges). None of them adapts to analogue gain in the pipeline, so all
# three must be chosen empirically:
#
#   - threshold: sweep candidates and measure spatial noise on the grey
#     patches of a Macbeth chart in the processed output, relative to a
#     sharpen-disabled baseline. Recommend the smallest threshold whose noise
#     stays within tolerance (most real-detail sharpening, no noise crunch).
#   - strength: sweep candidates and measure a moderate-contrast slanted edge
#     in the processed output: MTF50 boost and ESF overshoot/undershoot
#     (halos) versus the same baseline. Recommend the largest strength whose
#     halos and MTF peak stay under their caps (most acuity, no visible
#     halos). Processed output is tone-curve-encoded, so absolute MTF is
#     biased — every figure here is a ratio against the identically-encoded
#     baseline.
#   - limit: at the chosen strength, sweep candidates and measure halos on a
#     high-contrast edge (the only place the delta cap engages). Recommend
#     the largest limit that keeps them under the cap.
#
# Measurements use full-resolution unencoded frames: JPEG quantisation and
# preview downscaling both destroy the fine detail these sweeps measure.
# Results persist to <project>/sharpen/results.json — a subdirectory, so
# calibration runs never see it.

from __future__ import annotations

import contextlib
import json
import shutil
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import mtf
from .camera import CameraError, reload_shared_camera
from .sessions import Project

_RESULTS_DIRNAME = 'sharpen'
_RESULTS_FILENAME = 'results.json'
_TMP_DIRNAME = 'tmp'
RESULTS_VERSION = 2

DEFAULT_GAIN = 8.0
DEFAULT_FRAMES = 4
MAX_FRAMES = 8
# Threshold grid: geometric-ish, bracketing the template default (0.75). The
# tuning threshold multiplies the whole hardware noise gate linearly, so
# geometric spacing samples the response evenly.
DEFAULT_THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.75, 1.5, 2.5, 4.0)
# The default tolerance allows visible-instrument, invisible-eye noise growth:
# the metric's own run-to-run scatter is ~5-10%, and at the high measurement
# gain the extra noise is masked by the noise already present — whereas the
# detail a higher threshold gates away is lost at every gain. A tighter
# tolerance systematically over-gates (measured: it pushed the threshold to
# 1.5 and left sharpening with nothing to work on).
DEFAULT_TOLERANCE = 1.15
# A tuned strength should buy at least this much acutance over sharpening-off;
# anything less means the threshold is gating away the detail the sharpener
# needs, and the threshold choice should be revisited instead.
MIN_USEFUL_BOOST = 1.1
# Strength grid: linear (strength scales the filter responses linearly),
# bracketing the template default 1.0. Limits: geometric around template 0.5.
DEFAULT_STRENGTHS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
DEFAULT_LIMITS = (0.125, 0.25, 0.5, 1.0, 2.0)
# Calibrated against perception: 5% halo growth reads as a clean, subtle
# crispening on real charts; 10% is already punchy and anything near the
# strength-2/limit-1 look (visible ringing) sits far above it.
DEFAULT_OVERSHOOT_CAP = 0.05
# Calibrated against perception: default-tuning sharpening (which reads as
# pleasantly crisp, not over-processed) measures an MTF peak around 1.4, so
# the cap only exists to reject genuinely crunchy response above that.
DEFAULT_PEAK_CAP = 1.5
MAX_EDGES = 5

# The grey (achromatic) patches are the chart's bottom row: every 4th patch
# from index 3 in patch-detector order (see ctt.detection.patches).
GREY_SLICE = slice(3, None, 4)

# High-pass filter for the noise metric: the residual after a Gaussian blur of
# this sigma. Wide enough to pass the band the 5x5 sharpen filters amplify,
# narrow enough to reject illumination shading across the patch window.
_HP_SIGMA = 3.0

# Measurements drifting further than this from the baseline capture suggest
# the lighting or framing moved mid-sweep; the point is flagged, not discarded.
_DRIFT_FRACTION = 0.05

# Edge classification by ESF step height (8-bit luma): the strength sweep
# wants a moderate step (tone curve locally near-linear, limit not engaged);
# the limit sweep wants a strong step (the delta cap actually bites).
_HIGH_CONTRAST_STEP = 0.55 * 255
_MODERATE_CONTRAST_STEP = 0.10 * 255

_SHARPEN_KEY = 'rpi.sharpen'
_SHARPEN_DEFAULTS = {'threshold': 0.75, 'limit': 0.5, 'strength': 1.0}

# One sweep at a time: it owns the shared camera (repeated tuning reloads).
_sharpen_lock = threading.Lock()


class SweepAbort(Exception):
    """A user-visible reason to abort a sweep (streamed as an error event)."""


def is_running() -> bool:
    return _sharpen_lock.locked()


def results_path(project: Project) -> Path:
    return project.path / _RESULTS_DIRNAME / _RESULTS_FILENAME


def read_results(project: Project) -> dict | None:
    """The stored results in version 2 shape (v1 files migrate on read).

    v2 keeps one section per sweep ({'threshold': ..., 'strength': ...}) so a
    run of one sweep never wipes the other's results. The migration never
    rewrites the file on disk.
    """
    path = results_path(project)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if data.get('version') == 1:
        return {
            'version': RESULTS_VERSION,
            'threshold': {k: v for k, v in data.items() if k != 'version'},
            'strength': None,
        }
    return data


def _merge_results(project: Project, section: str, data: dict) -> None:
    """Replace one results section, preserving the other, and persist."""
    results = read_results(project) or {'version': RESULTS_VERSION, 'threshold': None, 'strength': None}
    results['version'] = RESULTS_VERSION
    results[section] = data
    out = results_path(project)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec='seconds')


# --- analysis (camera-free) --------------------------------------------------
def patch_luma(patch_bgr: np.ndarray) -> np.ndarray:
    """Rec.601 luma of a BGR-ordered image or patch window, as float64.

    The sharpen block operates on the luminance channel, so its effects are
    measured there too.
    """
    patch = np.asarray(patch_bgr, dtype=np.float64)
    return 0.299 * patch[..., 2] + 0.587 * patch[..., 1] + 0.114 * patch[..., 0]


def patch_spatial_noise(luma: np.ndarray) -> float:
    """Spatial noise of a flat patch: std of the Gaussian high-pass residual.

    Subtracting a Gaussian-smoothed copy cancels illumination shading of any
    low order while passing the high-frequency band sharpening amplifies. The
    absolute value depends on the filter choice, but the sweep only compares
    ratios against an identically-filtered baseline. The border is excluded
    from the statistic so the blur's edge handling never enters it.
    """
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415 (heavy import)

    luma = np.asarray(luma, dtype=np.float64)
    residual = luma - gaussian_filter(luma, _HP_SIGMA)
    inset = int(np.ceil(2 * _HP_SIGMA))
    if min(luma.shape) <= 2 * inset + 4:
        inset = max((min(luma.shape) - 5) // 2, 0)
    core = residual[inset : luma.shape[0] - inset, inset : luma.shape[1] - inset]
    return float(core.std())


def patch_window_half(centres: np.ndarray) -> int:
    """Half-width of the per-patch sample window (colour_check sizing rule).

    A fifth of the median nearest-neighbour spacing between patch centres, so
    the window stays comfortably inside a patch at any chart size.
    """
    pts = np.asarray(centres, dtype=float)
    dists = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return max(8, int(np.median(dists.min(axis=1)) * 0.2))


def grey_patch_noise(frames: list[np.ndarray], centres: np.ndarray) -> tuple[list[float], list[float]]:
    """Per-grey-patch (spatial noise, mean luma), averaged over the frames.

    centres: all 24 detected patch centres (full-resolution pixel coords);
    the achromatic bottom row is selected here. Frames are BGR-ordered.
    """
    pts = np.asarray(centres, dtype=float)
    half = patch_window_half(pts)
    noise, means = [], []
    for x, y in pts[GREY_SLICE].astype(int):
        per_frame_noise, per_frame_mean = [], []
        for frame in frames:
            h, w = frame.shape[:2]
            x0, x1 = max(x - half, 0), min(x + half + 1, w)
            y0, y1 = max(y - half, 0), min(y + half + 1, h)
            luma = patch_luma(frame[y0:y1, x0:x1])
            per_frame_noise.append(patch_spatial_noise(luma))
            per_frame_mean.append(float(luma.mean()))
        noise.append(float(np.mean(per_frame_noise)))
        means.append(float(np.mean(per_frame_mean)))
    return noise, means


def recommend_threshold(points: list[dict], tolerance: float = DEFAULT_TOLERANCE) -> dict:
    """Pick the smallest threshold whose aggregate noise ratio is in tolerance.

    Smaller thresholds sharpen more real detail, so the smallest acceptable
    one keeps the most sharpening without amplifying noise.
    """
    for point in sorted(points, key=lambda p: p['threshold']):
        if point['aggregate'] is not None and point['aggregate'] <= tolerance:
            return {'threshold': point['threshold'], 'aggregate': point['aggregate'], 'reason': None}
    return {
        'threshold': None,
        'aggregate': None,
        'reason': (
            'no threshold kept grey-patch noise within tolerance — '
            'check the chart is in focus and steadily lit, or raise the tolerance'
        ),
    }


def classify_edges(edges: list[dict]) -> list[str | None]:
    """Classify detected edges by ESF step height: 'high', 'moderate' or None.

    Uses the plateau levels analyse_edge measured (8-bit luma domain). The
    strength sweep measures moderate edges; the limit sweep needs high ones.
    """
    classes: list[str | None] = []
    for edge in edges:
        low, high = edge.get('edge_low'), edge.get('edge_high')
        step = (high - low) if (low is not None and high is not None) else 0.0
        if step >= _HIGH_CONTRAST_STEP:
            classes.append('high')
        elif step >= _MODERATE_CONTRAST_STEP:
            classes.append('moderate')
        else:
            classes.append(None)
    return classes


def _halo(point: dict) -> float | None:
    """The worse of a point's overshoot/undershoot (None when unmeasured)."""
    if point.get('overshoot') is None or point.get('undershoot') is None:
        return None
    return max(point['overshoot'], point['undershoot'])


def recommend_strength(
    points: list[dict], overshoot_cap: float = DEFAULT_OVERSHOOT_CAP, peak_cap: float = DEFAULT_PEAK_CAP
) -> dict:
    """Pick the strength with the greatest perceived sharpness within the caps.

    Sharpness is scored by acutance gain (CSF-weighted MTF area over the
    baseline): sharpening lifts the mid/high band that acutance integrates,
    where MTF50 — often pinned by the optics — barely moves. The gain is not
    guaranteed to rise with strength (the threshold gates responses and the
    limit clips them), so the highest gain inside the halo and MTF-peak caps
    is what "best" means; ties break towards the larger strength.
    """
    best = None
    for point in sorted(points, key=lambda p: p['strength'], reverse=True):
        halo = _halo(point)
        if halo is None or point.get('mtf_peak') is None or point.get('acutance_gain') is None:
            continue
        if halo > overshoot_cap or point['mtf_peak'] > peak_cap:
            continue
        if best is None or point['acutance_gain'] > best['acutance_gain']:
            best = point
    if best is not None:
        return {
            'strength': best['strength'],
            'overshoot': _halo(best),
            'mtf_peak': best['mtf_peak'],
            'acutance_gain': best['acutance_gain'],
            'mtf50_boost': best.get('mtf50_boost'),
            'reason': None,
        }
    return {
        'strength': None,
        'overshoot': None,
        'mtf_peak': None,
        'acutance_gain': None,
        'mtf50_boost': None,
        'reason': 'every strength exceeded the halo or MTF-peak cap — lower the candidates or raise the caps',
    }


def recommend_limit(points: list[dict], overshoot_cap: float = DEFAULT_OVERSHOOT_CAP) -> dict:
    """Pick the largest limit whose high-contrast-edge halo is within the cap.

    Recommends nothing when the halo barely responds to the limit at all —
    at a weak sharpening strength the delta cap never engages, every
    candidate passes trivially, and "largest within cap" would just return
    the top of the grid.
    """
    measured = [h for h in (_halo(p) for p in points) if h is not None]
    # Disengaged = flat AND comfortably low. Flat-but-high halos mean the
    # limit cannot rein them in — that is the every-candidate-fails case.
    if measured and max(measured) - min(measured) < 0.01 and max(measured) < overshoot_cap / 2:
        return {
            'limit': None,
            'overshoot': None,
            'reason': (
                'the limit never engages at this strength (halos barely change across the '
                'candidates) — keep the current value'
            ),
        }
    for point in sorted(points, key=lambda p: p['limit'], reverse=True):
        halo = _halo(point)
        if halo is not None and halo <= overshoot_cap:
            return {'limit': point['limit'], 'overshoot': halo, 'reason': None}
    return {
        'limit': None,
        'overshoot': None,
        'reason': 'every limit exceeded the halo cap on the high-contrast edge — lower the candidates or raise the cap',
    }


def set_sharpen(
    tuning: dict, *, threshold: float | None = None, strength: float | None = None, limit: float | None = None
) -> dict:
    """Set rpi.sharpen values in a tuning dict, in place.

    Handles both tuning shapes: version 2.0 ({'algorithms': [{name: {...}},
    ...]}) and the legacy flat dict keyed by algorithm name. A missing
    rpi.sharpen block is created with the template defaults.
    """
    algorithms = tuning.get('algorithms')
    if isinstance(algorithms, list):
        block = next((entry[_SHARPEN_KEY] for entry in algorithms if _SHARPEN_KEY in entry), None)
        if block is None:
            block = dict(_SHARPEN_DEFAULTS)
            algorithms.append({_SHARPEN_KEY: block})
    else:
        block = tuning.setdefault(_SHARPEN_KEY, dict(_SHARPEN_DEFAULTS))
    if threshold is not None:
        block['threshold'] = float(threshold)
    if strength is not None:
        block['strength'] = float(strength)
    if limit is not None:
        block['limit'] = float(limit)
    return tuning


# --- shared sweep harness -----------------------------------------------------
def base_tuning(project: Project, target: str, model: str | None) -> tuple[Path, str] | None:
    """The tuning file a sweep starts from: (path, 'generated' | 'system').

    Prefers the project's generated tuning for the live ISP; falls back to the
    installed system tuning for the camera model, so a sweep can run before
    any calibration exists.
    """
    generated = project.output_dir / f'{project.name}_{target}.json'
    if generated.exists():
        return generated, 'generated'
    if model:
        from .app import _system_tuning_dirs  # noqa: PLC0415 (avoid an import cycle at load)

        for d in _system_tuning_dirs(target):
            path = d / f'{model}.json'
            if path.exists():
                return path, 'system'
    return None


def _guarded(inner: Iterator[dict]) -> Iterator[dict]:
    """Run a sweep generator under the shared one-at-a-time lock."""
    from . import ctt_runner  # noqa: PLC0415 (local import to avoid a cycle at module load)

    if ctt_runner.is_running():
        yield {'event': 'error', 'error': 'a calibration run is in progress; wait for it to finish'}
        return
    if not _sharpen_lock.acquire(blocking=False):
        yield {'event': 'error', 'error': 'a sharpen sweep is already running'}
        return
    try:
        yield from inner
    finally:
        _sharpen_lock.release()


def _resolve_base(project: Project, camera) -> tuple[str, Path, str]:
    """The live ISP target and base tuning for a sweep, or a SweepAbort."""
    from .camera import platform_target  # noqa: PLC0415

    target = platform_target()
    if target is None:
        raise SweepAbort('could not determine the ISP platform')
    base = base_tuning(project, target, camera.model)
    if base is None:
        raise SweepAbort(f'no tuning file found for {target} (run CTT or install a system tuning)')
    return target, base[0], base[1]


def _lock_operating_point(prev: dict, gain: float) -> dict:
    """The manual operating point a sweep holds: the requested gain at the
    same image brightness (exposure rescaled from the AE-settled values),
    with the current white balance frozen."""
    current_exposure = prev.get('exposure') or 10_000
    current_gain = max(prev.get('gain') or 1.0, 0.1)
    return {
        'exposure': max(int(current_exposure * current_gain / gain), 50),
        'gain': gain,
        'colour_gains': prev.get('colour_gains') or [2.0, 2.0],
    }


def _apply_locked_controls(camera, locked: dict) -> None:
    """Re-apply the sweep's manual operating point and wait for it to land.

    Every tuning reload builds a fresh camera with default (auto) controls, so
    the locked exposure/gain/colour-gains must be pushed again and verified via
    metadata readback — controls take a few frames to land.
    """
    camera.set_controls(
        {
            'auto_exposure': False,
            'exposure': locked['exposure'],
            'gain': locked['gain'],
            'fps': 0,
            'colour_gains': locked['colour_gains'],
        }
    )
    for _ in range(20):
        ctrl = camera.get_controls()
        exp_ok = abs(ctrl['exposure'] - locked['exposure']) <= max(0.02 * locked['exposure'], 25)
        gain_ok = abs(ctrl['gain'] - locked['gain']) <= 0.02 * locked['gain']
        if exp_ok and gain_ok:
            return
    raise CameraError(
        f'controls did not settle after tuning reload: requested {locked["exposure"]} us '
        f'/ gain {locked["gain"]:.2f}, applied {ctrl["exposure"]} us / {ctrl["gain"]:.2f}'
    )


def _write_point_tunings(base_path: Path, tmp_dir: Path, points: list[dict], prefix: str = 'p') -> list[Path]:
    """One uniquely-named tuning file per sweep point.

    Each entry in points is a set_sharpen kwargs dict (the caller includes the
    baseline). reload_shared_camera() no-ops when asked for the path it
    already runs, so every point needs its own file — never one file
    rewritten in place.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    base_text = base_path.read_text()
    paths = []
    for i, kwargs in enumerate(points):
        path = tmp_dir / f'{prefix}_{i}.json'
        path.write_text(json.dumps(set_sharpen(json.loads(base_text), **kwargs), indent=4))
        paths.append(path)
    return paths


def _capture_point(tuning_path: Path, locked: dict, frames: int):
    """Reload the camera with one point's tuning, settle, capture the burst."""
    camera = reload_shared_camera(tuning_file=str(tuning_path))
    _apply_locked_controls(camera, locked)
    return camera, camera.capture_still_frames(frames)


def _restore_camera(orig_tuning: str | None, prev: dict, tmp_dir: Path) -> None:
    """Best-effort: original tuning + controls back, temp tunings removed."""
    with contextlib.suppress(Exception):
        restored = reload_shared_camera(tuning_file=orig_tuning)
        restored.set_controls(
            {
                'auto_exposure': prev.get('auto_exposure', True),
                'exposure': prev.get('exposure'),
                'gain': prev.get('gain'),
                'fps': prev.get('fps', 30),
                'awb': True,
            }
        )
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _locate_chart_fullres(frame: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Detect the Macbeth chart on a full-res frame; centres in full-res coords.

    The detector works best around preview size, so the frame is downscaled
    for detection and the centres scaled back up.
    """
    import cv2  # noqa: PLC0415

    from ctt.detection.macbeth import locate_chart  # noqa: PLC0415

    h, w = frame.shape[:2]
    scale = min(1.0, 1920 / w)
    small = frame
    if scale < 1:
        small = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    try:
        res = locate_chart(small)
    except Exception:  # detection is best-effort; treat any failure as not found
        res = None
    if res is None:
        return None
    _corners, centres, confidence = res
    return np.asarray(centres, dtype=float) / scale, float(confidence)


# --- threshold sweep -----------------------------------------------------------
def sweep_stream(
    project: Project,
    camera,
    *,
    gain: float = DEFAULT_GAIN,
    frames: int = DEFAULT_FRAMES,
    thresholds: list[float] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Iterator[dict]:
    """Run a sharpen-threshold sweep, yielding structured progress events.

    Reloads the shared camera once per point (baseline + each threshold), so
    the caller's camera handle goes stale: the sweep tracks the live handle
    internally and restores the original tuning and controls on the way out.
    """
    yield from _guarded(_threshold_sweep(project, camera, gain, frames, thresholds, tolerance))


def _threshold_sweep(
    project: Project,
    camera,
    gain: float,
    frames: int,
    thresholds: list[float] | None,
    tolerance: float,
) -> Iterator[dict]:
    gain = max(1.0, float(gain))
    frames = max(1, min(int(frames), MAX_FRAMES))
    thresholds = sorted({round(float(t), 4) for t in (thresholds or DEFAULT_THRESHOLDS)})
    tolerance = max(1.0, float(tolerance))
    if not thresholds or any(t <= 0 for t in thresholds):
        yield {'event': 'error', 'error': 'thresholds must be positive numbers'}
        return

    orig_tuning = camera.tuning_file
    prev = camera.get_controls()
    tmp_dir = project.path / _RESULTS_DIRNAME / _TMP_DIRNAME
    try:
        target, base_path, base_kind = _resolve_base(project, camera)

        # Pre-flight on the live preview: the chart must be visible and usable
        # before we start reloading tunings.
        chart = camera.detect_chart()
        if not chart['found']:
            raise SweepAbort('no Macbeth chart detected — frame the chart and try again')
        if chart['small']:
            raise SweepAbort('the Macbeth chart is too small in frame — move it closer')
        if chart['saturated']:
            raise SweepAbort('the Macbeth chart is over-exposed — reduce the lighting or exposure')

        yield {
            'event': 'start',
            'gain': gain,
            'frames': frames,
            'thresholds': thresholds,
            'tolerance': tolerance,
            'base': {'kind': base_kind, 'name': base_path.name},
            'total': len(thresholds) + 1,
        }

        locked = _lock_operating_point(prev, gain)
        yield {
            'event': 'log',
            'line': f'operating point: {locked["exposure"]} us at gain {gain:g}, colour gains '
            f'{locked["colour_gains"][0]:.2f}/{locked["colour_gains"][1]:.2f}',
        }

        point_paths = _write_point_tunings(
            base_path, tmp_dir, [{'strength': 0.0}] + [{'threshold': t} for t in thresholds]
        )

        # Baseline: sharpening disabled entirely (strength 0).
        yield {'event': 'log', 'line': 'capturing baseline (sharpening disabled)'}
        camera, captured = _capture_point(point_paths[0], locked, frames)
        located = _locate_chart_fullres(captured[0])
        if located is None:
            raise SweepAbort('chart detection failed on the full-resolution capture')
        centres, confidence = located
        yield {'event': 'chart', 'confidence': round(confidence, 3)}
        baseline_noise, baseline_means = grey_patch_noise(captured, centres)
        warnings = []
        if min(baseline_noise) < 0.5:
            warnings.append('baseline noise is below 0.5 LSB on some patches; ratios may be quantisation-limited')
            yield {'event': 'log', 'line': f'warning: {warnings[-1]}'}
        yield {
            'event': 'baseline',
            'patch_noise': [round(n, 4) for n in baseline_noise],
            'patch_means': [round(m, 2) for m in baseline_means],
            'index': 0,
            'total': len(thresholds) + 1,
        }

        points = []
        for i, threshold in enumerate(thresholds):
            camera, captured = _capture_point(point_paths[i + 1], locked, frames)
            noise, means = grey_patch_noise(captured, centres)
            ratios = [n / max(b, 1e-6) for n, b in zip(noise, baseline_noise, strict=True)]
            point_warnings = []
            drift = max(abs(m - b) / max(b, 1.0) for m, b in zip(means, baseline_means, strict=True))
            if drift > _DRIFT_FRACTION:
                point_warnings.append(f'patch brightness drifted {drift * 100:.1f}% from baseline')
            point = {
                'threshold': threshold,
                'patch_noise': [round(n, 4) for n in noise],
                'ratios': [round(r, 4) for r in ratios],
                'aggregate': round(float(np.median(ratios)), 4),
                'warnings': point_warnings,
            }
            points.append(point)
            yield {'event': 'point', **point, 'index': i + 1, 'total': len(thresholds) + 1}

        recommended = recommend_threshold(points, tolerance)
        _merge_results(
            project,
            'threshold',
            {
                'generated_at': _now_iso(),
                'target': target,
                'base_tuning': {'kind': base_kind, 'name': base_path.name},
                'settings': {
                    'gain': gain,
                    'exposure_us': locked['exposure'],
                    'frames': frames,
                    'tolerance': tolerance,
                    'thresholds': thresholds,
                },
                'chart': {'confidence': round(confidence, 3)},
                'baseline': {
                    'patch_noise': [round(n, 4) for n in baseline_noise],
                    'patch_means': [round(m, 2) for m in baseline_means],
                },
                'points': points,
                'recommended': recommended,
                'applied': None,
                'warnings': warnings,
            },
        )
        yield {'event': 'done', 'ok': True, 'recommended': recommended}
    except SweepAbort as err:
        yield {'event': 'error', 'error': str(err)}
    except Exception as err:  # a failed sweep must still end the stream cleanly
        yield {'event': 'error', 'error': str(err)}
    finally:
        _restore_camera(orig_tuning, prev, tmp_dir)


# --- strength + limit sweep ----------------------------------------------------
def _edge_metrics(mean_luma: np.ndarray, box: dict) -> dict:
    """Analyse one locked edge box on a mean-luma frame (full-res, step 1).

    The box was validated as a single clean edge on the baseline capture, so
    the monotonicity bar is relaxed here (strong sharpening halos and
    high-gain noise legitimately add ESF variation on top of the step) and
    the baseline's fitted edge line is reused rather than re-estimated —
    refitting each noisy capture lets a marginal fit flap across the
    angle/border guards even though the scene has not moved.
    """
    crop = mean_luma[box['y'] : box['y'] + box['h'], box['x'] : box['x'] + box['w']]
    return mtf.analyse_edge(crop, plane_step=1, min_monotonicity=0.15, edge_line=box.get('edge_line'))


def _aggregate_edges(per_edge: list[dict], baseline: dict[int, dict]) -> dict:
    """Median metrics across the edges that analysed successfully.

    MTF50 is expressed as a boost over the same edge's baseline, and the halo
    figures as growth over the baseline's: a printed chart edge can carry
    static structure that reads as a constant over/undershoot at every
    strength, and only the part sharpening adds is of interest.
    """
    ok = [e for e in per_edge if e.get('ok')]
    if not ok:
        return {'acutance_gain': None, 'mtf50_boost': None, 'mtf_peak': None, 'overshoot': None, 'undershoot': None}

    def med(vals):
        return round(float(np.median(vals)), 4) if vals else None

    def halo_growth(key):
        vals = [max(0.0, e[key] - (baseline.get(e['edge'], {}).get(key) or 0.0)) for e in ok if e.get(key) is not None]
        return med(vals)

    def gain_over_baseline(key):
        vals = [
            e[key] / baseline[e['edge']][key]
            for e in ok
            if e.get(key) is not None and baseline.get(e['edge'], {}).get(key)
        ]
        return med(vals)

    return {
        'acutance_gain': gain_over_baseline('acutance'),
        'mtf50_boost': gain_over_baseline('mtf50'),
        'mtf_peak': med([e['mtf_peak'] for e in ok if e.get('mtf_peak') is not None]),
        'overshoot': halo_growth('overshoot'),
        'undershoot': halo_growth('undershoot'),
    }


def _slim_edge(index: int, result: dict) -> dict:
    """A per-edge record without the bulky MTF curve."""
    keep = ('mtf50', 'mtf_peak', 'acutance', 'overshoot', 'undershoot', 'edge_low', 'edge_high', 'angle_deg')
    slim = {'edge': index, 'ok': bool(result.get('ok'))}
    slim.update({k: result[k] for k in keep if k in result})
    if not slim['ok']:
        slim['reason'] = result.get('reason')
    return slim


def strength_sweep_stream(
    project: Project,
    camera,
    *,
    gain: float = DEFAULT_GAIN,
    frames: int = DEFAULT_FRAMES,
    strengths: list[float] | None = None,
    limits: list[float] | None = None,
    overshoot_cap: float = DEFAULT_OVERSHOOT_CAP,
    peak_cap: float = DEFAULT_PEAK_CAP,
) -> Iterator[dict]:
    """Run a sharpen strength (+ limit) sweep, yielding structured progress events.

    Strength is measured on moderate-contrast slanted edges (MTF50 boost, MTF
    peak, ESF halos vs the strength-0 baseline); at the recommended strength,
    limit is then measured on high-contrast edges where the delta cap engages.
    Grey-patch noise is recorded per point when the Macbeth chart is also in
    the scene. Same camera-reload/restore behaviour as the threshold sweep.
    """
    yield from _guarded(_strength_sweep(project, camera, gain, frames, strengths, limits, overshoot_cap, peak_cap))


def _strength_sweep(
    project: Project,
    camera,
    gain: float,
    frames: int,
    strengths: list[float] | None,
    limits: list[float] | None,
    overshoot_cap: float,
    peak_cap: float,
) -> Iterator[dict]:
    gain = max(1.0, float(gain))
    frames = max(1, min(int(frames), MAX_FRAMES))
    strengths = sorted({round(float(s), 4) for s in (strengths or DEFAULT_STRENGTHS)})
    limits = sorted({round(float(v), 4) for v in (limits or DEFAULT_LIMITS)})
    overshoot_cap = float(overshoot_cap)
    peak_cap = float(peak_cap)
    if not strengths or any(s <= 0 for s in strengths) or any(v <= 0 for v in limits):
        yield {'event': 'error', 'error': 'strengths and limits must be positive numbers'}
        return
    if overshoot_cap <= 0 or peak_cap <= 1.0:
        yield {'event': 'error', 'error': 'the overshoot cap must be positive and the MTF-peak cap above 1'}
        return

    orig_tuning = camera.tuning_file
    prev = camera.get_controls()
    tmp_dir = project.path / _RESULTS_DIRNAME / _TMP_DIRNAME
    total = 1 + len(strengths) + len(limits)
    try:
        target, base_path, base_kind = _resolve_base(project, camera)
        yield {
            'event': 'start',
            'gain': gain,
            'frames': frames,
            'strengths': strengths,
            'limits': limits,
            'overshoot_cap': overshoot_cap,
            'peak_cap': peak_cap,
            'base': {'kind': base_kind, 'name': base_path.name},
            'total': total,
        }

        locked = _lock_operating_point(prev, gain)
        yield {
            'event': 'log',
            'line': f'operating point: {locked["exposure"]} us at gain {gain:g}, colour gains '
            f'{locked["colour_gains"][0]:.2f}/{locked["colour_gains"][1]:.2f}',
        }

        point_paths = _write_point_tunings(
            base_path, tmp_dir, [{'strength': 0.0}] + [{'strength': s} for s in strengths], prefix='s'
        )

        # Baseline (strength 0): find and lock the edges, classify by contrast.
        yield {'event': 'log', 'line': 'capturing baseline (sharpening disabled)'}
        camera, captured = _capture_point(point_paths[0], locked, frames)
        mean_luma = np.mean([patch_luma(f) for f in captured], axis=0)
        edges = mtf.detect_edges(mean_luma, plane_step=1, max_regions=MAX_EDGES)
        if not edges:
            raise SweepAbort(
                'no slanted edge found in the processed capture — place a slanted-edge '
                'target (a few degrees off vertical, moderate contrast) in frame'
            )
        classes = classify_edges(edges)
        moderate = [i for i, c in enumerate(classes) if c == 'moderate']
        high = [i for i, c in enumerate(classes) if c == 'high']
        warnings = []
        if not moderate:
            if not high:
                raise SweepAbort('no usable edge contrast found — the edge step is too shallow to measure')
            moderate = high
            warnings.append('no moderate-contrast edge found; measuring strength on the high-contrast edge')
        if not high:
            warnings.append('no high-contrast edge found; the limit sweep is skipped')
        for w in warnings:
            yield {'event': 'log', 'line': f'warning: {w}'}
        yield {
            'event': 'edges',
            'count': len(edges),
            'edges': [
                {**{k: e[k] for k in ('x', 'y', 'w', 'h', 'zone')}, 'contrast_class': c}
                for e, c in zip(edges, classes, strict=True)
            ],
        }

        # Per-edge baseline: MTF50 for boosts, halos for growth-over-baseline,
        # plateau level for the drift check.
        baseline_by_edge = {i: edges[i] for i in range(len(edges))}
        baseline_high = {i: edges[i]['edge_high'] for i in range(len(edges))}
        baseline_edges = [_slim_edge(i, edges[i]) for i in range(len(edges))]

        # Grey-patch noise is free when the Macbeth chart shares the scene.
        located = _locate_chart_fullres(captured[0])
        centres = located[0] if located else None
        baseline_noise = None
        if centres is not None:
            baseline_noise, _ = grey_patch_noise(captured, centres)
            yield {'event': 'log', 'line': 'Macbeth chart co-visible: grey-patch noise recorded per point'}
        baseline_mtf50_median = round(float(np.median([edges[i]['mtf50'] for i in moderate])), 4)
        yield {
            'event': 'baseline',
            'mtf50': baseline_mtf50_median,
            'per_edge': baseline_edges,
            'index': 0,
            'total': total,
        }

        def measure_point(mean_luma_pt, captured_pt, edge_idx):
            per_edge = [_slim_edge(i, _edge_metrics(mean_luma_pt, edges[i])) for i in edge_idx]
            agg = _aggregate_edges(per_edge, baseline_by_edge)
            point_warnings = []
            dropped = [e['edge'] for e in per_edge if not e['ok']]
            if dropped:
                point_warnings.append(f'edge(s) {dropped} failed to analyse at this point')
            drifted = [
                e['edge']
                for e in per_edge
                if e['ok']
                and abs(e['edge_high'] - baseline_high[e['edge']])
                > _DRIFT_FRACTION * max(baseline_high[e['edge']], 1.0)
            ]
            if drifted:
                point_warnings.append(f'edge(s) {drifted} brightness drifted from baseline (lighting/framing moved?)')
            noise_aggregate = None
            if centres is not None and baseline_noise is not None:
                noise, _means = grey_patch_noise(captured_pt, centres)
                ratios = [n / max(b, 1e-6) for n, b in zip(noise, baseline_noise, strict=True)]
                noise_aggregate = round(float(np.median(ratios)), 4)
            return per_edge, agg, point_warnings, noise_aggregate

        # Strength phase: measure the moderate-contrast edges.
        points = []
        for i, strength in enumerate(strengths):
            camera, captured = _capture_point(point_paths[i + 1], locked, frames)
            mean_luma = np.mean([patch_luma(f) for f in captured], axis=0)
            per_edge, agg, point_warnings, noise_aggregate = measure_point(mean_luma, captured, moderate)
            point = {
                'strength': strength,
                **agg,
                'noise_aggregate': noise_aggregate,
                'per_edge': per_edge,
                'warnings': point_warnings,
            }
            points.append(point)
            yield {'event': 'point', 'phase': 'strength', **point, 'index': i + 1, 'total': total}

        recommended = recommend_strength(points, overshoot_cap, peak_cap)
        # Close the loop with the threshold sweep: when even the best in-cap
        # strength buys almost no sharpness, the base tuning's threshold is
        # gating away the detail — the fix is a lower threshold, not more
        # strength.
        if recommended['strength'] is not None and (recommended['acutance_gain'] or 0) < MIN_USEFUL_BOOST:
            warnings.append(
                f'the tuned sharpening only gains x{recommended["acutance_gain"]:.2f} acutance — the base '
                'threshold is likely gating away detail; revisit the threshold sweep before accepting this'
            )
            yield {'event': 'log', 'line': f'warning: {warnings[-1]}'}

        # Limit phase: at the recommended strength, measure the high-contrast edges.
        limit_points = []
        recommended_limit = None
        if recommended['strength'] is None:
            warnings.append('limit sweep skipped: no strength satisfied the caps')
        elif high:
            limit_paths = _write_point_tunings(
                base_path,
                tmp_dir,
                [{'strength': recommended['strength'], 'limit': v} for v in limits],
                prefix='l',
            )
            for i, limit in enumerate(limits):
                camera, captured = _capture_point(limit_paths[i], locked, frames)
                mean_luma = np.mean([patch_luma(f) for f in captured], axis=0)
                per_edge, agg, point_warnings, noise_aggregate = measure_point(mean_luma, captured, high)
                point = {
                    'limit': limit,
                    **agg,
                    'noise_aggregate': noise_aggregate,
                    'per_edge': per_edge,
                    'warnings': point_warnings,
                }
                limit_points.append(point)
                yield {'event': 'point', 'phase': 'limit', **point, 'index': len(strengths) + i + 1, 'total': total}
            recommended_limit = recommend_limit(limit_points, overshoot_cap)

        _merge_results(
            project,
            'strength',
            {
                'generated_at': _now_iso(),
                'target': target,
                'base_tuning': {'kind': base_kind, 'name': base_path.name},
                'settings': {
                    'gain': gain,
                    'exposure_us': locked['exposure'],
                    'frames': frames,
                    'strengths': strengths,
                    'limits': limits,
                    'overshoot_cap': overshoot_cap,
                    'peak_cap': peak_cap,
                },
                'edges': [
                    {**{k: e[k] for k in ('x', 'y', 'w', 'h', 'zone')}, 'contrast_class': c}
                    for e, c in zip(edges, classes, strict=True)
                ],
                'baseline': {
                    'mtf50': baseline_mtf50_median,
                    'per_edge': baseline_edges,
                    'noise': [round(n, 4) for n in baseline_noise] if baseline_noise else None,
                },
                'points': points,
                'limit_points': limit_points,
                'recommended': recommended,
                'recommended_limit': recommended_limit,
                'applied': None,
                'warnings': warnings,
            },
        )
        yield {'event': 'done', 'ok': True, 'recommended': recommended, 'recommended_limit': recommended_limit}
    except SweepAbort as err:
        yield {'event': 'error', 'error': str(err)}
    except Exception as err:  # a failed sweep must still end the stream cleanly
        yield {'event': 'error', 'error': str(err)}
    finally:
        _restore_camera(orig_tuning, prev, tmp_dir)
