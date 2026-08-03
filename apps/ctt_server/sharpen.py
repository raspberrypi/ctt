# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Empirical ISP sharpen-threshold tuning against the Macbeth grey patches.
#
# The rpi.sharpen 'threshold' scales the sharpen block's noise gate: filter
# responses below it are treated as noise and left alone, responses above it
# get sharpened. Too low and flat areas visibly crunch with amplified sensor
# noise; too high and fine real detail loses its sharpening. Nothing in the
# pipeline adapts the value to analogue gain, so it must be chosen
# empirically: sweep candidate thresholds over the loaded tuning, measure
# spatial noise on the grey patches of a Macbeth chart in the processed
# (post-ISP) output, and recommend the smallest threshold whose noise stays
# within tolerance of a sharpen-disabled baseline.
#
# Measurements use full-resolution unencoded frames: JPEG quantisation and
# preview downscaling both destroy the fine-grain noise this sweep exists to
# detect. Results persist to <project>/sharpen/results.json — a subdirectory,
# so calibration runs never see it.

from __future__ import annotations

import contextlib
import json
import shutil
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .camera import CameraError, reload_shared_camera
from .sessions import Project

_RESULTS_DIRNAME = 'sharpen'
_RESULTS_FILENAME = 'results.json'
_TMP_DIRNAME = 'tmp'
RESULTS_VERSION = 1

DEFAULT_GAIN = 8.0
DEFAULT_FRAMES = 4
MAX_FRAMES = 8
# Geometric-ish grid bracketing the template default (0.75). The tuning
# threshold multiplies the whole hardware noise gate linearly, so geometric
# spacing samples the response evenly.
DEFAULT_THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.75, 1.5, 2.5, 4.0)
DEFAULT_TOLERANCE = 1.05

# The grey (achromatic) patches are the chart's bottom row: every 4th patch
# from index 3 in patch-detector order (see ctt.detection.patches).
GREY_SLICE = slice(3, None, 4)

# High-pass filter for the noise metric: the residual after a Gaussian blur of
# this sigma. Wide enough to pass the band the 5x5 sharpen filters amplify,
# narrow enough to reject illumination shading across the patch window.
_HP_SIGMA = 3.0

# Patch means drifting further than this from the baseline capture suggest the
# lighting or exposure moved mid-sweep; the point is flagged, not discarded.
_DRIFT_FRACTION = 0.05

_SHARPEN_KEY = 'rpi.sharpen'
_SHARPEN_DEFAULTS = {'threshold': 0.75, 'limit': 0.5, 'strength': 1.0}

# One sweep at a time: it owns the shared camera (repeated tuning reloads).
_sharpen_lock = threading.Lock()


def is_running() -> bool:
    return _sharpen_lock.locked()


def results_path(project: Project) -> Path:
    return project.path / _RESULTS_DIRNAME / _RESULTS_FILENAME


def read_results(project: Project) -> dict | None:
    path = results_path(project)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec='seconds')


# --- analysis (camera-free) --------------------------------------------------
def patch_luma(patch_bgr: np.ndarray) -> np.ndarray:
    """Rec.601 luma of a BGR-ordered patch window, as float64.

    The sharpen block operates on the luminance channel, so its noise
    amplification is measured there too.
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


def set_sharpen(tuning: dict, *, threshold: float | None = None, strength: float | None = None) -> dict:
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
    return tuning


# --- sweep orchestration ------------------------------------------------------
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


def _write_point_tunings(base_path: Path, tmp_dir: Path, thresholds: list[float]) -> list[Path]:
    """One uniquely-named tuning file per sweep point (baseline first).

    reload_shared_camera() no-ops when asked for the path it already runs, so
    every point needs its own file — never one file rewritten in place.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    base_text = base_path.read_text()
    paths = []
    baseline = tmp_dir / 'baseline.json'
    baseline.write_text(json.dumps(set_sharpen(json.loads(base_text), strength=0.0), indent=4))
    paths.append(baseline)
    for i, threshold in enumerate(thresholds):
        path = tmp_dir / f't_{i}.json'
        path.write_text(json.dumps(set_sharpen(json.loads(base_text), threshold=threshold), indent=4))
        paths.append(path)
    return paths


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
    from . import ctt_runner  # noqa: PLC0415 (local import to avoid a cycle at module load)

    if ctt_runner.is_running():
        yield {'event': 'error', 'error': 'a calibration run is in progress; wait for it to finish'}
        return
    if not _sharpen_lock.acquire(blocking=False):
        yield {'event': 'error', 'error': 'a sharpen sweep is already running'}
        return
    try:
        yield from _sweep(project, camera, gain, frames, thresholds, tolerance)
    finally:
        _sharpen_lock.release()


def _sweep(
    project: Project,
    camera,
    gain: float,
    frames: int,
    thresholds: list[float] | None,
    tolerance: float,
) -> Iterator[dict]:
    from .camera import platform_target  # noqa: PLC0415

    gain = max(1.0, float(gain))
    frames = max(1, min(int(frames), MAX_FRAMES))
    thresholds = sorted({round(float(t), 4) for t in (thresholds or DEFAULT_THRESHOLDS)})
    tolerance = max(1.0, float(tolerance))
    if not thresholds or any(t <= 0 for t in thresholds):
        yield {'event': 'error', 'error': 'thresholds must be positive numbers'}
        return

    target = platform_target()
    if target is None:
        yield {'event': 'error', 'error': 'could not determine the ISP platform'}
        return
    base = base_tuning(project, target, camera.model)
    if base is None:
        yield {'event': 'error', 'error': f'no tuning file found for {target} (run CTT or install a system tuning)'}
        return
    base_path, base_kind = base

    # Pre-flight on the live preview: the chart must be visible and usable
    # before we start reloading tunings.
    chart = camera.detect_chart()
    if not chart['found']:
        yield {'event': 'error', 'error': 'no Macbeth chart detected — frame the chart and try again'}
        return
    if chart['small']:
        yield {'event': 'error', 'error': 'the Macbeth chart is too small in frame — move it closer'}
        return
    if chart['saturated']:
        yield {'event': 'error', 'error': 'the Macbeth chart is over-exposed — reduce the lighting or exposure'}
        return

    orig_tuning = camera.tuning_file
    prev = camera.get_controls()
    tmp_dir = project.path / _RESULTS_DIRNAME / _TMP_DIRNAME
    try:
        yield {
            'event': 'start',
            'gain': gain,
            'frames': frames,
            'thresholds': thresholds,
            'tolerance': tolerance,
            'base': {'kind': base_kind, 'name': base_path.name},
            'total': len(thresholds) + 1,
        }

        # Lock the operating point: the requested gain at equivalent brightness
        # (exposure rescaled from the current AE-settled values), fixed white
        # balance. Every reload re-applies exactly these numbers.
        current_exposure = prev.get('exposure') or 10_000
        current_gain = max(prev.get('gain') or 1.0, 0.1)
        locked = {
            'exposure': max(int(current_exposure * current_gain / gain), 50),
            'gain': gain,
            'colour_gains': prev.get('colour_gains') or [2.0, 2.0],
        }
        yield {
            'event': 'log',
            'line': f'operating point: {locked["exposure"]} us at gain {gain:g}, colour gains '
            f'{locked["colour_gains"][0]:.2f}/{locked["colour_gains"][1]:.2f}',
        }

        point_paths = _write_point_tunings(base_path, tmp_dir, thresholds)

        # Baseline: sharpening disabled entirely (strength 0).
        yield {'event': 'log', 'line': 'capturing baseline (sharpening disabled)'}
        camera = reload_shared_camera(tuning_file=str(point_paths[0]))
        _apply_locked_controls(camera, locked)
        captured = camera.capture_still_frames(frames)
        located = _locate_chart_fullres(captured[0])
        if located is None:
            yield {'event': 'error', 'error': 'chart detection failed on the full-resolution capture'}
            return
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
            camera = reload_shared_camera(tuning_file=str(point_paths[i + 1]))
            _apply_locked_controls(camera, locked)
            captured = camera.capture_still_frames(frames)
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
        results = {
            'version': RESULTS_VERSION,
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
        }
        out = results_path(project)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        yield {'event': 'done', 'ok': True, 'recommended': recommended}
    except Exception as err:  # a failed sweep must still end the stream cleanly
        yield {'event': 'error', 'error': str(err)}
    finally:
        # Best-effort: put the camera back on its original tuning and controls
        # (the tuning reload rebuilt it with everything on auto).
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
