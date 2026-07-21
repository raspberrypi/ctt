# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Offline sensor characterisation over a project's existing captures.
#
# Orchestrates ctt.characterisation against the DNGs already in a project:
# dark bursts give the black-level pedestal, DSNU and direct read noise; ALSC
# flat-field bursts give PRNU and photon-transfer points; Macbeth bursts add
# show-only PTC points. Results persist to <project>/characterisation/
# results.json — a subdirectory, so calibration runs never see it. Full
# PTC/linearity/full-well/gain-sweep metrics need live exposure sweeps and are
# reported as unavailable until that capture path exists.

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Iterator
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ctt.algorithms.black_level import measure_dark_image
from ctt.characterisation import (
    CaptureGroup,
    FrameSet,
    PtcPoint,
    centre_roi,
    fit_ptc,
    frameset_from_dngs,
    gr_plane,
    ptc_point,
    scan_project,
    spatial_stats,
    temporal_stats,
)
from ctt.characterisation.ptc import dynamic_range, full_well, linearity, snr_curve
from ctt.core.image_loader import dng_load_image

from .sessions import Project

_RESULTS_DIRNAME = 'characterisation'
_RESULTS_FILENAME = 'results.json'
RESULTS_VERSION = 1

# Analysis loads every DNG in the project; only one walk at a time (and never
# alongside a calibration run — both read the same files on a small machine).
_char_lock = threading.Lock()


def is_running() -> bool:
    return _char_lock.locked()


def results_path(project: Project) -> Path:
    return project.path / _RESULTS_DIRNAME / _RESULTS_FILENAME


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec='seconds')


def _fingerprint(groups: list[CaptureGroup]) -> list[dict]:
    out = []
    for g in groups:
        for p in g.paths:
            stat = p.stat()
            out.append({'name': p.name, 'mtime': stat.st_mtime, 'size': stat.st_size})
    return sorted(out, key=lambda e: e['name'])


def _excluded(project: Project) -> set[str]:
    return {c.filename for c in project.captures if c.excluded}


def _group_summary(g: CaptureGroup) -> dict:
    return {
        'label': g.label,
        'kind': g.kind,
        'frames': len(g.paths),
        'exposure_us': g.exposure_us,
        'gain': g.gain,
        'width': g.width,
        'height': g.height,
        'sigbits': g.sigbits,
        'blacklevel': g.blacklevel,
        'colour_temp': g.colour_temp,
        'lux': g.lux,
        'warnings': list(g.warnings),
    }


def read_results(project: Project) -> dict | None:
    path = results_path(project)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# Page-load scans are EXIF-only but still ~100 ms per DNG; cache per project,
# keyed by the directory's stat fingerprint, so unchanged projects are free.
_scan_cache: dict[str, tuple] = {}


def quick_scan(project: Project) -> dict:
    """A fast inventory of the project's characterisation inputs + staleness.

    EXIF-only (no pixel data) and cached against the directory's stat
    fingerprint; the staleness flag compares the current fingerprint with the
    one recorded in the persisted results.
    """
    excluded = _excluded(project)
    stat_key = tuple((p.name, p.stat().st_mtime, p.stat().st_size) for p in sorted(project.path.glob('*.dng'))) + tuple(
        sorted(excluded)
    )
    cached = _scan_cache.get(project.name)
    if cached is not None and cached[0] == stat_key:
        summaries, fingerprint = cached[1], cached[2]
    else:
        groups = scan_project(project.path, excluded)
        summaries = [_group_summary(g) for g in groups]
        fingerprint = _fingerprint(groups)
        _scan_cache[project.name] = (stat_key, summaries, fingerprint)
    results = read_results(project)
    stale = results is not None and results.get('inputs') != fingerprint
    return {
        'groups': summaries,
        'has_results': results is not None,
        'stale': stale,
    }


def _dark_analysis(group: CaptureGroup, roi_fraction: float) -> tuple[dict, FrameSet]:
    """Pedestal (shared measurement), read noise and DSNU from a dark burst.

    Loads each frame once: the per-channel pedestal reuses
    ctt.algorithms.black_level.measure_dark_image — identical by construction
    with the /blacklevel endpoint and calibration runs — while the Gr ROI crop
    feeds the temporal/spatial statistics.
    """
    frames = []
    pedestals = []
    roi = None
    first = None
    for path in group.paths:
        img = dng_load_image(None, str(path), demosaic=False)
        pedestals.append(measure_dark_image(img))
        plane = gr_plane(img)
        if roi is None:
            roi = centre_roi(plane.shape, roi_fraction)
            first = img
        frames.append(plane[roi].copy())
    fs = FrameSet(
        frames=frames,
        exposure_us=first.exposure,
        gain=first.againQ8_norm,
        blacklevel_16=float(first.blacklevel_16),
        sigbits=first.sigbits,
        channel='Y' if first.pattern == 128 else 'Gr',
        label=group.label,
        sources=[p.name for p in group.paths],
    )
    ts = temporal_stats(fs)
    ss = spatial_stats(fs)
    measured = float(np.mean([p['black_level'] for p in pedestals]))
    dark = {
        'available': True,
        'label': group.label,
        'pedestal': {
            'black_level_16': round(measured, 1),
            'metadata_black_level_16': fs.blacklevel_16,
            'frames': pedestals,
        },
        'read_noise_dn': round(float(np.sqrt(ts.var_dn2)), 2) if ts else None,
        'read_noise_e': None,  # filled in when a reliable conversion gain exists
        'dsnu_dn': round(ss.residual_std_dn, 2),
        'dsnu_e': None,
        'exposure_us': group.exposure_us,
        'gain': group.gain,
        'n_frames': len(group.paths),
        'channel': fs.channel,
    }
    if ts is None:
        dark['warnings'] = ['single dark frame: no temporal read noise']
    return dark, fs


def analyse_stream(project: Project, roi_fraction: float = 0.5) -> Iterator[str]:
    """Run the offline analysis, yielding progress lines; ends with CHAR_EXIT <n>.

    The full walk loads every usable DNG once (roughly 0.5-1 s each on a Pi),
    so this streams over SSE exactly like a calibration run.
    """
    from . import ctt_runner  # local import to avoid a cycle at module load

    if ctt_runner.is_running():
        yield 'ERROR: a calibration is running; wait for it to finish'
        yield 'CHAR_EXIT 2'
        return
    if not _char_lock.acquire(blocking=False):
        yield 'ERROR: a characterisation analysis is already running'
        yield 'CHAR_EXIT 2'
        return
    try:
        yield f'$ characterise (offline)  project={project.name}  roi={roi_fraction:.2f}'
        groups = scan_project(project.path, _excluded(project))
        if not groups:
            yield 'ERROR: no usable captures found (need dark/, alsc_ or Macbeth DNG bursts)'
            yield 'CHAR_EXIT 1'
            return

        results: dict = {
            'version': RESULTS_VERSION,
            'generated_at': _now_iso(),
            'roi_fraction': roi_fraction,
            'inputs': _fingerprint(groups),
            'camera': None,
            'groups': [_group_summary(g) for g in groups],
            'dark': {'available': False, 'unavailable_reason': 'no dark frames captured'},
            'ptc': {'points': [], 'fits': [], 'unavailable_reason': None},
            'prnu': {'available': False, 'groups': [], 'best_pct': None},
            'warnings': [],
        }
        for g in groups:
            for w in g.warnings:
                results['warnings'].append({'group': g.label, 'message': w})

        yield f'Found {len(groups)} capture groups'

        # --- darks: pedestal, read noise, DSNU (the foundation) -------------
        darks = [g for g in groups if g.kind == 'dark']
        dark_fs = None
        dark_group = None
        if darks:
            dark_group = max(darks, key=lambda g: len(g.paths))
            yield f'\t{dark_group.label}: {len(dark_group.paths)} dark frames (pedestal, read noise, DSNU)'
            dark, dark_fs = _dark_analysis(dark_group, roi_fraction)
            results['dark'] = dark
            yield (
                f'\t\tblack level {dark["pedestal"]["black_level_16"]:.0f} DN16'
                f'  ·  read noise {dark["read_noise_dn"]} DN16'
                f'  ·  DSNU {dark["dsnu_dn"]} DN16'
            )
            for extra in darks:
                if extra is not dark_group:
                    yield f'\t{extra.label}: additional dark operating point noted (not analysed)'

        # --- flats and charts: PTC points + PRNU -----------------------------
        points = []  # serialised entries for results.json
        point_objs = []  # the PtcPoint objects, for fitting
        prnu_entries = []
        for g in groups:
            if g.kind not in ('flat', 'chart'):
                continue
            yield f'\t{g.label}: {len(g.paths)} frames ({g.kind})'
            try:
                fs = frameset_from_dngs([str(p) for p in g.paths], roi_fraction)
            except Exception as err:
                results['warnings'].append({'group': g.label, 'message': f'load failed: {err}'})
                yield f'\t\tWARNING: load failed ({err})'
                continue
            fs.label = g.label
            # Substitute the measured dark pedestal only on an exact mode match;
            # flats from a different sensor mode keep their own metadata pedestal.
            if dark_group is not None and results['dark'].get('available'):
                same_mode = (g.width, g.height, g.sigbits, g.blacklevel) == (
                    dark_group.width,
                    dark_group.height,
                    dark_group.sigbits,
                    dark_group.blacklevel,
                )
                if same_mode:
                    fs.blacklevel_16 = results['dark']['pedestal']['black_level_16']
                    pedestal_source = 'dark'
                else:
                    pedestal_source = 'metadata'
                    results['warnings'].append(
                        {
                            'group': g.label,
                            'message': 'sensor mode differs from the dark burst; using the metadata black level',
                        }
                    )
            else:
                pedestal_source = 'metadata'

            point = ptc_point(fs, source='flat' if g.kind == 'flat' else 'chart')
            if point is not None:
                entry = asdict(point)
                entry['pedestal_source'] = pedestal_source
                points.append(entry)
                point_objs.append(point)
                state = 'clipped' if point.clipped else 'ok'
                yield f'\t\tPTC point: mean {point.mean_dn:.0f} DN16, variance {point.var_dn2:.1f} ({state})'
            else:
                yield '\t\tsingle frame: no temporal statistics'
            if g.kind == 'flat' and len(fs.frames) >= 2 and point is not None and not point.clipped:
                ss = spatial_stats(fs)
                if ss.nonuniformity_pct is not None:
                    prnu_entries.append(
                        {'label': g.label, 'prnu_pct': round(ss.nonuniformity_pct, 3), 'mean_dn': round(ss.mean_dn, 1)}
                    )
                    yield f'\t\tPRNU {ss.nonuniformity_pct:.2f}% at mean {ss.mean_dn:.0f} DN16'

        # --- fits + honesty ---------------------------------------------------
        results['ptc']['points'] = points
        fits = fit_ptc(point_objs)  # clip/chart exclusion happens inside
        results['ptc']['fits'] = [asdict(f) for f in fits]
        reliable = next((f for f in fits if f.reliable), None)
        if reliable is None:
            n_flat = sum(1 for p in point_objs if not p.clipped and p.source == 'flat')
            results['ptc']['unavailable_reason'] = (
                f'{n_flat} flat-field operating point{"s" if n_flat != 1 else ""} — a conversion-gain fit needs '
                f'at least 3 well-spread exposures at one gain (run the live sweep)'
            )
            yield f'\tPTC: {results["ptc"]["unavailable_reason"]}'
        else:
            rn = f'{reliable.read_noise_e:.2f} e-' if reliable.read_noise_e is not None else 'n/a'
            yield f'\tPTC fit (gain {reliable.gain}): K {reliable.k_e_per_dn:.3f} e-/DN16, read noise {rn}'
            # Electron-referred dark metrics become quotable.
            if results['dark'].get('available'):
                k = reliable.k_e_per_dn
                if results['dark']['read_noise_dn'] is not None:
                    results['dark']['read_noise_e'] = round(results['dark']['read_noise_dn'] * k, 2)
                results['dark']['dsnu_e'] = round(results['dark']['dsnu_dn'] * k, 2)

        if prnu_entries:
            results['prnu'] = {
                'available': True,
                'groups': prnu_entries,
                'best_pct': min(e['prnu_pct'] for e in prnu_entries),
            }
        else:
            results['prnu']['unavailable_reason'] = 'no unclipped multi-frame flat-field burst'

        if dark_fs is not None:
            results['camera'] = {
                'width': dark_group.width,
                'height': dark_group.height,
                'sigbits': dark_group.sigbits,
                'channel': dark_fs.channel,
            }
        elif groups:
            g0 = groups[0]
            results['camera'] = {'width': g0.width, 'height': g0.height, 'sigbits': g0.sigbits, 'channel': None}

        out = results_path(project)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        yield f'Analysis complete — results in {out.parent.name}/{out.name}'
        yield 'CHAR_EXIT 0'
    except Exception as err:  # a failed walk must still end the stream cleanly
        yield f'ERROR: {err}'
        yield 'CHAR_EXIT 1'
    finally:
        _char_lock.release()


# --- live exposure sweep (phase 2) -------------------------------------------
# A controlled exposure sweep at fixed gain(s) provides what existing captures
# cannot: enough well-spread PTC points for a reliable conversion-gain fit,
# plus linearity, full well, SNR and dynamic range. Per-pixel temporal
# statistics average linearly, so the sweep is valid on any *static* scene;
# a flat field is only required for uniformity metrics (PRNU).

_SWEEP_PROBE_START_US = 100
_SWEEP_EXPOSURE_CAP_US = 1_000_000  # 1 s: enough to saturate any sane scene


def _point_from_dict(entry: dict) -> PtcPoint:
    fields = ('mean_dn', 'var_dn2', 'exposure_us', 'gain', 'n_frames', 'clipped', 'label', 'source')
    return PtcPoint(**{k: entry[k] for k in fields})


def _sweep_frameset(camera, frames: int, exposure_us: int, gain: float, roi_fraction: float, label: str) -> FrameSet:
    d = camera.capture_raw_burst(frames, exposure_us, gain, roi_fraction)
    return FrameSet(
        frames=d['frames'],
        exposure_us=d['exposure_us'],
        gain=d['gain'],
        blacklevel_16=d['blacklevel_16'],
        sigbits=d['sigbits'],
        channel='Gr',
        label=label,
    )


def _find_saturation(camera, gain: float, roi_fraction: float) -> tuple[int, float]:
    """Double the exposure until the response saturates; return (exposure_us, swing).

    The probe uses two-frame bursts (cheap) and the black-subtracted mean, and
    stops on either signature of saturation: the mean crossing 85% of the DN
    swing (container-limited clipping), or the mean plateauing between
    doublings (electron full well, or a scene that cannot get brighter — points
    beyond that would sit at collapsed variance and poison the fit). Capped at
    1 s for scenes that never saturate at all.
    """
    exposure = _SWEEP_PROBE_START_US
    swing = None
    prev_mean = None
    while exposure <= _SWEEP_EXPOSURE_CAP_US:
        fs = _sweep_frameset(camera, 2, exposure, gain, roi_fraction, 'probe')
        swing = 65535.0 - fs.blacklevel_16
        ts = temporal_stats(fs)
        if ts is not None:
            if ts.mean_dn > 0.85 * swing:
                return exposure, swing
            if prev_mean is not None and prev_mean > 0.02 * swing and ts.mean_dn < prev_mean * 1.1:
                # Doubling the exposure barely moved the mean: response plateau.
                return exposure, swing
            prev_mean = ts.mean_dn
        exposure *= 2
    return _SWEEP_EXPOSURE_CAP_US, swing or 61695.0


def sweep_stream(
    project: Project,
    camera,
    gains: list[float],
    points_per_gain: int = 10,
    frames: int = 8,
    roi_fraction: float = 0.5,
) -> Iterator[str]:
    """Run a live exposure sweep and merge the metrics into results.json.

    Yields progress lines; ends with CHAR_EXIT <n>. Camera controls (AE, AWB,
    exposure, gain, frame-rate target) are restored afterwards.
    """
    from . import ctt_runner  # local import to avoid a cycle at module load

    if ctt_runner.is_running():
        yield 'ERROR: a calibration is running; wait for it to finish'
        yield 'CHAR_EXIT 2'
        return
    if not _char_lock.acquire(blocking=False):
        yield 'ERROR: a characterisation analysis is already running'
        yield 'CHAR_EXIT 2'
        return
    gains = sorted({round(float(g), 4) for g in gains}) or [1.0]
    points_per_gain = max(4, min(int(points_per_gain), 16))
    frames = max(2, min(int(frames), 16))
    prev = None
    try:
        yield (
            f'$ characterise (live sweep)  project={project.name}  '
            f'gains={",".join(str(g) for g in gains)}  points={points_per_gain}  frames={frames}'
        )
        yield 'Scene must stay static and steadily lit for the duration of the sweep.'
        prev = camera.get_controls()
        # Manual everything; unconstrained frame duration so long exposures land.
        camera.set_controls(
            {
                'auto_exposure': False,
                'exposure': prev.get('exposure') or 10_000,
                'gain': gains[0],
                'fps': 0,
                'awb': False,
            }
        )

        points: list[PtcPoint] = []
        gain_summaries = []
        for gain in gains:
            yield f'\tgain {gain:g}: probing for saturation'
            sat_exposure, swing = _find_saturation(camera, gain, roi_fraction)
            lo = max(camera.get_controls().get('exposure_min', 50), sat_exposure // 500)
            exposures = np.unique(np.geomspace(lo, sat_exposure, points_per_gain).astype(int)).tolist()
            exposures.append(min(int(sat_exposure * 1.3), _SWEEP_EXPOSURE_CAP_US))  # deliberate clip: full well
            yield f'\tgain {gain:g}: {len(exposures)} exposures, {lo} us to {exposures[-1]} us'
            family = []
            for exposure in exposures:
                fs = _sweep_frameset(camera, frames, exposure, gain, roi_fraction, f'sweep g{gain:g} {exposure}us')
                point = ptc_point(fs, source='live')
                if point is None:
                    continue
                family.append(point)
                state = 'clipped' if point.clipped else 'ok'
                yield f'\t\t{point.exposure_us} us: mean {point.mean_dn:.0f} DN16, var {point.var_dn2:.1f} ({state})'
            points.extend(family)

            fits = fit_ptc(family)
            fit = fits[0] if fits else None
            summary = {'gain': gain, 'n_points': len(family)}
            if fit is not None:
                summary.update(
                    {
                        'k_e_per_dn': round(fit.k_e_per_dn, 4),
                        'read_noise_dn': round(fit.read_noise_dn, 2) if fit.read_noise_dn else None,
                        'read_noise_e': round(fit.read_noise_e, 2) if fit.read_noise_e else None,
                        'r2': round(fit.r2, 5),
                        'reliable': fit.reliable,
                    }
                )
                rn = f'{fit.read_noise_e:.2f} e-' if fit.read_noise_e is not None else 'n/a (negative intercept)'
                yield (
                    f'\tgain {gain:g}: K {fit.k_e_per_dn:.3f} e-/DN16, read noise '
                    f'{rn} (r2 {fit.r2:.4f}{", reliable" if fit.reliable else ""})'
                )
            gain_summaries.append(summary)

        yield 'Merging sweep metrics into results'
        results = read_results(project)
        if results is None:
            results = {
                'version': RESULTS_VERSION,
                'inputs': [],
                'camera': None,
                'groups': [],
                'dark': {'available': False, 'unavailable_reason': 'no dark frames captured'},
                'ptc': {'points': [], 'fits': [], 'unavailable_reason': None},
                'prnu': {'available': False, 'groups': [], 'best_pct': None},
                'warnings': [],
            }
        results['generated_at'] = _now_iso()
        results['sweep'] = {
            'ran_at': results['generated_at'],
            'gains': gains,
            'points_per_gain': points_per_gain,
            'frames': frames,
        }

        # Live points replace any previous sweep; offline points are kept.
        kept = [p for p in results['ptc']['points'] if p.get('source') != 'live']
        results['ptc']['points'] = kept + [asdict(p) for p in points]
        all_points = [_point_from_dict(p) for p in results['ptc']['points']]
        fits = fit_ptc(all_points)
        results['ptc']['fits'] = [asdict(f) for f in fits]
        reliable_by_gain = {f.gain: f for f in fits if f.reliable}
        if reliable_by_gain:
            results['ptc']['unavailable_reason'] = None
        else:
            n_clipped = sum(1 for p in points if p.clipped)
            results['ptc']['unavailable_reason'] = (
                f'sweep completed but no reliable fit — {n_clipped} of {len(points)} points were clipped. '
                'A flat, evenly lit target (no bright highlights) keeps points usable across the sweep.'
            )
        results['gain_sweep'] = {'available': len(gains) > 1, 'gains': gain_summaries}

        base_gain = gains[0]
        base_fit = reliable_by_gain.get(base_gain)
        live_base = [p for p in points if p.gain == base_gain]
        lin = linearity(live_base)
        results['linearity'] = {'available': lin is not None, **(lin or {})}
        fw = full_well(live_base)
        if fw is not None and base_fit is not None:
            fw['full_well_e'] = round(fw['full_well_dn'] * base_fit.k_e_per_dn, 0)
        results['full_well'] = {'available': fw is not None, **(fw or {})}
        results['snr_curve'] = snr_curve(live_base)

        # Electron-referred dark metrics, and dynamic range, once K is known.
        dark = results.get('dark', {})
        if dark.get('available') and reliable_by_gain:
            k_dark = reliable_by_gain.get(round(dark.get('gain', 1.0), 4)) or base_fit
            if k_dark is not None:
                if dark.get('read_noise_dn') is not None:
                    dark['read_noise_e'] = round(dark['read_noise_dn'] * k_dark.k_e_per_dn, 2)
                if dark.get('dsnu_dn') is not None:
                    dark['dsnu_e'] = round(dark['dsnu_dn'] * k_dark.k_e_per_dn, 2)
        read_noise_e = (dark.get('read_noise_e') if dark.get('available') else None) or (
            base_fit.read_noise_e if base_fit else None
        )
        if fw is not None and fw.get('full_well_e') and read_noise_e:
            results['dynamic_range'] = {
                'available': True,
                'db': round(dynamic_range(fw['full_well_e'], read_noise_e), 1),
                'full_well_e': fw['full_well_e'],
                'read_noise_e': read_noise_e,
            }
        else:
            results['dynamic_range'] = {'available': False}

        out = results_path(project)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        yield f'Sweep complete — results in {out.parent.name}/{out.name}'
        yield 'CHAR_EXIT 0'
    except Exception as err:  # a failed sweep must still end the stream cleanly
        yield f'ERROR: {err}'
        yield 'CHAR_EXIT 1'
    finally:
        if prev is not None:
            # The camera may have gone away mid-sweep; restoring is best-effort.
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
        _char_lock.release()
