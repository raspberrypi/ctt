# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Parse a CTT output tuning file into data the UI can chart with Chart.js,
# without invoking CTT's interactive matplotlib plots.
#
# Output file shape (ctt.core.camera.Camera.write_json):
#   {"version": 2.0, "target": "pisp"|"bcm2835", "algorithms": [{name: data}, ...]}

import contextlib
import json
import math
from pathlib import Path

# ISP grid dimensions (cols, rows) keyed by tuning-file target.
_GRID = {'pisp': (32, 32), 'bcm2835': (16, 12)}

# Delta E (CIE2000) quality bands. Rough rules of thumb for ΔE₀₀: ~1 is a
# just-noticeable difference, so ≤2 reads as good and ≤4 as acceptable.
_DELTAE_GOOD = 2.0
_DELTAE_FAIR = 4.0


def _deltae_band(value: float | None) -> str | None:
    if value is None:
        return None
    if value <= _DELTAE_GOOD:
        return 'good'
    if value <= _DELTAE_FAIR:
        return 'fair'
    return 'poor'


def _ccm_quality(ccm_list: list) -> dict:
    """Summarise CCM delta E into a headline mean/worst + a quality band.

    The band is driven by the *mean* per-patch delta E, not the worst patch: a
    few out-of-gamut patches (saturated blues/cyans) sit high on any good 3x3
    calibration, so worst-case would peg every result to 'poor'. Worst is still
    reported as a secondary figure.
    """
    entries = [c for c in ccm_list if isinstance(c, dict)]
    all_de = [p['de'] for c in entries for p in c.get('patches', []) if 'de' in p]
    all_norm = [p['de_norm'] for c in entries for p in c.get('patches', []) if 'de_norm' in p]
    if all_de:
        mean = sum(all_de) / len(all_de)
        worst = max(all_de)
        # Brightness-normalised "colour" ΔE: the honest hue/saturation figure the
        # band is graded on (falls back to raw mean for sidecars without it).
        colour = (sum(all_norm) / len(all_norm)) if all_norm else mean
    else:
        # Older sidecars without per-patch data: fall back to the per-CT figures.
        metrics = [c['metric_after'] for c in entries if c.get('metric_after') is not None]
        maxes = [c['max_after'] for c in entries if c.get('max_after') is not None]
        if not metrics and not maxes:
            return {}
        mean = (sum(metrics) / len(metrics)) if metrics else (sum(maxes) / len(maxes))
        worst = max(maxes) if maxes else mean
        colour = mean
    return {'mean': mean, 'worst': worst, 'colour': colour, 'band': _deltae_band(colour)}


def _alsc_falloff(grid: list) -> dict:
    """Centre vs corner luminance gain from the ALSC grid (gains rise toward edges)."""
    if not grid or not grid[0]:
        return {}
    rows, cols = len(grid), len(grid[0])
    centre = grid[rows // 2][cols // 2]
    corners = [grid[0][0], grid[0][-1], grid[-1][0], grid[-1][-1]]
    return {'centre': centre, 'corner_max': max(corners)}


def _algorithms(data: dict) -> dict:
    """Flatten the algorithms list into a {name: data} mapping."""
    out: dict = {}
    for entry in data.get('algorithms', []):
        if isinstance(entry, dict):
            out.update(entry)
    return out


def _awb_points(ct_curve: list) -> list[dict]:
    """ct_curve is a flat [ct, r, b, ct, r, b, ...] list (increasing CT)."""
    points = []
    for i in range(0, len(ct_curve) - 2, 3):
        points.append({'ct': ct_curve[i], 'r': ct_curve[i + 1], 'b': ct_curve[i + 2]})
    return points


def _alsc_grid(luminance_lut: list, target: str) -> dict:
    cols, rows = _GRID.get(target, (0, 0))
    if cols and rows and len(luminance_lut) == cols * rows:
        grid = [luminance_lut[r * cols : (r + 1) * cols] for r in range(rows)]
    else:
        grid = [luminance_lut]  # fall back to a single row if dims don't match
    return {
        'cols': cols,
        'rows': rows,
        'grid': grid,
        'min': min(luminance_lut) if luminance_lut else None,
        'max': max(luminance_lut) if luminance_lut else None,
    }


def _load_metrics(metrics_path: str | Path | None) -> dict:
    """Read the metrics sidecar, returning {} if absent or unreadable."""
    if metrics_path is None:
        return {}
    p = Path(metrics_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _summary_charts(data: dict) -> dict:
    """Build {summary, charts} (coefficient views) from a parsed tuning-file dict."""
    target = data.get('target', '')
    algos = _algorithms(data)
    summary = {'target': target, 'version': data.get('version'), 'algorithms': sorted(algos.keys())}
    charts: dict = {}

    awb = algos.get('rpi.awb', {})
    if isinstance(awb, dict) and awb.get('ct_curve'):
        charts['awb'] = {'points': _awb_points(awb['ct_curve'])}

    ccm = algos.get('rpi.ccm', {})
    if isinstance(ccm, dict) and ccm.get('ccms'):
        charts['ccm'] = {
            'matrices': [{'ct': m.get('ct'), 'ccm': m.get('ccm', [])} for m in ccm['ccms'] if isinstance(m, dict)]
        }

    alsc = algos.get('rpi.alsc', {})
    if isinstance(alsc, dict) and alsc.get('luminance_lut'):
        charts['alsc'] = _alsc_grid(alsc['luminance_lut'], target)
        charts['alsc'].update(_alsc_falloff(charts['alsc']['grid']))

    lux = algos.get('rpi.lux', {})
    if isinstance(lux, dict):
        charts['lux'] = {k: lux[k] for k in ('reference_shutter_speed', 'reference_gain', 'reference_lux') if k in lux}

    noise = algos.get('rpi.noise', {})
    if isinstance(noise, dict):
        charts['noise'] = {k: noise[k] for k in ('reference_constant', 'reference_slope') if k in noise}

    black = algos.get('rpi.black_level', {})
    if isinstance(black, dict) and 'black_level' in black:
        summary['black_level'] = black['black_level']

    return {'summary': summary, 'charts': charts}


def _finite(obj):
    """Recursively replace non-finite floats (NaN/Inf) with None.

    A degenerate calibration fit can leave a NaN/Inf in the tuning file (e.g. the
    noise slope on under-exposed captures). Python's json emits these as the bare
    tokens NaN/Infinity, which are invalid JSON: the browser's response.json()
    then throws and the whole Results page fails to render. Sanitising here keeps
    the payload valid JSON, so the page renders and the bad value just shows blank.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


def parse_tuning_file(path: str | Path, metrics_path: str | Path | None = None) -> dict:
    """Return a JSON-serialisable summary + chart data for a tuning file.

    ``metrics_path`` is the optional structured sidecar written during the run;
    when present, its calibration-quality metrics are merged into the result, plus
    a parsed view + re-evaluated ΔE for the built-in default tuning (for new-vs-old).
    """
    result = _summary_charts(json.loads(Path(path).read_text()))

    raw_metrics = _load_metrics(metrics_path)
    if raw_metrics:
        ccm_list = raw_metrics.get('ccm', [])
        ccm_default = raw_metrics.get('ccm_default', [])
        metrics = {
            'ccm': sorted(ccm_list, key=lambda c: c.get('ct', 0)),
            'ccm_quality': _ccm_quality(ccm_list),
            'warnings': raw_metrics.get('warnings', []),
            'counts': raw_metrics.get('counts', {}),
            'coverage': raw_metrics.get('coverage', {}),
            'config': raw_metrics.get('config', {}),
            'lux': raw_metrics.get('lux', {}),
        }
        if ccm_default:
            metrics['ccm_default'] = sorted(ccm_default, key=lambda c: c.get('ct', 0))
            metrics['ccm_default_quality'] = _ccm_quality(ccm_default)
        # Parsed coefficient view of the built-in default tuning (the "old" side).
        default_path = raw_metrics.get('default_tuning_path')
        if default_path and Path(default_path).exists():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                metrics['default'] = _summary_charts(json.loads(Path(default_path).read_text()))
        result['metrics'] = metrics

    sanitised = _finite(result)
    assert isinstance(sanitised, dict)
    return sanitised
