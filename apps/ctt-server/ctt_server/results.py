# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Parse a CTT output tuning file into data the UI can chart with Chart.js,
# without invoking CTT's interactive matplotlib plots.
#
# Output file shape (ctt.core.camera.Camera.write_json):
#   {"version": 2.0, "target": "pisp"|"bcm2835", "algorithms": [{name: data}, ...]}

import json
from pathlib import Path

# ISP grid dimensions (cols, rows) keyed by tuning-file target.
_GRID = {'pisp': (32, 32), 'bcm2835': (16, 12)}


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


def parse_tuning_file(path: str | Path) -> dict:
    """Return a JSON-serialisable summary + chart data for a tuning file."""
    data = json.loads(Path(path).read_text())
    target = data.get('target', '')
    algos = _algorithms(data)

    summary = {
        'target': target,
        'version': data.get('version'),
        'algorithms': sorted(algos.keys()),
    }

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
