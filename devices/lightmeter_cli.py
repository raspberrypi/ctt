# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Device-agnostic light-meter CLI (console script `ctt-lightmeter`). Works against any
# registered devices.LightMeter driver via the get_lightmeter() factory.

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .lightmeter import LightmeterError, Measurement
from .registry import get_lightmeter


def _format_reading(m: Measurement) -> str:
    """A one-line human summary: illuminance and colour temperature."""
    parts = [f'{m.illuminance_lux:.1f} lx', f'{m.cct:.0f} K']
    if m.duv is not None:
        parts.append(f'Δuv {m.duv:+.4f}')
    if m.cie1931_xy is not None:
        parts.append(f'xy ({m.cie1931_xy[0]:.4f}, {m.cie1931_xy[1]:.4f})')
    if m.cri_ra is not None:
        parts.append(f'Ra {m.cri_ra:.1f}')
    return '  '.join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='ctt-lightmeter', description='Read a light meter (e.g. CL-70F).')
    parser.add_argument('--serial', help='target a specific device by serial number')
    parser.add_argument('-v', '--verbose', action='store_true')
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('probe', help='find the light meter and show its identity')

    p_measure = sub.add_parser('measure', help='take a single measurement')
    p_measure.add_argument('--json', action='store_true', help='print the full reading as JSON')

    p_sample = sub.add_parser('sample', help='take measurements periodically')
    p_sample.add_argument('--interval', type=float, default=5.0, help='seconds between samples (default 5)')
    p_sample.add_argument('--count', type=int, default=0, help='number of samples (0 = until interrupted)')
    p_sample.add_argument('--json', action='store_true', help='print each reading as JSON')

    sub.add_parser('calibrate', help='run a dark calibration')

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(message)s')

    if args.cmd is None:
        parser.print_help()
        return 0

    try:
        meter = get_lightmeter(args.serial)
    except LightmeterError as err:
        print(f'error: {err}', file=sys.stderr)
        return 1

    try:
        if args.cmd == 'probe':
            print(f'{meter.model} (serial {meter.serial or "?"})')
        elif args.cmd == 'measure':
            reading = meter.measure()
            print(json.dumps(reading.to_dict()) if args.json else _format_reading(reading))
        elif args.cmd == 'sample':
            _sample(meter, args.interval, args.count, args.json)
        elif args.cmd == 'calibrate':
            meter.calibrate()
            print('dark calibration complete')
    except LightmeterError as err:
        print(f'error: {err}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        meter.close()
    return 0


def _sample(meter, interval: float, count: int, as_json: bool) -> None:
    """Print a reading every `interval` seconds, up to `count` (0 = forever)."""
    taken = 0
    while count == 0 or taken < count:
        reading = meter.measure()
        if as_json:
            print(json.dumps(reading.to_dict()), flush=True)
        else:
            print(f'{time.strftime("%H:%M:%S")}  {_format_reading(reading)}', flush=True)
        taken += 1
        if count == 0 or taken < count:
            time.sleep(interval)


if __name__ == '__main__':
    raise SystemExit(main())
