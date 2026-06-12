# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Device-agnostic lightbox CLI (console script `ctt-lightbox`). Works against any
# registered ctt.devices.Lightbox driver via the get_lightbox() factory.

from __future__ import annotations

import argparse
import logging
import sys

from .lightbox import LightboxError
from .registry import get_lightbox


def _print_status(box) -> None:
    info = box.info()
    name = info['illuminant'] or '—'
    print(f'{info["model"]} (serial {info["serial"] or "?"})')
    print(f'channel {info["channel"]} ({name}) at {info["intensity"]:.0f}%')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='ctt-lightbox', description='Control a lightbox (e.g. lightSTUDIO-S).')
    parser.add_argument('--serial', help='target a specific device by serial number')
    parser.add_argument('-v', '--verbose', action='store_true')
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('probe', help='find the lightbox and list its illuminants')
    sub.add_parser('status', help='show current channel + intensity')
    sub.add_parser('get', help="read the active channel's intensity")

    p_set = sub.add_parser('set', help='set an illuminant to an intensity (0-100 %%)')
    p_set.add_argument('illuminant', help='illuminant name (e.g. D65) or channel number')
    p_set.add_argument('percent', type=float, help='intensity 0-100')

    p_chan = sub.add_parser('channel', help='switch to an illuminant at its default intensity')
    p_chan.add_argument('illuminant', help='illuminant name (e.g. D65) or channel number')

    p_illum = sub.add_parser('illuminant', help='switch to a named illuminant (e.g. D65)')
    p_illum.add_argument('name', help='illuminant name or channel number')
    p_illum.add_argument('percent', type=float, nargs='?', default=None, help='optional intensity 0-100')

    sub.add_parser('off', help='turn the active channel off')

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(message)s')

    if args.cmd is None:
        parser.print_help()
        return 0

    try:
        box = get_lightbox(args.serial)
    except LightboxError as err:
        print(f'error: {err}', file=sys.stderr)
        return 1

    try:
        if args.cmd == 'probe':
            print(f'{box.model} (serial {box.serial or "?"})')
            print('illuminants:')
            labels = box.illuminant_labels
            for channel, name in box.illuminants.items():
                label = labels.get(channel, '')
                extra = f'  ({label})' if label and label != name else ''
                print(f'  {channel}: {name}{extra}')
        elif args.cmd == 'status':
            _print_status(box)
        elif args.cmd == 'get':
            state = box.get_state()
            print(f'{state.illuminant or state.channel} at {state.intensity:.0f}%')
        elif args.cmd == 'set':
            box.set_illuminant(args.illuminant, args.percent)
            _print_status(box)
        elif args.cmd == 'channel':
            box.set_illuminant(args.illuminant)
            _print_status(box)
        elif args.cmd == 'illuminant':
            box.set_illuminant(args.name, args.percent)
            _print_status(box)
        elif args.cmd == 'off':
            box.off()
            _print_status(box)
    except LightboxError as err:
        print(f'error: {err}', file=sys.stderr)
        return 1
    finally:
        box.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
