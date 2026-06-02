#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Camera tuning tool - command-line entry point.
#
# A thin CLI shell around the shared pipeline in ctt.core.runner: argument
# parsing, terminal colouring, and the one-shot --convert / --prettify
# utilities. All calibration work happens in ctt.core.runner.

import argparse
import json
import logging
import sys
from pathlib import Path

from .core.runner import get_platform, get_target_from_tuning_file, run_ctt_targets
from .output.converter import convert_v2
from .output.json_formatter import pretty_print

# Optional ANSI styling (disabled when not a TTY)
_USE_COLOR = sys.stdout.isatty()
_RESET = '\033[0m' if _USE_COLOR else ''
_YELLOW = '\033[33m' if _USE_COLOR else ''
_RED = '\033[31m' if _USE_COLOR else ''
_CYAN = '\033[36m' if _USE_COLOR else ''


class _ConsoleFormatter(logging.Formatter):
    """Format log records with level-based color when stdout is a TTY (no icon prefix)."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if _USE_COLOR:
            if record.levelno >= logging.ERROR:
                return f'{_RED}{msg}{_RESET}'
            if record.levelno >= logging.WARNING:
                return f'{_YELLOW}{msg}{_RESET}'
            if 'Loading images from' in msg:
                return f'{_CYAN}{msg}{_RESET}'
            return msg
        return msg


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    for h in logging.root.handlers:
        h.setFormatter(_ConsoleFormatter())

    parser = argparse.ArgumentParser(
        prog='ctt',
        description='Raspberry Pi Camera Tuning Tool',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Mode flags (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--alsc-only', action='store_true', help='Run only ALSC calibration')
    mode_group.add_argument('--colour-only', action='store_true', help='Run only AWB and CCM calibrations')
    mode_group.add_argument('--convert', action='store_true', help='Convert tuning file between vc4 <-> pisp')
    mode_group.add_argument('--prettify', action='store_true', help='Prettify an existing tuning file')

    # Common options
    parser.add_argument(
        '-t',
        '--target',
        action='append',
        metavar='TARGET',
        help='Target platform(s): pisp and/or vc4 (e.g. -t pisp -t vc4 or -t pisp,vc4). Default: both.',
    )

    # Calibration options
    parser.add_argument('-i', '--input', type=str, help='Calibration image directory')
    parser.add_argument('-o', '--output', type=str, help='Output directory (default: current directory)')
    parser.add_argument('--name', type=str, help='Base name for output files (default: derived from input directory)')
    parser.add_argument('-c', '--config', type=str, help='Configuration file')
    parser.add_argument('--template', type=str, help='Custom template JSON file')
    parser.add_argument('--update', type=str, help='Existing tuning file to update')
    parser.add_argument(
        '--plot',
        action='append',
        metavar='ALGO',
        help='Show matplotlib debug plot for algorithm (e.g. awb, alsc, geq, noise or rpi.awb). '
        'Can be repeated or comma-separated.',
    )

    # Positional args for --convert and --prettify
    parser.add_argument('positional', nargs='*', help='Positional args for --convert/--prettify modes')

    args = parser.parse_args()

    # Normalise -t to list of 'pisp' | 'vc4' (allow -t pisp -t vc4 or -t pisp,vc4; default both)
    def _targets(raw: list[str] | None) -> list[str]:
        if not raw:
            return ['pisp', 'vc4']
        expanded = [t.strip() for s in raw for t in s.split(',') if t.strip()]
        valid = {'pisp', 'vc4'}
        for t in expanded:
            if t not in valid:
                parser.error(f'--target must be pisp and/or vc4, got: {t!r}')
        return list(dict.fromkeys(expanded))  # dedupe, preserve order

    if args.update is not None and args.target is not None:
        parser.error('--target (-t) cannot be used with --update; target is taken from the tuning file')
    targets = _targets(args.target)
    if args.update is not None:
        targets = [get_target_from_tuning_file(args.update)]

    if args.convert:
        if len(args.positional) < 1:
            parser.error('--convert requires at least one input file')
        if len(targets) != 1:
            parser.error('--convert requires exactly one target; use -t pisp or -t vc4')
        input_file = args.positional[0]
        output_file = args.positional[1] if len(args.positional) > 1 else None
        target = 'bcm2835' if targets[0] == 'vc4' else targets[0]

        with open(input_file) as f:
            in_json = json.load(f)

        out_json = convert_v2(in_json, target)

        with open(output_file if output_file is not None else input_file, 'w') as f:
            f.write(out_json)
        return

    if args.prettify:
        if len(args.positional) < 1:
            parser.error('--prettify requires at least one input file')
        input_file = args.positional[0]
        output_file = args.positional[1] if len(args.positional) > 1 else None
        prettify_target = targets[0] if len(targets) == 1 else 'vc4'

        with open(input_file) as f:
            in_json = json.load(f)

        platform = get_platform(prettify_target)
        grid_size = platform.grid_size

        out_json = pretty_print(in_json, custom_elems={'table': grid_size[0], 'luminance_lut': grid_size[0]})

        with open(output_file if output_file is not None else input_file, 'w') as f:
            f.write(out_json)
        return

    # Calibration mode
    if args.input is None:
        parser.error('Calibration mode requires -i/--input')

    output_dir = Path(args.output) if args.output else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name if args.name else Path(args.input).resolve().name

    run_ctt_targets(
        output_dir,
        name,
        args.input,
        args.config,
        targets,
        alsc_only=args.alsc_only,
        colour_only=args.colour_only,
        template_path=args.template,
        update_path=args.update,
        plot_cli=args.plot,
    )
