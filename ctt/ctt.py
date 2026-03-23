#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Camera tuning tool - main entry point

import argparse
import json
import logging
import sys
from pathlib import Path

from .algorithms.alsc import AlscCalibration
from .algorithms.awb import AwbCalibration
from .algorithms.cac import CacCalibration
from .algorithms.ccm import CcmCalibration
from .algorithms.geq import GeqCalibration
from .algorithms.lux import LuxCalibration
from .algorithms.noise import NoiseCalibration
from .core.camera import Camera
from .output.converter import convert_v2
from .output.json_formatter import pretty_print
from .platforms import pisp as pisp_mod
from .platforms import vc4 as vc4_mod
from .platforms.base import PlatformConfig
from .utils.errors import ArgError

logger = logging.getLogger(__name__)

# Optional ANSI styling (disabled when not a TTY)
_USE_COLOR = sys.stdout.isatty()
_RESET = '\033[0m' if _USE_COLOR else ''
_DIM = '\033[2m' if _USE_COLOR else ''
_GREEN = '\033[32m' if _USE_COLOR else ''
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


def get_platform(target_name: str) -> object:
    if target_name == 'pisp':
        return pisp_mod.get_config()
    elif target_name == 'vc4':
        return vc4_mod.get_config()
    else:
        raise ArgError(f'\n\nError: Unknown target platform: {target_name}')


def get_target_from_tuning_file(path: str) -> str:
    """Read target (pisp or vc4) from an existing tuning JSON. Used with --update."""
    with open(path) as f:
        data = json.load(f)
    t = data.get('target', '')
    if t == 'pisp':
        return 'pisp'
    if t == 'bcm2835':
        return 'vc4'
    raise ArgError(f'\n\nError: Tuning file target must be "pisp" or "bcm2835", got: {t!r}')


def load_template(platform: object, template_path: str | None = None, update_path: str | None = None) -> dict:
    if template_path is not None:
        with open(template_path) as f:
            return json.load(f)
    if update_path is not None:
        with open(update_path) as f:
            in_json = json.load(f)
        tmpl = {}
        for algo in in_json.get('algorithms', []):
            for name, data in algo.items():
                tmpl[name] = data
        return tmpl
    with open(str(platform.default_template)) as f:
        return json.load(f)


def run_ctt(
    output_dir: Path,
    name: str,
    directory: str,
    config: str | None,
    json_template: dict,
    grid_size: tuple[int, int],
    target: str,
    alsc_only: bool = False,
    colour_only: bool = False,
    output_json_path: str | None = None,
    plot_cli: list[str] | None = None,
) -> None:
    if output_json_path is not None:
        json_output = output_json_path
        log_output = str(Path(output_json_path).with_suffix('.log'))
    else:
        json_output = str(output_dir / f'{name}_{target}.json')
        log_output = str(output_dir / f'{name}_{target}.log')
    if config is not None:
        if Path(config).suffix != '.json':
            raise ArgError('\n\nError: Config file must be a json file!')
        try:
            with open(config) as config_json:
                configs = json.load(config_json)
        except FileNotFoundError:
            configs = {}
            config = False
        except json.decoder.JSONDecodeError:
            configs = {}
            config = True

    else:
        configs = {}

    disable = configs.get('disable', [])

    def _normalise_plot_name(s: str) -> str:
        return s if s.startswith('rpi.') else f'rpi.{s}'

    plot = [_normalise_plot_name(p) for p in configs.get('plot', [])]
    if plot_cli:
        for item in plot_cli:
            plot.extend(_normalise_plot_name(a.strip()) for a in item.split(',') if a.strip())
        plot = list(dict.fromkeys(plot))  # dedupe, preserve order
    awb_d = configs.get('awb', {})
    greyworld = int(bool(awb_d.get('greyworld', 0)))
    alsc_d = configs.get('alsc', {})
    do_alsc_colour = int(bool(alsc_d.get('do_alsc_colour', 1)))
    luminance_strength = alsc_d.get('luminance_strength', 0.8)
    lsc_max_gain = alsc_d.get('max_gain', 8.0)
    blacklevel = configs.get('blacklevel', -1)
    macbeth_d = configs.get('macbeth', {})
    mac_small = int(bool(macbeth_d.get('small', 0)))
    mac_show = int(bool(macbeth_d.get('show', 0)))
    mac_config = (mac_small, mac_show)
    ccm_d = configs.get('ccm', {})
    ccm_matrix_selection = ccm_d.get('matrix_selection', 'average')
    ccm_matrix_selection_types = ccm_d.get('matrix_selection_types', ['average', 'maximum', 'patches'])
    ccm_test_patches = ccm_d.get('test_patches', [1, 2, 5, 8, 9, 12, 14])

    if blacklevel < -1 or blacklevel >= 2**16:
        logger.warning('\nInvalid blacklevel, defaulted to 64')
        blacklevel = -1

    if luminance_strength < 0 or luminance_strength > 1:
        logger.warning('\nInvalid luminance_strength strength, defaulted to 0.5')
        luminance_strength = 0.5

    if directory[-1] != '/':
        directory += '/'

    # Print config/options summary at start (Run vs Options, stable layout)
    config_src = 'default' if config is None or config is True or config is False else config
    mode = 'ALSC only' if alsc_only else ('AWB + CCM only' if colour_only else 'Full')
    disable_str = ', '.join(sorted(disable)) if disable else '(none)'
    plot_str = ', '.join(plot) if plot else '(none)'
    summary_lines = [
        f'{_CYAN}Run{_RESET}',
        f'  Input   {directory}',
        f'  Output  {json_output}',
        f'  Config  {config_src}   Target  {target}   Mode  {mode}',
        '',
        f'{_CYAN}Options{_RESET}',
        f'  ALSC    do_alsc_colour={do_alsc_colour}  luminance_strength={luminance_strength}  max_gain={lsc_max_gain}',
        f'  AWB     greyworld={greyworld}   Blacklevel  {blacklevel}   Macbeth  small={mac_small}  show={mac_show}',
        f'  CCM     matrix_selection={ccm_matrix_selection}  test_patches={ccm_test_patches}',
        f'  Disable {disable_str}',
        f'  Plot    {plot_str}',
    ]
    print('\n' + '\n'.join(summary_lines) + '\n', flush=True)

    try:
        cam = Camera(json_output, json=json_template)
        cam.output_dir = output_dir
        cam.log_user_input(json_output, directory, config, log_output)
        cam.add_imgs(directory, mac_config, blacklevel)
        # Infer ALSC-only when only ALSC images present (e.g. mono LSC-only from DNGs).
        if len(cam.imgs) == 0 and len(cam.imgs_cac) == 0 and len(cam.imgs_alsc) > 0:
            alsc_only = True
        if alsc_only:
            disable = set(cam.json.keys()).symmetric_difference({'rpi.alsc'})
        elif colour_only:
            disable = set(cam.json.keys()).symmetric_difference({'rpi.awb', 'rpi.ccm'})
        cam.disable = disable
        cam.plot = plot
    except FileNotFoundError as err:
        raise ArgError('\n\nError: Input image directory not found!') from err

    # Build platform object for algorithms
    # Note: default_template not needed as json_template is already loaded
    platform = PlatformConfig(target, grid_size, None)

    if cam.check_imgs(macbeth=not alsc_only):
        # Mono (greyscale) sensors: disable AWB and CCM in output; they are not applicable.
        if cam.mono:
            logger.info('Mono sensor: disabling AWB and CCM in output.')
            disable = set(disable) | {'rpi.awb', 'rpi.ccm'}
            cam.disable = disable
        if not alsc_only:
            cam.json['rpi.black_level']['black_level'] = cam.blacklevel_16
        # When updating a file in-place with --alsc-only or --colour-only, keep all
        # other algorithm sections unchanged; only skip removing them from the output.
        if not (output_json_path and (alsc_only or colour_only)):
            cam.json_remove(disable)
        print(f'\n{_CYAN}Starting calibrations{_RESET}', flush=True)
        algorithms = [
            AlscCalibration(cam, platform, luminance_strength, do_alsc_colour, lsc_max_gain),
            GeqCalibration(cam, platform),
            LuxCalibration(cam, platform),
            NoiseCalibration(cam, platform),
        ]
        if 'rpi.cac' in json_template:
            algorithms.append(CacCalibration(cam, platform, do_alsc_colour))
        algorithms += [
            AwbCalibration(cam, platform, greyworld, do_alsc_colour),
            CcmCalibration(
                cam,
                platform,
                do_alsc_colour,
                ccm_matrix_selection,
                ccm_test_patches,
                ccm_matrix_selection_types,
            ),
        ]

        for algo in algorithms:
            if algo.json_key not in cam.disable:
                print(f'\t{algo.json_key.replace("rpi.", "").upper()}', flush=True)
                result = algo.run()
                if result is not None:
                    cam.json[algo.json_key].update(result)

        # PiSP-only: populate rpi.nn.awb with AWB curve and CCM at 5000 K
        if platform.target == 'pisp' and 'rpi.nn.awb' in cam.json:
            if 'rpi.awb' not in cam.disable and 'rpi.awb' in cam.json:
                awb_data = cam.json['rpi.awb']
                for key in ('ct_curve', 'transverse_pos', 'transverse_neg'):
                    if key in awb_data:
                        cam.json['rpi.nn.awb'][key] = awb_data[key]
            if 'rpi.ccm' not in cam.disable and 'rpi.ccm' in cam.json:
                ccms = cam.json['rpi.ccm'].get('ccms')
                if ccms:
                    cam.json['rpi.nn.awb']['ccm'] = _interpolate_ccm(ccms, 5000)

        print('', flush=True)
        cam.write_json(target=target, grid_size=grid_size)
        cam.write_log(log_output)
        # Final summary
        mode = 'ALSC only' if alsc_only else ('AWB + CCM only' if colour_only else 'Full')
        n_macbeth = len(cam.imgs)
        n_alsc = len(cam.imgs_alsc)
        n_cac = len(cam.imgs_cac)
        counts = f'Macbeth: {n_macbeth}  ALSC: {n_alsc}  CAC: {n_cac}'
        lines = [
            f'{_GREEN}Calibration complete{_RESET}',
            f'  Output  {_CYAN}{json_output}{_RESET}',
            f'  Log     {_DIM}{log_output}{_RESET}',
            f'  Target  {target}  ·  Mode  {mode}',
            f'  Images  {counts}',
        ]
        print('\n' + '\n'.join(lines), flush=True)
    else:
        cam.write_log(log_output)


def _interpolate_ccm(ccms: list[dict], target_ct: int) -> list[float]:
    """Return a CCM (9-element list) interpolated from ccms at target_ct. Clamps at endpoints."""
    cts = [e['ct'] for e in ccms]
    mats = [e['ccm'] for e in ccms]
    if target_ct <= cts[0]:
        return mats[0]
    if target_ct >= cts[-1]:
        return mats[-1]
    for i in range(len(cts) - 1):
        if cts[i] <= target_ct <= cts[i + 1]:
            t = (target_ct - cts[i]) / (cts[i + 1] - cts[i])
            return [round(a + t * (b - a), 5) for a, b in zip(mats[i], mats[i + 1], strict=True)]
    return mats[-1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
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

    for target in targets:
        platform = get_platform(target)
        json_template = load_template(platform, args.template, args.update)
        run_ctt(
            output_dir,
            name,
            args.input,
            args.config,
            json_template,
            platform.grid_size,
            target,
            alsc_only=args.alsc_only,
            colour_only=args.colour_only,
            output_json_path=args.update,
            plot_cli=args.plot,
        )
