# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Camera tuning tool - core calibration pipeline.
#
# This is the shared library that both the CLI (ctt.ctt) and the web server
# (ctt_server.ctt_runner) call into. It owns platform/template resolution and
# the calibration run itself; the callers only supply inputs and consume the
# progress emitted through the `ctt` logger.

import json
import logging
from pathlib import Path

from ..algorithms.alsc import AlscCalibration
from ..algorithms.awb import AwbCalibration
from ..algorithms.cac import CacCalibration
from ..algorithms.ccm import CcmCalibration
from ..algorithms.geq import GeqCalibration
from ..algorithms.lux import LuxCalibration
from ..algorithms.noise import NoiseCalibration
from ..platforms import pisp as pisp_mod
from ..platforms import vc4 as vc4_mod
from ..platforms.base import PlatformConfig
from ..utils.errors import ArgError
from .camera import Camera

logger = logging.getLogger(__name__)


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
        'Run',
        f'  Input   {directory}',
        f'  Output  {json_output}',
        f'  Config  {config_src}   Target  {target}   Mode  {mode}',
        '',
        'Options',
        f'  ALSC    do_alsc_colour={do_alsc_colour}  luminance_strength={luminance_strength}  max_gain={lsc_max_gain}',
        f'  AWB     greyworld={greyworld}   Blacklevel  {blacklevel}   Macbeth  small={mac_small}  show={mac_show}',
        f'  CCM     matrix_selection={ccm_matrix_selection}  test_patches={ccm_test_patches}',
        f'  Disable {disable_str}',
        f'  Plot    {plot_str}',
    ]
    logger.info('\n' + '\n'.join(summary_lines) + '\n')

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
        logger.info('\nStarting calibrations')
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
                logger.info(f'\t{algo.json_key.replace("rpi.", "").upper()}')
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

        logger.info('')
        cam.write_json(target=target, grid_size=grid_size)
        cam.write_log(log_output)
        # Final summary
        mode = 'ALSC only' if alsc_only else ('AWB + CCM only' if colour_only else 'Full')
        n_macbeth = len(cam.imgs)
        n_alsc = len(cam.imgs_alsc)
        n_cac = len(cam.imgs_cac)
        counts = f'Macbeth: {n_macbeth}  ALSC: {n_alsc}  CAC: {n_cac}'
        lines = [
            'Calibration complete',
            f'  Output  {json_output}',
            f'  Log     {log_output}',
            f'  Target  {target}  ·  Mode  {mode}',
            f'  Images  {counts}',
        ]
        logger.info('\n' + '\n'.join(lines))
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


def run_ctt_targets(
    output_dir: Path,
    name: str,
    directory: str,
    config: str | None,
    targets: list[str],
    alsc_only: bool = False,
    colour_only: bool = False,
    template_path: str | None = None,
    update_path: str | None = None,
    plot_cli: list[str] | None = None,
) -> None:
    """Run a calibration for each target, resolving platform + template per target.

    The single entry point shared by the CLI and the web server: both supply
    inputs and let this drive the per-target loop. When ``update_path`` is set
    (CLI ``--update``) it is the template source *and* the in-place output file.
    """
    for target in targets:
        platform = get_platform(target)
        json_template = load_template(platform, template_path, update_path)
        run_ctt(
            output_dir,
            name,
            directory,
            config,
            json_template,
            platform.grid_size,
            target,
            alsc_only=alsc_only,
            colour_only=colour_only,
            output_json_path=update_path,
            plot_cli=plot_cli,
        )
