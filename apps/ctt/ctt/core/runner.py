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
from ..algorithms.black_level import BlackLevelCalibration
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
    black_level_only: bool = False,
    output_json_path: str | None = None,
    plot_cli: list[str] | None = None,
    images: list[str] | None = None,
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
    lux_d = configs.get('lux', {})
    # reference_target: lux to anchor the lux calibration on (single capture nearest it);
    # 0 calibrates from a robust average across all captures instead, combined via
    # reference_method ('trimmed-mean' or 'median').
    lux_reference_target = lux_d.get('reference_target', 1000)
    if not isinstance(lux_reference_target, (int, float)) or lux_reference_target < 0:
        logger.warning('\nInvalid lux reference_target, defaulted to 1000')
        lux_reference_target = 1000
    lux_reference_target = int(lux_reference_target)
    lux_reference_method = lux_d.get('reference_method', 'trimmed-mean')
    if lux_reference_method not in ('trimmed-mean', 'median'):
        logger.warning('\nInvalid lux reference_method, defaulted to trimmed-mean')
        lux_reference_method = 'trimmed-mean'

    if blacklevel < -1 or blacklevel >= 2**16:
        logger.warning('\nInvalid blacklevel, defaulted to 64')
        blacklevel = -1

    if luminance_strength < 0 or luminance_strength > 1:
        logger.warning('\nInvalid luminance_strength strength, defaulted to 0.5')
        luminance_strength = 0.5

    if directory[-1] != '/':
        directory += '/'

    def _mode_label() -> str:
        if alsc_only:
            return 'ALSC only'
        if colour_only:
            return 'AWB + CCM only'
        if black_level_only:
            return 'Black level only'
        return 'Full'

    # Print config/options summary at start (Run vs Options, stable layout)
    config_src = 'default' if config is None or config is True or config is False else config
    mode = _mode_label()
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
        f'  LUX     reference_target={lux_reference_target}'
        + (f'  method={lux_reference_method}' if lux_reference_target == 0 else ''),
        f'  Disable {disable_str}',
        f'  Plot    {plot_str}',
    ]
    if images is not None:
        summary_lines.append(f'  Images  {len(images)} selected')
    logger.info('\n' + '\n'.join(summary_lines) + '\n')

    try:
        cam = Camera(json_output, json=json_template)
        cam.output_dir = output_dir
        cam.log_user_input(json_output, directory, config, log_output)
        cam.add_imgs(directory, mac_config, blacklevel, images=images)
        # Infer ALSC-only when only ALSC images present (e.g. mono LSC-only from DNGs),
        # and black-level-only when the directory holds nothing but dark frames.
        if len(cam.imgs) == 0 and len(cam.imgs_cac) == 0 and len(cam.imgs_alsc) > 0:
            alsc_only = True
        elif len(cam.imgs) == 0 and len(cam.imgs_cac) == 0 and len(cam.imgs_alsc) == 0 and len(cam.imgs_dark) > 0:
            black_level_only = True
        if black_level_only and len(cam.imgs_dark) == 0:
            raise ArgError('\n\nError: Black level only mode requires dark frames (dark_<n>.dng)')
        keep = None
        if alsc_only:
            keep = {'rpi.alsc'}
        elif colour_only:
            keep = {'rpi.awb', 'rpi.ccm'}
        elif black_level_only:
            keep = {'rpi.black_level'}
        if keep is not None:
            if cam.imgs_dark:
                # Dark frames make the black level measurable, and the measured
                # value feeds the kept algorithms; keep it enabled in any mode.
                keep |= {'rpi.black_level'}
            disable = set(cam.json.keys()) - keep
        cam.disable = disable
        cam.plot = plot
    except FileNotFoundError as err:
        raise ArgError('\n\nError: Input image directory not found!') from err

    # Build platform object for algorithms
    # Note: default_template not needed as json_template is already loaded
    platform = PlatformConfig(target, grid_size, None)

    if cam.check_imgs(macbeth=not (alsc_only or black_level_only)):
        # Mono (greyscale) sensors: disable AWB and CCM in output; they are not applicable.
        if cam.mono:
            logger.info('Mono sensor: disabling AWB and CCM in output.')
            disable = set(disable) | {'rpi.awb', 'rpi.ccm'}
            cam.disable = disable
        # Report the black level actually in use now that the DNGs have been read
        # (the config summary above only shows the -1 'auto' flag, not the value).
        bl_source = 'auto, from DNG' if blacklevel == -1 else 'override'
        logger.info(f'Black level: {cam.blacklevel_16} ({bl_source})')
        if not alsc_only:
            cam.json['rpi.black_level']['black_level'] = cam.blacklevel_16
        # When updating a file in-place with --alsc-only, --colour-only or
        # --blacklevel-only, keep all other algorithm sections unchanged; only
        # skip removing them from the output.
        if not (output_json_path and (alsc_only or colour_only or black_level_only)):
            cam.json_remove(disable)
        logger.info('\nStarting calibrations')
        # Locate the sensor's built-in tuning for this ISP, so CCM can also report
        # the shipped default's colour accuracy on these captures (best-effort).
        default_ccms, default_path = _default_tuning_ccms(cam, target)
        if default_path:
            cam.metrics['default_tuning_path'] = default_path
        algorithms = [
            # First: the measured black level must be in place before anything
            # downstream consumes blacklevel_16.
            BlackLevelCalibration(cam, platform, blacklevel),
            AlscCalibration(cam, platform, luminance_strength, do_alsc_colour, lsc_max_gain),
            GeqCalibration(cam, platform),
            LuxCalibration(cam, platform, lux_reference_target, lux_reference_method),
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
                default_ccms,
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
        mode = _mode_label()
        n_macbeth = len(cam.imgs)
        n_alsc = len(cam.imgs_alsc)
        n_cac = len(cam.imgs_cac)
        n_dark = len(cam.imgs_dark)
        _write_metrics(
            cam,
            json_output,
            target=target,
            mode=mode,
            config={
                'luminance_strength': luminance_strength,
                'max_gain': lsc_max_gain,
                'do_alsc_colour': do_alsc_colour,
                'matrix_selection': ccm_matrix_selection,
                'greyworld': greyworld,
                'blacklevel': blacklevel,
                'lux_reference_target': lux_reference_target,
                'lux_reference_method': lux_reference_method,
            },
        )
        counts = f'Macbeth: {n_macbeth}  ALSC: {n_alsc}  CAC: {n_cac}  Dark: {n_dark}'
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


def _default_tuning_ccms(cam: Camera, target: str) -> tuple[list | None, str | None]:
    """Locate the sensor's built-in libcamera tuning for this ISP; return its CCMs + path.

    Best-effort: returns (None, None) if no camera images, no shipped tuning for the
    sensor, or no CCM in it. Used only to compare the new calibration against the default.
    """
    if not cam.imgs:
        return None, None
    sensor = str(getattr(cam.imgs[0], 'cam_name', '') or '').strip().lower()
    if not sensor:
        return None, None
    isp = 'pisp' if target == 'pisp' else 'vc4'
    for base in ('/usr/share/libcamera/ipa/rpi', '/usr/local/share/libcamera/ipa/rpi'):
        path = Path(base) / isp / f'{sensor}.json'
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None, str(path)
        for entry in data.get('algorithms', []):
            if isinstance(entry, dict) and 'rpi.ccm' in entry:
                ccms = entry['rpi.ccm'].get('ccms')
                return (ccms or None), str(path)
        return None, str(path)
    return None, None


def _write_metrics(cam: Camera, json_output: str, *, target: str, mode: str, config: dict) -> None:
    """Write a structured metrics sidecar ({stem}_metrics.json) for the web UI.

    Best-effort: a failure here must never abort an otherwise-successful
    calibration, so any error is logged and swallowed.
    """
    try:
        cols = sorted({img.col for img in cam.imgs if getattr(img, 'col', None) is not None})
        awb = cam.json.get('rpi.awb', {}) if isinstance(cam.json.get('rpi.awb'), dict) else {}
        ccm = cam.json.get('rpi.ccm', {}) if isinstance(cam.json.get('rpi.ccm'), dict) else {}
        cam.metrics['counts'] = {
            'macbeth': len(cam.imgs),
            'alsc': len(cam.imgs_alsc),
            'cac': len(cam.imgs_cac),
            'dark': len(cam.imgs_dark),
        }
        cam.metrics['coverage'] = {
            'ct_min': cols[0] if cols else None,
            'ct_max': cols[-1] if cols else None,
            'ct_count': len(cols),
            'ccm_matrices': len(ccm.get('ccms', [])),
            'awb_points': len(awb.get('ct_curve', [])) // 3,
        }
        cam.metrics['config'] = config
        cam.metrics['target'] = target
        cam.metrics['mode'] = mode

        metrics_output = f'{Path(json_output).with_suffix("")}_metrics.json'
        with open(metrics_output, 'w') as f:
            json.dump(cam.metrics, f, indent=2)
    except Exception as err:  # pragma: no cover - defensive, never break a run
        logger.warning(f'Could not write metrics sidecar: {type(err).__name__}: {err}')


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
    black_level_only: bool = False,
    template_path: str | None = None,
    update_path: str | None = None,
    plot_cli: list[str] | None = None,
    images: list[str] | None = None,
) -> None:
    """Run a calibration for each target, resolving platform + template per target.

    The single entry point shared by the CLI and the web server: both supply
    inputs and let this drive the per-target loop. When ``update_path`` is set
    (CLI ``--update``) it is the template source *and* the in-place output file.
    ``images`` optionally restricts the run to those filenames within
    ``directory`` (default: every .dng found there).
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
            black_level_only=black_level_only,
            output_json_path=update_path,
            plot_cli=plot_cli,
            images=images,
        )
