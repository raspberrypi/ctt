# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Flask application: the ctt-server web UI + in-process Picamera2 capture + CTT.
#
# Everything runs in one process (on the Pi). The remote client is just a
# browser. Captures use Picamera2 directly; the tuner runs as a subprocess.

import contextlib
import difflib
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import exifread
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    stream_with_context,
    url_for,
)

from ctt.devices import LightboxError, get_shared_lightbox

from . import colour_check, ctt_runner, mtf, results
from .camera import (
    MJPEG_CONTENT_TYPE,
    CameraError,
    get_shared_camera,
    platform_target,
    reload_shared_camera,
)
from .naming import NamingError, detect_type, validate_filename
from .sessions import Project, Workspace

logger = logging.getLogger(__name__)

# rawpy lets us develop a preview JPEG from a DNG that has no saved sibling JPEG
# (uploaded or pre-existing captures). It is normally present (a ctt dependency);
# if missing, those captures simply aren't viewable.
try:
    import rawpy  # noqa: F401

    _RAWPY = True
except Exception:  # pragma: no cover - depends on the environment
    _RAWPY = False
    logger.warning('rawpy not available: DNG-only captures will have no in-browser preview')


# libcamera installs its built-in (system) tuning files here, one JSON per
# sensor, split by ISP platform. A local build under /usr/local takes precedence
# over the distro package, mirroring libcamera's own search order.
_SYSTEM_TUNING_ROOTS = ('/usr/local/share', '/usr/share')


def _system_tuning_dirs(target: str) -> list[Path]:
    """Installed libcamera tuning directories for an ISP platform ('pisp'/'vc4')."""
    return [Path(root) / 'libcamera' / 'ipa' / 'rpi' / target for root in _SYSTEM_TUNING_ROOTS]


def _run_info(available: dict) -> dict:
    """Map target -> {label, epoch} from each tuning file's mtime, for the UI."""
    info = {}
    for target, f in available.items():
        mtime = f.get('mtime')
        if mtime:
            info[target] = {'label': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M'), 'epoch': int(mtime)}
    return info


def _serialise_captures(project: Project) -> list[dict]:
    """Capture metadata for the browser, with a per-file validity flag."""
    out = []
    for c in project.captures:
        valid, _ = validate_filename(c.filename)
        if project.has_saved_jpeg(c.filename):
            jpeg = 'saved'
        elif _RAWPY and c.filename.endswith('.dng'):
            jpeg = 'dng'  # no saved JPEG, but rawpy can develop a preview on demand
        else:
            jpeg = None
        out.append(
            {
                'filename': c.filename,
                'image_type': c.image_type,
                'colour_temp': c.colour_temp,
                'lux': c.lux,
                'label': c.label,
                'valid': valid,
                'jpeg': jpeg,
                'excluded': c.excluded,
            }
        )
    return out


def _dng_exif(path: Path) -> dict:
    """Read a DNG's EXIF metadata: a curated summary plus the full tag dump.

    Raspberry Pi DNGs vary by writer (rpicam/libcamera vs PiDNG), so each
    summary tag is looked up across the three IFD prefixes they use — the same
    tolerance as the core's dng_load_image. Tags a writer omits are skipped.
    """
    with open(path, 'rb') as f:
        tags = exifread.process_file(f, details=False)

    def tag(name):
        for prefix in ('EXIF SubIFD0', 'Image', 'EXIF'):
            value = tags.get(f'{prefix} {name}')
            if value is not None:
                return value
        raise KeyError(name)

    def exposure_ms(t):
        r = t.values[0]
        return f'{r.num / r.den * 1000:.2f} ms'

    rows = [
        ('Camera model', 'Model', str),
        ('Make', 'Make', str),
        ('Software', 'Software', str),
        ('Captured', 'DateTime', str),
        ('Width', 'ImageWidth', lambda t: str(t.values[0])),
        ('Height', 'ImageLength', lambda t: str(t.values[0])),
        ('Exposure time', 'ExposureTime', exposure_ms),
        ('ISO', 'ISOSpeedRatings', lambda t: str(t.values[0])),
        ('Black level', 'BlackLevel', str),
        ('White level', 'Tag 0xC61D', lambda t: str(t.values[0])),  # exifread's name for DNG WhiteLevel
        ('CFA pattern', 'CFAPattern', str),
    ]
    summary = []
    for label, name, fmt in rows:
        try:
            summary.append({'label': label, 'value': fmt(tag(name))})
        except Exception:  # noqa: S112 - tag missing or unexpected shape: skip the row
            continue
    # Full dump for the expandable "all tags" view; thumbnails are raw bytes, skip them.
    all_tags = [{'key': k, 'value': str(v)} for k, v in sorted(tags.items()) if not isinstance(v, bytes)]
    return {'summary': summary, 'tags': all_tags}


def create_app(workspace_root: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config['WORKSPACE'] = Workspace(workspace_root)
    # Allow large raw-image uploads (DNGs are tens of MB; a batch can be large).
    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024

    def workspace() -> Workspace:
        return app.config['WORKSPACE']

    @app.context_processor
    def _asset_version() -> dict:
        """Cache-busting token so browsers re-fetch app.js/app.css after a deploy."""
        static = Path(app.static_folder)
        try:
            v = int(max((static / f).stat().st_mtime for f in ('app.js', 'app.css')))
        except OSError:
            v = 0
        return {'asset_v': v}

    def get_project_or_404(name: str):
        try:
            return workspace().get_project(name)
        except FileNotFoundError:
            abort(404, f'No such project: {name}')

    def camera_or_503():
        try:
            return get_shared_camera()
        except CameraError as err:
            abort(503, str(err))

    def lightbox_or_none():
        """The shared lightbox, or None when no device (or pyusb) is available."""
        try:
            return get_shared_lightbox()
        except LightboxError:
            return None

    def lightbox_or_503():
        box = lightbox_or_none()
        if box is None:
            abort(503, 'No lightbox detected')
        return box

    def lightbox_status() -> dict:
        box = lightbox_or_none()
        if box is None:
            return {'present': False}
        try:
            return {'present': True, **box.info()}
        except LightboxError:  # device went away mid-session
            return {'present': False}

    # --- pages -------------------------------------------------------------
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/projects')
    def projects():
        return render_template(
            'projects.html',
            projects=workspace().list_projects(),
            workspace_root=str(workspace().root),
        )

    @app.route('/projects', methods=['POST'])
    def create_project():
        name = request.form.get('name', '').strip()
        try:
            project = workspace().create_project(name)
        except (ValueError, FileExistsError) as err:
            return render_template(
                'projects.html',
                projects=workspace().list_projects(),
                workspace_root=str(workspace().root),
                error=str(err),
            )
        return redirect(url_for('project', name=project.name))

    @app.route('/projects/<name>/delete', methods=['POST'])
    def delete_project(name: str):
        workspace().delete_project(name)
        return redirect(url_for('projects'))

    @app.route('/projects/<name>')
    def project(name: str):
        proj = get_project_or_404(name)
        return render_template(
            'capture.html',
            project=proj,
            captures=_serialise_captures(proj),
            counts=proj.counts(),
        )

    @app.route('/projects/<name>/images')
    def images_page(name: str):
        proj = get_project_or_404(name)
        return render_template('images.html', project=proj, captures=_serialise_captures(proj), rawpy_available=_RAWPY)

    # --- camera API --------------------------------------------------------
    @app.route('/api/health')
    def api_health():
        try:
            return jsonify({'camera': True, **get_shared_camera().health()})
        except CameraError as err:
            return jsonify({'camera': False, 'error': str(err)}), 503

    @app.route('/api/preview')
    def api_preview():
        camera = camera_or_503()
        return Response(stream_with_context(camera.mjpeg_frames()), mimetype=MJPEG_CONTENT_TYPE)

    @app.route('/api/controls', methods=['GET'])
    def api_get_controls():
        return jsonify(camera_or_503().get_controls())

    @app.route('/api/controls', methods=['POST'])
    def api_set_controls():
        return jsonify(camera_or_503().set_controls(request.get_json(force=True) or {}))

    @app.route('/api/transform', methods=['POST'])
    def api_transform():
        body = request.get_json(force=True) or {}
        return jsonify(camera_or_503().set_transform(bool(body.get('hflip')), bool(body.get('vflip'))))

    @app.route('/api/mode', methods=['POST'])
    def api_mode():
        """Switch the camera to one of its advertised sensor modes."""
        cam = camera_or_503()
        body = request.get_json(force=True) or {}
        try:
            width, height = int(body['width']), int(body['height'])
        except (KeyError, TypeError, ValueError):
            return jsonify({'error': 'width and height are required'}), 400
        try:
            return jsonify(cam.set_mode(width, height))
        except CameraError as err:
            return jsonify({'error': str(err)}), 400

    @app.route('/api/histogram')
    def api_histogram():
        return jsonify(camera_or_503().histogram())

    @app.route('/api/macbeth')
    def api_macbeth():
        return jsonify(camera_or_503().detect_chart())

    @app.route('/api/macbeth-deltae')
    def api_macbeth_deltae():
        """Measure the colour accuracy actually achieved by the loaded tuning.

        Locates the Macbeth chart in a live ISP-processed frame and reports the
        delta E of each patch against the reference colours — comparable with
        the calibration-time CCM metrics, but measured through the real pipeline.
        """
        info = camera_or_503().chart_patches()
        if not info.get('found'):
            return jsonify({'found': False})
        # Picamera2 'RGB888' arrays are BGR-ordered; the colour maths wants RGB.
        rgb_frame = info['frame'][..., ::-1]
        means = colour_check.patch_means(rgb_frame, info['centres'])
        report = colour_check.deltae_report(means)
        return jsonify({'found': True, 'confidence': info['confidence'], **report})

    # --- live preview test (load a generated tuning into the camera) -------
    @app.route('/projects/<name>/preview-test', methods=['POST'])
    def preview_test(name: str):
        """Restart the camera with this project's generated (or custom) tuning for live preview."""
        proj = get_project_or_404(name)
        body = request.get_json(silent=True) or {}
        kind = body.get('kind', 'generated')
        target = platform_target()
        if target is None:
            return jsonify({'error': 'Could not determine the camera ISP platform.'}), 503
        label = None
        variant_id = None
        if kind == 'custom':
            # Pick the requested variant (by slug), falling back to the first one.
            variant = body.get('variant')
            variants = _list_variants(proj, target)
            sel = next((v for v in variants if v['id'] == variant), None) or (variants[0] if variants else None)
            variant_id = sel['id'] if sel else None
            if sel is None:
                return jsonify(
                    {'error': f'No custom tuning for {target.upper()} — save one on the Tuning tab first.'}
                ), 400
            json_path = _variant_path(proj, target, variant_id)
            label = sel['label']
        else:
            json_path = proj.output_dir / f'{proj.name}_{target}.json'
            if not json_path.exists():
                return jsonify(
                    {
                        'error': f'No {target.upper()} tuning for this project — this Pi runs {target.upper()}. '
                        f'Re-run CTT for {target}.'
                    }
                ), 400
        try:
            cam = reload_shared_camera(tuning_file=str(json_path), preview_max_width=1920)
        except CameraError as err:
            return jsonify({'error': str(err)}), 503
        return jsonify(
            {
                'target': target,
                'tuning': json_path.name,
                'kind': kind,
                'model': cam.model,
                'variant': variant_id,
                'label': label,
            }
        )

    @app.route('/api/preview-default', methods=['POST'])
    def api_preview_default():
        """Restore the camera to its default (built-in) tuning."""
        try:
            cam = reload_shared_camera(tuning_file=None)
        except CameraError as err:
            return jsonify({'error': str(err)}), 503
        return jsonify({'tuning': None, 'model': cam.model})

    @app.route('/projects/<name>/preview-capture')
    def preview_capture(name: str):
        """Download a PNG snapshot of the live preview (zero shutter lag).

        Grabs the frame currently on screen at the selected mode's preview
        resolution, so manual exposure/gain is preserved (no mode-switch blip).
        """
        proj = get_project_or_404(name)
        cam = camera_or_503()
        try:
            png = cam.capture_preview_png()
        except CameraError as err:
            abort(503, str(err))
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return Response(
            png,
            mimetype='image/png',
            headers={'Content-Disposition': f'attachment; filename="{proj.name}_{stamp}.png"'},
        )

    # --- lightbox API ------------------------------------------------------
    @app.route('/api/lightbox', methods=['GET'])
    def api_lightbox():
        return jsonify(lightbox_status())

    @app.route('/api/lightbox', methods=['POST'])
    def api_set_lightbox():
        box = lightbox_or_503()
        body = request.get_json(force=True) or {}
        percent = body.get('percent')
        if percent is not None:
            percent = float(percent)
        # set_illuminant accepts names and channel numbers alike, so 'illuminant'
        # and 'channel' are interchangeable; a bare 'percent' adjusts the active one.
        target = body.get('illuminant') if body.get('illuminant') is not None else body.get('channel')
        try:
            if body.get('off'):
                box.off()
            elif target is not None:
                box.set_illuminant(target, percent)
            elif percent is not None:
                state = box.get_state()
                if state.illuminant is None:  # fresh device: channel 0, nothing selected yet
                    return jsonify({'error': 'No illuminant is active — select one first.'}), 400
                box.set_illuminant(state.channel, percent)
        except (LightboxError, TypeError, ValueError) as err:
            return jsonify({'error': str(err)}), 400
        return jsonify(lightbox_status())

    @app.route('/projects/<name>/capture', methods=['POST'])
    def capture(name: str):
        """Capture a still — or a burst of `frames` stills, saved as indexed
        files (alsc_5000k_0/1/2..., d65_5000k_800l_0/1/2...) which CTT averages
        internally during a run."""
        proj = get_project_or_404(name)
        body = request.get_json(force=True) or {}
        image_type = body.get('image_type')
        try:
            # Dark frames carry no tags; everything else needs a colour temp.
            colour_temp = None if image_type == 'dark' else int(body.get('colour_temp'))
            lux = int(body['lux']) if body.get('lux') not in (None, '') else None
            label = body.get('label') or None
            frames = max(1, min(int(body.get('frames', 1) or 1), 16))
            shots = get_shared_camera().capture_burst(frames)
            caps = [
                proj.add_capture(
                    dng_bytes,
                    image_type,
                    colour_temp,
                    lux=lux,
                    label=label,
                    controls=meta,
                    jpeg_bytes=jpg_bytes,
                    indexed=frames > 1,  # burst frames get _<n> names; singles keep overwrite semantics
                )
                for dng_bytes, jpg_bytes, meta in shots
            ]
        except CameraError as err:
            return jsonify({'error': str(err)}), 503
        except (TypeError, ValueError) as err:
            return jsonify({'error': str(err)}), 400
        added = [
            {
                'filename': cap.filename,
                'image_type': cap.image_type,
                'colour_temp': cap.colour_temp,
                'lux': cap.lux,
                'label': cap.label,
                'valid': True,
                'jpeg': 'saved',  # a fresh capture always has a full-res JPEG
            }
            for cap in caps
        ]
        return jsonify({'added': added, 'counts': proj.counts()})

    @app.route('/projects/<name>/upload', methods=['POST'])
    def upload(name: str):
        """Import custom image files, auto-tagging each from its CTT-format filename."""
        proj = get_project_or_404(name)
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        added, skipped = [], []
        for f in files:
            if not f or not f.filename:
                continue
            try:
                cap = proj.import_capture(f.filename, f.read())
                added.append(
                    {
                        'filename': cap.filename,
                        'image_type': cap.image_type,
                        'colour_temp': cap.colour_temp,
                        'lux': cap.lux,
                        'label': cap.label,
                        'valid': True,
                        'jpeg': 'dng' if _RAWPY else None,  # no saved JPEG; develop on demand if possible
                    }
                )
            except NamingError as err:
                skipped.append({'filename': f.filename, 'reason': str(err)})
        return jsonify({'added': added, 'skipped': skipped, 'counts': proj.counts()})

    @app.route('/projects/<name>/captures/<filename>/delete', methods=['POST'])
    def delete_capture(name: str, filename: str):
        proj = get_project_or_404(name)
        proj.delete_capture(filename)
        return jsonify({'ok': True, 'counts': proj.counts()})

    @app.route('/projects/<name>/captures/<filename>/exclude', methods=['POST'])
    def set_capture_excluded(name: str, filename: str):
        """Exclude a capture from (or re-include it in) CTT runs; the file stays on disk."""
        proj = get_project_or_404(name)
        body = request.get_json(force=True) or {}
        try:
            cap = proj.set_excluded(filename, bool(body.get('excluded', True)))
        except KeyError:
            abort(404, f'No such capture: {filename}')
        return jsonify({'filename': cap.filename, 'excluded': cap.excluded})

    # Quick black level measurements, cached per (path, mtime) so repeated Run
    # tab visits don't re-read multi-megabyte DNGs.
    _blacklevel_cache: dict[str, dict] = {}

    @app.route('/projects/<name>/blacklevel')
    def project_blacklevel(name: str):
        """Black level measured from the project's dark frames (seeds the Run tab).

        Uses the same measurement helper as the calibration algorithm, so the
        seeded value matches what a full run would compute.
        """
        from ctt.algorithms.black_level import measure_dark_dng  # noqa: PLC0415 - defer the ctt import

        proj = get_project_or_404(name)
        excluded = {c.filename for c in proj.captures if c.excluded}
        frames = []
        for p in sorted(proj.path.glob('*.dng')):
            if detect_type(p.name) != 'dark' or p.name in excluded:
                continue
            mtime = p.stat().st_mtime
            cached = _blacklevel_cache.get(str(p))
            if cached is None or cached['mtime'] != mtime:
                try:
                    cached = {'mtime': mtime, 'frame': measure_dark_dng(str(p))}
                except Exception as err:
                    logger.warning(f'Black level measurement failed for {p.name}: {err}')
                    continue
                _blacklevel_cache[str(p)] = cached
            frames.append(cached['frame'])
        if not frames:
            return jsonify({'black_level': None, 'frames': []})
        black_level = round(sum(f['black_level'] for f in frames) / len(frames))
        return jsonify({'black_level': black_level, 'frames': frames})

    def _develop_dng(dng: Path, thumb: bool = False) -> Response:
        """Develop a DNG into a preview JPEG with rawpy (won't match the ISP look).

        thumb=True develops at half size with a lower JPEG quality — the
        Images-tab grid would otherwise trigger a full-resolution develop per
        thumbnail, which is far too slow on a Pi."""
        import cv2  # noqa: PLC0415

        try:
            with rawpy.imread(str(dng)) as raw:
                rgb = raw.postprocess(half_size=thumb)
            quality = 70 if thumb else 90
            ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        except Exception:
            logger.exception('Failed to develop DNG preview for %s', dng.name)
            abort(500)
        if not ok:
            abort(500)
        return Response(buf.tobytes(), mimetype='image/jpeg')

    @app.route('/projects/<name>/captures/<filename>/jpeg')
    def capture_jpeg(name: str, filename: str):
        """Serve a capture's preview JPEG.

        ?source=jpeg serves only the saved sibling JPEG (404 if none);
        ?source=raw develops the DNG with rawpy (404 without rawpy);
        ?source=auto (default) prefers the saved JPEG, falling back to rawpy."""
        proj = get_project_or_404(name)
        source = request.args.get('source', 'auto')
        thumb = request.args.get('thumb') == '1'
        if not filename.endswith('.dng'):  # <filename> can't contain '/', so no traversal
            abort(404)
        jpg = (proj.path / filename).with_suffix('.jpg')
        if source != 'raw' and jpg.exists():
            return send_file(jpg, mimetype='image/jpeg')
        if source == 'jpeg':
            abort(404)
        dng = proj.path / filename
        if not (_RAWPY and dng.exists()):
            abort(404)
        return _develop_dng(dng, thumb=thumb)

    @app.route('/projects/<name>/captures/<filename>/exif')
    def capture_exif(name: str, filename: str):
        proj = get_project_or_404(name)
        dng = proj.path / filename
        if not filename.endswith('.dng') or not dng.exists():
            abort(404)
        try:
            return jsonify({'filename': filename, **_dng_exif(dng)})
        except Exception:
            logger.exception('Failed to read EXIF for %s', filename)
            abort(500)

    # --- run ---------------------------------------------------------------
    @app.route('/projects/<name>/run')
    def run_page(name: str):
        proj = get_project_or_404(name)
        excluded_count = sum(1 for c in proj.captures if c.excluded)
        return render_template('run.html', project=proj, counts=proj.counts(), excluded_count=excluded_count)

    @app.route('/projects/<name>/run/stream')
    def run_stream(name: str):
        proj = get_project_or_404(name)
        targets = [t for t in request.args.get('targets', 'pisp,vc4').split(',') if t]
        mode = request.args.get('mode', 'full')
        update = request.args.get('update') == '1'
        options = {
            'awb': {'greyworld': request.args.get('greyworld') == '1'},
            'alsc': {
                'do_alsc_colour': request.args.get('do_alsc_colour', '1') == '1',
                'luminance_strength': float(request.args.get('luminance_strength', 0.8)),
                'max_gain': float(request.args.get('max_gain', 8.0)),
            },
            'blacklevel': int(request.args.get('blacklevel', -1)),
            'disable': [d for d in request.args.get('disable', '').split(',') if d],
            'ccm': {
                'matrix_selection': request.args.get('matrix_selection', 'average'),
                'test_patches': [int(p) for p in request.args.get('test_patches', '').split(',') if p.strip()],
            },
            'lux': {
                'reference_target': int(request.args.get('lux_reference_target', 1000)),
                'reference_method': request.args.get('lux_reference_method', 'trimmed-mean'),
            },
        }

        def generate():
            for line in ctt_runner.run_ctt_stream(proj, targets, mode, options, update=update):
                yield f'data: {json.dumps(line)}\n\n'

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
        )

    @app.route('/projects/<name>/run/import-tuning', methods=['POST'])
    def import_tuning(name: str):
        """Import a tuning file as this project's tuning for its target.

        The tuning comes from an uploaded file or, with ``system_path``, an
        installed libcamera tuning. The target is read from the file itself; the
        file then becomes {name}_{target}.json so a subsequent update run
        refreshes it in place.
        """
        from ctt.core.runner import get_target_from_tuning_file
        from ctt.utils.errors import ArgError

        proj = get_project_or_404(name)
        system_path = request.form.get('system_path')
        if system_path:
            # Restrict to the installed tuning directories so a request can't read
            # arbitrary files off the Pi via this endpoint.
            src = Path(system_path)
            plat = platform_target()
            allowed = plat is not None and src.resolve().parent in {d.resolve() for d in _system_tuning_dirs(plat)}
            if not allowed or not src.is_file():
                return jsonify({'error': 'Not an installed tuning file'}), 400
            try:
                text = src.read_text(encoding='utf-8')
                json.loads(text)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
                return jsonify({'error': f'Could not read system tuning: {err}'}), 400
        else:
            upload = request.files.get('file')
            if upload is None or not upload.filename:
                return jsonify({'error': 'No file uploaded'}), 400
            try:
                text = upload.read().decode('utf-8')
                json.loads(text)
            except (UnicodeDecodeError, json.JSONDecodeError) as err:
                return jsonify({'error': f'Invalid JSON: {err}'}), 400
        # Stage to a temp file so the target can be read before we know the path.
        proj.output_dir.mkdir(parents=True, exist_ok=True)
        tmp = proj.output_dir / f'.import-{proj.name}.json'
        tmp.write_text(text)
        try:
            target = get_target_from_tuning_file(str(tmp))
        except ArgError as err:
            tmp.unlink(missing_ok=True)
            return jsonify({'error': str(err).strip()}), 400
        tmp.replace(proj.output_dir / f'{proj.name}_{target}.json')
        return jsonify({'target': target})

    @app.route('/projects/<name>/run/system-tunings')
    def system_tunings(name: str):
        """List installed libcamera tunings for this Pi's ISP, for the update base.

        Defaults the selection to the file matching the live sensor model (e.g.
        imx708.json), so an update can be seeded from the camera's stock tuning.
        """
        get_project_or_404(name)
        target = platform_target()
        if target is None:
            return jsonify({'error': 'Could not determine the camera ISP platform.'}), 503
        model = None
        with contextlib.suppress(CameraError):
            model = get_shared_camera().model
        files: list[dict] = []
        seen: set[str] = set()
        for d in _system_tuning_dirs(target):
            if d.is_dir():
                for f in sorted(d.glob('*.json')):
                    if f.name not in seen:  # /usr/local entry wins over /usr/share
                        seen.add(f.name)
                        files.append({'name': f.name, 'path': str(f)})
        default = next((f['path'] for f in files if model and f['name'] == f'{model}.json'), None)
        if default is None and files:
            default = files[0]['path']
        return jsonify({'target': target, 'model': model, 'default': default, 'files': files})

    # --- results -----------------------------------------------------------
    @app.route('/projects/<name>/results')
    def results_page(name: str):
        proj = get_project_or_404(name)
        files = ctt_runner.output_files(proj, ['pisp', 'vc4'])
        available = {t: f for t, f in files.items() if f['json']}
        return render_template('results.html', project=proj, files=available, runs=_run_info(available))

    @app.route('/projects/<name>/tuning')
    def tuning_page(name: str):
        proj = get_project_or_404(name)
        files = ctt_runner.output_files(proj, ['pisp', 'vc4'])
        available = {t: f for t, f in files.items() if f['json']}
        return render_template('tuning.html', project=proj, files=available, runs=_run_info(available))

    @app.route('/projects/<name>/preview')
    def preview_page(name: str):
        proj = get_project_or_404(name)
        files = ctt_runner.output_files(proj, ['pisp', 'vc4'])
        available = {t: f for t, f in files.items() if f['json']}
        variants = {t: _list_variants(proj, t) for t in available}
        return render_template(
            'preview.html', project=proj, files=available, runs=_run_info(available), variants=variants
        )

    # --- MTF (slanted-edge) --------------------------------------------------
    # Chart captures live in <project>/mtf/, which CTT runs never see (they only
    # scan project-root *.dng), so MTF charts can't pollute a calibration.

    @app.route('/projects/<name>/mtf')
    def mtf_page(name: str):
        proj = get_project_or_404(name)
        return render_template('mtf.html', project=proj)

    @app.route('/projects/<name>/mtf/capture', methods=['POST'])
    def mtf_capture(name: str):
        """Capture a fresh full-res frame of the MTF chart (raw DNG + preview JPEG)."""
        proj = get_project_or_404(name)
        cam = camera_or_503()
        try:
            dng_bytes, jpg_bytes, _meta = cam.capture_still()
        except CameraError as err:
            return jsonify({'error': str(err)}), 503
        mtf_dir = proj.path / 'mtf'
        mtf_dir.mkdir(parents=True, exist_ok=True)
        (mtf_dir / 'chart.dng').write_bytes(dng_bytes)
        (mtf_dir / 'chart.jpg').write_bytes(jpg_bytes)
        return jsonify({'ok': True})

    @app.route('/projects/<name>/mtf/preview')
    def mtf_preview(name: str):
        proj = get_project_or_404(name)
        jpg = proj.path / 'mtf' / 'chart.jpg'
        if not jpg.exists():
            abort(404)
        return send_file(jpg, mimetype='image/jpeg')

    @app.route('/projects/<name>/mtf/auto', methods=['POST'])
    def mtf_auto(name: str):
        """Auto-detect measurable slanted edges on the captured chart."""
        proj = get_project_or_404(name)
        dng = proj.path / 'mtf' / 'chart.dng'
        if not dng.exists():
            return jsonify({'error': 'No chart captured yet'}), 404
        try:
            rois = mtf.auto_detect(str(dng))
        except Exception:
            logger.exception('MTF auto-detect failed for %s', name)
            return jsonify({'error': 'Edge detection failed (is the capture a valid DNG?)'}), 500
        return jsonify({'rois': rois})

    @app.route('/projects/<name>/mtf/measure', methods=['POST'])
    def mtf_measure(name: str):
        """Measure the MTF of each requested ROI on the captured chart's raw DNG."""
        proj = get_project_or_404(name)
        dng = proj.path / 'mtf' / 'chart.dng'
        if not dng.exists():
            return jsonify({'error': 'No chart captured yet'}), 404
        body = request.get_json(force=True) or {}
        rois = body.get('rois') or []
        try:
            rois = [{k: int(r[k]) for k in ('x', 'y', 'w', 'h')} for r in rois]
        except (KeyError, TypeError, ValueError):
            return jsonify({'error': 'rois must be a list of {x, y, w, h}'}), 400
        if not rois:
            return jsonify({'error': 'No ROIs supplied'}), 400
        try:
            results_list = mtf.measure_rois(str(dng), rois)
        except Exception:
            logger.exception('MTF measurement failed for %s', name)
            return jsonify({'error': 'MTF measurement failed (is the capture a valid DNG?)'}), 500
        return jsonify({'rois': results_list})

    @app.route('/projects/<name>/results/data')
    def results_data(name: str):
        proj = get_project_or_404(name)
        target = request.args.get('target', 'pisp')
        json_path = proj.output_dir / f'{proj.name}_{target}.json'
        if not json_path.exists():
            abort(404, 'No tuning file for that target')
        metrics_path = proj.output_dir / f'{proj.name}_{target}_metrics.json'
        return jsonify(results.parse_tuning_file(json_path, metrics_path))

    @app.route('/projects/<name>/archive')
    def download_archive(name: str):
        """Download the whole project (captured DNGs + tuning outputs) as a zip."""
        import io
        import zipfile

        proj = get_project_or_404(name)
        mem = io.BytesIO()
        # DNGs are already-compressed raw; ZIP_STORED avoids pointless CPU on the Pi.
        with zipfile.ZipFile(mem, 'w', zipfile.ZIP_STORED) as zf:
            for path in sorted(proj.path.rglob('*')):
                if path.is_file():
                    zf.write(path, path.relative_to(proj.path.parent))
        mem.seek(0)
        return send_file(mem, mimetype='application/zip', as_attachment=True, download_name=f'{proj.name}.zip')

    @app.route('/projects/<name>/download/<kind>/<target>')
    def download(name: str, kind: str, target: str):
        proj = get_project_or_404(name)
        if kind == 'custom':
            # A specific variant by ?variant=<slug>, else the first one. Only a
            # listed slug reaches a file path (so this can't read arbitrary files).
            variant = request.args.get('variant')
            slugs = [v['id'] for v in _list_variants(proj, target)]
            slug = variant if variant in slugs else (slugs[0] if not variant and slugs else None)
            path = _variant_path(proj, target, slug) if slug is not None else None
            if path is None or not path.exists():
                abort(404)
            return send_file(path, as_attachment=True, download_name=path.name)
        suffix = {'json': '.json', 'log': '.log'}.get(kind)
        if suffix is None:
            abort(404)
        path = proj.output_dir / f'{proj.name}_{target}{suffix}'
        if not path.exists():
            abort(404)
        return send_file(path, as_attachment=True, download_name=path.name)

    # --- custom (hand-edited) tuning variants -------------------------------
    # A target can carry several hand-edited tuning variants, each a full tuning
    # JSON with a user label. The files are authoritative for existence and are
    # named by a slug derived from the label, for readability:
    #   output/{proj}_{target}_custom_{slug}.json    (slug = 'warm-skin-v2')
    # A small per-target manifest maps slug -> the exact label + a staleness
    # baseline (the slug is lossy, so the label is kept verbatim here):
    #   output/{proj}_{target}_custom_index.json     [{id, label, base_mtime}]
    # The manifest is non-authoritative: every list reconciles it against the
    # files on disk, so an out-of-band delete can't leave a dangling entry. The
    # legacy single custom file ({proj}_{target}_custom.json) and any files from
    # the earlier integer-id scheme are migrated to slug names on first sight.
    # base_mtime is the generated original's mtime when the variant was last
    # saved; a variant is "stale" once the original has been regenerated past it.
    MAX_VARIANTS = 20
    MAX_LABEL_LEN = 60

    def _generated_path(proj: Project, target: str) -> Path:
        return proj.output_dir / f'{proj.name}_{target}.json'

    def _generated_mtime(proj: Project, target: str) -> float | None:
        gen = _generated_path(proj, target)
        return gen.stat().st_mtime if gen.exists() else None

    def _existing_tuning_text(proj: Project, target: str) -> str | None:
        """The camera's built-in (existing) tuning text for this target, if known.

        The run records which installed libcamera tuning it compared against as
        'default_tuning_path' in the metrics sidecar; that is the "Existing" file.
        """
        metrics_path = proj.output_dir / f'{proj.name}_{target}_metrics.json'
        if metrics_path.exists():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                dp = json.loads(metrics_path.read_text()).get('default_tuning_path')
                if dp and Path(dp).exists():
                    return Path(dp).read_text()
        return None

    def _index_path(proj: Project, target: str) -> Path:
        return proj.output_dir / f'{proj.name}_{target}_custom_index.json'

    def _variant_path(proj: Project, target: str, slug: str) -> Path:
        return proj.output_dir / f'{proj.name}_{target}_custom_{slug}.json'

    def _valid_target(target: str) -> None:
        if target not in ctt_runner.VALID_TARGETS:
            abort(404)

    def _valid_slug(slug: str) -> str:
        """Guard the untrusted slug path segment before it reaches a file path."""
        if slug == 'index' or not re.fullmatch(r'[a-z0-9][a-z0-9-]*', slug or ''):
            abort(404)
        return slug

    def _slugify(label: str) -> str:
        """A readable, path-safe filename stem from a label ('Warm Skin v2' -> 'warm-skin-v2')."""
        slug = re.sub(r'[^a-z0-9]+', '-', label.strip().lower()).strip('-')
        # 'index' is reserved (the manifest is {proj}_{target}_custom_index.json).
        return 'custom' if not slug or slug == 'index' else slug

    def _unique_slug(label: str, taken: set[str]) -> str:
        base = _slugify(label)
        slug, n = base, 2
        while slug in taken:
            slug, n = f'{base}-{n}', n + 1
        return slug

    def _validate_label(label: str) -> str | None:
        if not label:
            return 'A label is required.'
        if len(label) > MAX_LABEL_LEN:
            return f'Label too long (max {MAX_LABEL_LEN} characters).'
        return None

    def _disk_variant_slugs(proj: Project, target: str) -> dict[str, Path]:
        """slug -> file path for every {proj}_{target}_custom_<slug>.json on disk."""
        out: dict[str, Path] = {}
        prefix = f'{proj.name}_{target}_custom_'
        index_name = _index_path(proj, target).name
        for p in proj.output_dir.glob(f'{prefix}*.json'):
            if p.name == index_name:  # the manifest is not a variant
                continue
            slug = p.name[len(prefix) : -len('.json')]
            if slug:
                out[slug] = p
        return out

    def _read_index(proj: Project, target: str) -> list[dict]:
        p = _index_path(proj, target)
        if p.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                data = json.loads(p.read_text())
                if isinstance(data, list):
                    return data
        return []

    def _write_index(proj: Project, target: str, entries: list[dict]) -> None:
        idx = _index_path(proj, target)
        if entries:
            proj.output_dir.mkdir(parents=True, exist_ok=True)
            idx.write_text(json.dumps(entries, indent=2))
        elif idx.exists():
            idx.unlink()  # no variants left: don't leave an empty index lying around

    def _reconcile_variants(proj: Project, target: str) -> list[dict]:
        """Manifest reconciled against the files on disk (and persisted).

        Migrates legacy/integer-id files to slug names, drops entries whose file
        is gone, and synthesises an entry for any orphan file with no manifest
        record. Returns the variant list as [{id (slug), label, base_mtime}].
        """
        disk = _disk_variant_slugs(proj, target)
        entries = _read_index(proj, target)
        label_by_slug: dict[str, str] = {}
        base_by_slug: dict[str, float | None] = {}
        for e in entries:
            if isinstance(e, dict) and e.get('id') is not None:
                label_by_slug[str(e['id'])] = e.get('label')
                base_by_slug[str(e['id'])] = e.get('base_mtime')

        def adopt(old_slug: str | None, src: Path, label: str, base: float | None) -> None:
            new_slug = _unique_slug(label, set(disk) - ({old_slug} if old_slug else set()))
            dest = _variant_path(proj, target, new_slug)
            src.rename(dest)
            if old_slug:
                del disk[old_slug]
            disk[new_slug] = dest
            label_by_slug[new_slug] = label
            base_by_slug[new_slug] = base

        # Migrate files from the earlier integer-id scheme to label-based slugs.
        for old in [s for s in list(disk) if s.isdigit()]:
            adopt(old, disk[old], label_by_slug.get(old) or f'Custom {old}', base_by_slug.get(old))
        # Migrate the legacy single custom file.
        legacy = proj.output_dir / f'{proj.name}_{target}_custom.json'
        if legacy.exists():
            adopt(None, legacy, 'Custom', _generated_mtime(proj, target))

        # Manifest order first (for entries still on disk), then orphan files.
        order = [str(e['id']) for e in entries if isinstance(e, dict) and str(e.get('id')) in disk]
        order += [s for s in sorted(disk) if s not in order]
        reconciled, seen = [], set()
        for slug in order:
            if slug in disk and slug not in seen:
                seen.add(slug)
                reconciled.append(
                    {
                        'id': slug,
                        'label': label_by_slug.get(slug) or slug.replace('-', ' '),
                        'base_mtime': base_by_slug.get(slug),
                    }
                )
        _write_index(proj, target, reconciled)
        return reconciled

    def _list_variants(proj: Project, target: str) -> list[dict]:
        """Reconciled variants decorated with a derived 'stale' flag, for the UI."""
        gen_mtime = _generated_mtime(proj, target)
        out = []
        for e in _reconcile_variants(proj, target):
            base = e.get('base_mtime')
            stale = bool(gen_mtime is not None and base is not None and gen_mtime > base + 1e-6)
            out.append({'id': e['id'], 'label': e['label'], 'stale': stale})
        return out

    def _tuning_state(proj: Project, target: str, variant_id: str | None = None) -> dict:
        """The Tuning tab's contents: original text, the selected variant's text + diff,
        and the full variant list. variant_id picks which one's text/diff to return."""
        _valid_target(target)
        gen_path = _generated_path(proj, target)
        if not gen_path.exists():
            abort(404, 'No tuning file for that target')
        original = gen_path.read_text()
        variants = _list_variants(proj, target)
        selected = None
        if variant_id is not None and any(v['id'] == variant_id for v in variants):
            selected = variant_id
        elif variants:
            selected = variants[0]['id']
        custom = diff = None
        if selected is not None:
            cpath = _variant_path(proj, target, selected)
            if cpath.exists():
                custom = cpath.read_text()
                diff = '\n'.join(
                    difflib.unified_diff(
                        original.splitlines(),
                        custom.splitlines(),
                        fromfile=gen_path.name,
                        tofile=cpath.name,
                        lineterm='',
                    )
                )
        return {
            'original': original,
            'custom': custom,
            'diff': diff,
            'variants': variants,
            'selected': selected,
            'existing': _existing_tuning_text(proj, target),
        }

    @app.route('/projects/<name>/tuning-data/<target>')
    def tuning_data(name: str, target: str):
        """The Tuning tab's contents for a target, optionally for a chosen variant."""
        proj = get_project_or_404(name)
        return jsonify(_tuning_state(proj, target, request.args.get('variant') or None))

    @app.route('/projects/<name>/tuning/custom/<target>', methods=['POST'])
    def create_custom_tuning(name: str, target: str):
        """Save hand edits as a new labelled custom variant (validated JSON)."""
        proj = get_project_or_404(name)
        _valid_target(target)
        body = request.get_json(force=True) or {}
        label = (body.get('label') or '').strip()
        if err := _validate_label(label):
            return jsonify({'error': err}), 400
        try:
            json.loads(body.get('json', ''))
        except json.JSONDecodeError as err:
            return jsonify({'error': f'Invalid JSON: {err}'}), 400
        taken = {v['id'] for v in _reconcile_variants(proj, target)}
        if len(taken) >= MAX_VARIANTS:
            return jsonify({'error': f'Too many custom variants (max {MAX_VARIANTS}) — remove one first.'}), 400
        slug = _unique_slug(label, taken)
        proj.output_dir.mkdir(parents=True, exist_ok=True)
        _variant_path(proj, target, slug).write_text(body['json'])
        entries = _read_index(proj, target)
        entries.append({'id': slug, 'label': label, 'base_mtime': _generated_mtime(proj, target)})
        _write_index(proj, target, entries)
        return jsonify(_tuning_state(proj, target, slug))

    @app.route('/projects/<name>/tuning/custom/<target>/<vid>', methods=['POST'])
    def save_custom_tuning(name: str, target: str, vid: str):
        """Save edits to an existing variant in place; refresh its staleness baseline."""
        proj = get_project_or_404(name)
        _valid_target(target)
        slug = _valid_slug(vid)
        path = _variant_path(proj, target, slug)
        if not path.exists():
            abort(404)
        text = (request.get_json(force=True) or {}).get('json', '')
        try:
            json.loads(text)
        except json.JSONDecodeError as err:
            return jsonify({'error': f'Invalid JSON: {err}'}), 400
        path.write_text(text)
        entries = _reconcile_variants(proj, target)
        for e in entries:
            if e['id'] == slug:
                e['base_mtime'] = _generated_mtime(proj, target)  # edits are now against the current original
        _write_index(proj, target, entries)
        return jsonify(_tuning_state(proj, target, slug))

    @app.route('/projects/<name>/tuning/custom/<target>/<vid>/copy', methods=['POST'])
    def copy_custom_tuning(name: str, target: str, vid: str):
        """Duplicate a variant's contents under a new label."""
        proj = get_project_or_404(name)
        _valid_target(target)
        src = _variant_path(proj, target, _valid_slug(vid))
        if not src.exists():
            abort(404)
        label = ((request.get_json(force=True) or {}).get('label') or '').strip()
        if err := _validate_label(label):
            return jsonify({'error': err}), 400
        taken = {v['id'] for v in _reconcile_variants(proj, target)}
        if len(taken) >= MAX_VARIANTS:
            return jsonify({'error': f'Too many custom variants (max {MAX_VARIANTS}) — remove one first.'}), 400
        slug = _unique_slug(label, taken)
        _variant_path(proj, target, slug).write_text(src.read_text())
        entries = _read_index(proj, target)
        entries.append({'id': slug, 'label': label, 'base_mtime': _generated_mtime(proj, target)})
        _write_index(proj, target, entries)
        return jsonify(_tuning_state(proj, target, slug))

    @app.route('/projects/<name>/tuning/custom/<target>/<vid>/delete', methods=['POST'])
    def delete_custom_tuning(name: str, target: str, vid: str):
        """Remove a single custom variant (file + manifest entry)."""
        proj = get_project_or_404(name)
        _valid_target(target)
        slug = _valid_slug(vid)
        _variant_path(proj, target, slug).unlink(missing_ok=True)
        _write_index(proj, target, [e for e in _read_index(proj, target) if str(e.get('id')) != slug])
        return jsonify(_tuning_state(proj, target))

    return app
