# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Flask application: the ctt-server web UI + in-process Picamera2 capture + CTT.
#
# Everything runs in one process (on the Pi). The remote client is just a
# browser. Captures use Picamera2 directly; the tuner runs as a subprocess.

import json
import logging
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

from . import ctt_runner, results
from .camera import (
    MJPEG_CONTENT_TYPE,
    CameraError,
    get_shared_camera,
    platform_target,
    reload_shared_camera,
)
from .naming import NamingError, validate_filename
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
            info = box.info()
            return {'present': True, 'illuminants': box.illuminants, **info}
        except LightboxError:  # device went away mid-session
            return {'present': False}

    # --- pages -------------------------------------------------------------
    @app.route('/')
    def index():
        return redirect(url_for('projects'))

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

    @app.route('/api/histogram')
    def api_histogram():
        return jsonify(camera_or_503().histogram())

    @app.route('/api/macbeth')
    def api_macbeth():
        return jsonify(camera_or_503().detect_chart())

    # --- live preview test (load a generated tuning into the camera) -------
    @app.route('/projects/<name>/preview-test', methods=['POST'])
    def preview_test(name: str):
        """Restart the camera with this project's generated tuning for live preview."""
        proj = get_project_or_404(name)
        target = platform_target()
        if target is None:
            return jsonify({'error': 'Could not determine the camera ISP platform.'}), 503
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
        return jsonify({'target': target, 'tuning': json_path.name, 'model': cam.model})

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
        """Download a full-resolution PNG still from the live preview camera."""
        proj = get_project_or_404(name)
        cam = camera_or_503()
        try:
            png = cam.capture_png()
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
        try:
            if body.get('off'):
                box.off()
            elif body.get('illuminant') is not None:
                box.set_illuminant(body['illuminant'], None if percent is None else float(percent))
            elif body.get('channel') is not None:
                channel = int(body['channel'])
                if percent is None:
                    box.set_channel(channel)
                else:
                    box.set_intensity(channel, float(percent))
            elif percent is not None:
                box.set_intensity(box.get_channel(), float(percent))
        except (LightboxError, TypeError, ValueError) as err:
            return jsonify({'error': str(err)}), 400
        return jsonify(lightbox_status())

    @app.route('/projects/<name>/capture', methods=['POST'])
    def capture(name: str):
        proj = get_project_or_404(name)
        body = request.get_json(force=True) or {}
        image_type = body.get('image_type')
        try:
            colour_temp = int(body.get('colour_temp'))
            lux = int(body['lux']) if body.get('lux') not in (None, '') else None
            label = body.get('label') or None
            dng_bytes, jpg_bytes, meta = get_shared_camera().capture_still()
            cap = proj.add_capture(
                dng_bytes, image_type, colour_temp, lux=lux, label=label, controls=meta, jpeg_bytes=jpg_bytes
            )
        except CameraError as err:
            return jsonify({'error': str(err)}), 503
        except (TypeError, ValueError) as err:
            return jsonify({'error': str(err)}), 400
        return jsonify(
            {
                'filename': cap.filename,
                'image_type': cap.image_type,
                'colour_temp': cap.colour_temp,
                'lux': cap.lux,
                'label': cap.label,
                'jpeg': 'saved',  # a fresh capture always has a full-res JPEG
                'counts': proj.counts(),
            }
        )

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
            for line in ctt_runner.run_ctt_stream(proj, targets, mode, options):
                yield f'data: {json.dumps(line)}\n\n'

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
        )

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
        return render_template('preview.html', project=proj, files=available, runs=_run_info(available))

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
        suffix = {'json': '.json', 'log': '.log'}.get(kind)
        if suffix is None:
            abort(404)
        path = proj.output_dir / f'{proj.name}_{target}{suffix}'
        if not path.exists():
            abort(404)
        return send_file(path, as_attachment=True, download_name=path.name)

    return app
