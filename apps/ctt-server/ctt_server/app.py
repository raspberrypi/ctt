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
        out.append(
            {
                'filename': c.filename,
                'image_type': c.image_type,
                'colour_temp': c.colour_temp,
                'lux': c.lux,
                'label': c.label,
                'valid': valid,
            }
        )
    return out


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
            dng_bytes, meta = get_shared_camera().capture_dng()
            cap = proj.add_capture(dng_bytes, image_type, colour_temp, lux=lux, label=label, controls=meta)
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

    # --- run ---------------------------------------------------------------
    @app.route('/projects/<name>/run')
    def run_page(name: str):
        proj = get_project_or_404(name)
        return render_template('run.html', project=proj, counts=proj.counts())

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
