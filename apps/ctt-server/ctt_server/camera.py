# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Picamera2 wrapper: the server captures images in-process on the Pi.
#
# Provides a live preview (JPEG/MJPEG), an RGB histogram for clipping checks,
# exposure/gain controls, and DNG capture from the raw stream. Picamera2 is
# imported lazily so the package can still be imported (and the tuner run) on
# machines without a camera, e.g. a desktop for development.

import contextlib
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# libcamera logs a lot at INFO — notably the one-time sensor-mode probe (reading
# Picamera2.sensor_modes configures the camera once per mode). Drop INFO so the server
# console stays readable; warnings/errors still show and an explicit user value wins.
os.environ.setdefault('LIBCAMERA_LOG_LEVELS', 'WARN')

# The advertised sensor modes are sensor-specific and constant, but reading
# Picamera2.sensor_modes probes every mode (slow + noisy). Cache them per model so
# the camera can be reopened (e.g. a tuning reload) without re-probing.
_SENSOR_MODES_CACHE: dict[str, list[dict]] = {}
_picamera2_logging_quieted = False

# JPEG frame boundary used for the MJPEG multipart stream.
MJPEG_BOUNDARY = 'frame'
MJPEG_CONTENT_TYPE = f'multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}'


class CameraError(RuntimeError):
    """Raised when the camera is unavailable or a capture fails."""


class Picamera2Camera:
    """Thin wrapper around Picamera2 for calibration capture.

    The sensor is driven at full resolution on the `raw` stream (the DNG source),
    and the live preview/histogram `main` stream is derived from that same
    full-resolution frame — merely scaled down for streaming. This guarantees the
    viewfinder's field of view is exactly what the captured DNG covers.
    """

    def __init__(self, preview_max_width: int = 1920, tuning_file: str | None = None) -> None:
        try:
            from picamera2 import Picamera2  # noqa: PLC0415 (lazy import)
        except ImportError as err:  # pragma: no cover - depends on Pi environment
            raise CameraError(
                'picamera2 is not available. It ships with Raspberry Pi OS; on a '
                'plain venv create it with --system-site-packages or install python3-picamera2.'
            ) from err

        global _picamera2_logging_quieted
        if not _picamera2_logging_quieted:
            Picamera2.set_logging(logging.WARNING)  # quieten Picamera2's own INFO chatter (once)
            _picamera2_logging_quieted = True

        self._lock = threading.Lock()
        self._ev = 0.0  # exposure compensation (EV); tracked here as metadata may omit it
        self._auto = True  # AeEnable state; tracked so we report it reliably (AeLocked is ambiguous)
        self._fps = 30.0  # framerate target; 0 = unconstrained (variable frame duration)
        # Optionally start the pipeline with a specific tuning file (e.g. a freshly
        # generated CTT tuning, for the Results-page live preview test). None = the
        # camera's built-in default tuning.
        self.tuning_file = tuning_file
        if tuning_file:
            tuning = Picamera2.load_tuning_file(os.path.basename(tuning_file), dir=os.path.dirname(tuning_file))
            self._picam2 = Picamera2(tuning=tuning)
        else:
            self._picam2 = Picamera2()
        try:
            self.model = self._picam2.camera_properties.get('Model', 'unknown')
            # Advertised sensor modes, one entry per raw size (preferring the deeper
            # readout when sizes repeat). Probing sensor_modes is slow, so cache per
            # model and reuse when the camera is reopened (e.g. a tuning reload).
            modes = _SENSOR_MODES_CACHE.get(self.model)
            if modes is None:
                by_size: dict[tuple[int, int], dict] = {}
                for m in self._picam2.sensor_modes:
                    size = (int(m['size'][0]), int(m['size'][1]))
                    entry = {
                        'size': size,
                        'bit_depth': int(m.get('bit_depth', 0)),
                        'fps': round(float(m.get('fps', 0.0)), 1),
                        'unpacked': str(m['unpacked']),
                    }
                    if size not in by_size or entry['bit_depth'] > by_size[size]['bit_depth']:
                        by_size[size] = entry
                modes = sorted(by_size.values(), key=lambda m: m['size'][0] * m['size'][1], reverse=True)
                _SENSOR_MODES_CACHE[self.model] = modes
            self.sensor_modes = modes
            self._preview_max_width = preview_max_width
            # Default to the largest mode (the full sensor readout) → full-res DNGs.
            # Picked over sensor_resolution since the pixel-array size isn't always a
            # usable mode.
            self._apply_mode(modes[0])
            # Sensor flip (applied at configure time via a libcamera Transform); affects
            # both the live preview and the captured raw/DNG/PNG.
            self._hflip = False
            self._vflip = False
            self._picam2.configure(self._video_config())
            self._picam2.start()
        except Exception:
            # A failed bring-up (e.g. a bad custom tuning rejected by libcamera)
            # must release the device, or the process holds the camera acquired
            # and every reopen fails until a restart.
            with contextlib.suppress(Exception):
                self._picam2.close()
            raise
        # Allow auto-exposure to settle before the first frame.
        time.sleep(0.5)

    def _transform(self):
        from libcamera import Transform  # noqa: PLC0415 (lazy; Pi-only)

        return Transform(hflip=1 if self._hflip else 0, vflip=1 if self._vflip else 0)

    def _apply_mode(self, mode: dict) -> None:
        """Adopt a sensor mode: raw/capture size, bit depth and preview size."""
        w, h = (int(mode['size'][0]), int(mode['size'][1]))
        self._raw_size = (w, h)
        self._raw_bit_depth = int(mode.get('bit_depth', 0)) or None
        # Unpacked Bayer format (e.g. 'SRGGB12') so the raw stream is delivered
        # one sample per 16-bit word rather than MIPI-packed (the _CSI2P default).
        self._raw_format = mode.get('unpacked')
        self.resolution = self._raw_size
        # Preview derived from the mode's frame, scaled down preserving aspect.
        prev_w = min(self._preview_max_width, w) & ~1
        prev_h = int(round(prev_w * h / w)) & ~1
        self._preview_size = (prev_w, prev_h)

    def _sensor_config(self) -> dict:
        # sensor= pins the actual camera mode (readout/binning) so libcamera
        # cannot satisfy the request by picking another mode and rescaling.
        cfg: dict = {'output_size': self._raw_size}
        if self._raw_bit_depth:
            cfg['bit_depth'] = self._raw_bit_depth
        return cfg

    def set_mode(self, width: int, height: int) -> dict:
        """Switch to an advertised sensor mode and reconfigure the live pipeline.

        All subsequent captures (DNG/JPEG/PNG) are at the selected mode's
        resolution, and the preview is rescaled from it.
        """
        size = (int(width), int(height))
        mode = next((m for m in self.sensor_modes if tuple(m['size']) == size), None)
        if mode is None:
            raise CameraError(f'{size[0]}x{size[1]} is not an advertised sensor mode')
        with self._lock:
            self._apply_mode(mode)
            self._picam2.stop()
            self._picam2.configure(self._video_config())
            self._picam2.start()
        time.sleep(0.3)  # let the pipeline settle before the next frame is served
        return {'resolution': list(self.resolution)}

    def _frame_duration_limits(self) -> tuple[int, int]:
        """FrameDurationLimits for the current fps target (0 = unconstrained).

        A fixed rate pins both limits (which also caps exposure at one frame,
        and is clamped by libcamera to the control's advertised range); 0 spans
        the camera's full FrameDurationLimits range so long exposures can
        stretch the frame, instead of create_video_configuration's locked-30fps
        default clamping them at ~33 ms.
        """
        if self._fps:
            frame_duration = int(round(1_000_000 / self._fps))
            return (frame_duration, frame_duration)
        fd_min, fd_max, _ = self._picam2.camera_controls['FrameDurationLimits']
        return (int(fd_min), int(fd_max))

    def _video_config(self):
        return self._picam2.create_video_configuration(
            main={'size': self._preview_size, 'format': 'RGB888'},
            raw={'size': self._raw_size, 'format': self._raw_format},
            sensor=self._sensor_config(),
            transform=self._transform(),
            buffer_count=4,
            controls={'FrameDurationLimits': self._frame_duration_limits()},
        )

    def set_transform(self, hflip: bool, vflip: bool) -> dict:
        """Set the sensor h/v flip and reconfigure the live pipeline immediately."""
        with self._lock:
            self._hflip, self._vflip = bool(hflip), bool(vflip)
            self._picam2.stop()
            self._picam2.configure(self._video_config())
            self._picam2.start()
        time.sleep(0.3)  # let the pipeline settle before the next frame is served
        return {'hflip': self._hflip, 'vflip': self._vflip}

    # --- controls ----------------------------------------------------------
    def get_controls(self) -> dict:
        md = self._picam2.capture_metadata()
        frame_duration = int(md.get('FrameDuration', 0))
        # The sensor's advertised control ranges (mode-dependent) are the
        # absolute truth for the UI sliders.
        ctrl = self._picam2.camera_controls
        exp_min, exp_max, _ = ctrl['ExposureTime']
        gain_min, gain_max, _ = ctrl['AnalogueGain']
        fd_min, fd_max, _ = ctrl['FrameDurationLimits']
        return {
            'exposure': int(md.get('ExposureTime', 0)),
            'gain': round(float(md.get('AnalogueGain', 0.0)), 3),
            'colour_temp': int(md.get('ColourTemperature', 0)),
            'lux': round(float(md.get('Lux', 0.0)), 1),
            'focus_fom': int(md.get('FocusFoM', 0)),  # focus figure of merit (higher = sharper)
            'ev': round(self._ev, 2),
            'auto_exposure': self._auto,
            'fps': self._fps,  # user target; 0 = unconstrained
            'measured_fps': round(1_000_000 / frame_duration, 1) if frame_duration else 0,
            'exposure_min': int(exp_min),
            'exposure_max': int(exp_max),
            'gain_min': round(float(gain_min), 3),
            'gain_max': round(float(gain_max), 3),
            'frame_duration_min': int(fd_min),
            'frame_duration_max': int(fd_max),
        }

    def set_controls(self, controls: dict) -> dict:
        new = {}
        if 'auto_exposure' in controls:
            self._auto = bool(controls['auto_exposure'])
        if self._auto:
            new['AeEnable'] = True
        else:
            # Manual: disable AEC and apply the requested exposure/gain.
            new['AeEnable'] = False
            if controls.get('exposure') is not None:
                new['ExposureTime'] = int(controls['exposure'])
            if controls.get('gain') is not None:
                new['AnalogueGain'] = float(controls['gain'])
        if 'ev' in controls and controls['ev'] is not None:
            self._ev = float(controls['ev'])
            new['ExposureValue'] = self._ev  # AEC bias; only affects auto-exposure
        if controls.get('fps') is not None:
            self._fps = max(float(controls['fps']), 0.0)
            new['FrameDurationLimits'] = self._frame_duration_limits()
        if 'awb' in controls:
            new['AwbEnable'] = bool(controls['awb'])
        if new:
            self._picam2.set_controls(new)
            time.sleep(0.3)  # let the pipeline apply the new controls
        return self.get_controls()

    # --- preview / histogram ----------------------------------------------
    def capture_jpeg(self, quality: int = 95) -> bytes:
        import cv2  # noqa: PLC0415

        with self._lock:
            arr = self._picam2.capture_array('main')
        ok, buf = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise CameraError('Failed to encode preview frame')
        return buf.tobytes()

    def capture_png(self) -> bytes:
        """Capture a full-resolution processed still (current tuning applied) as PNG.

        Briefly switches the pipeline to a full-sensor-resolution still mode, then
        back to the preview/video config — so the result is the full field of view
        at native resolution, not the downscaled preview stream.
        """
        import cv2  # noqa: PLC0415

        with self._lock:
            still = self._picam2.create_still_configuration(
                main={'size': self.resolution, 'format': 'RGB888'},
                sensor=self._sensor_config(),
                transform=self._transform(),
            )
            arr = self._picam2.switch_mode_and_capture_array(still, 'main')
        # Picamera2 'RGB888' arrays are BGR-ordered, which is exactly what cv2 wants.
        ok, buf = cv2.imencode('.png', arr)
        if not ok:
            raise CameraError('Failed to encode PNG')
        return buf.tobytes()

    def capture_preview_png(self) -> bytes:
        """Snapshot the current live preview frame as a PNG (zero shutter lag).

        Grabs the running main stream straight from the pipeline — no mode switch —
        so the snapshot is exactly the frame on screen, at the selected mode's preview
        resolution, with the manual exposure/gain (and the ISP denoise state) left
        untouched. Switching to a full-resolution still mode (capture_png) reverts
        auto-exposure and steps the live preview's brightness, which is why the
        on-screen snapshot avoids it.
        """
        import cv2  # noqa: PLC0415

        with self._lock:
            arr = self._picam2.capture_array('main')  # RGB888 == BGR-ordered == cv2 native
        ok, buf = cv2.imencode('.png', arr)
        if not ok:
            raise CameraError('Failed to encode PNG')
        return buf.tobytes()

    def mjpeg_frames(self, fps: float = 10.0):
        """Yield multipart MJPEG chunks for a streaming HTTP response."""
        period = 1.0 / max(fps, 1.0)
        boundary = f'--{MJPEG_BOUNDARY}\r\n'.encode()
        while True:
            try:
                frame = self.capture_jpeg()
            except Exception:
                # The camera may have been torn down mid-stream (e.g. a tuning
                # reload). End the stream cleanly; the browser reconnects to the
                # new camera.
                return
            yield boundary
            yield b'Content-Type: image/jpeg\r\n'
            yield f'Content-Length: {len(frame)}\r\n\r\n'.encode()
            yield frame
            yield b'\r\n'
            time.sleep(period)

    def histogram(self, bins: int = 64) -> dict:
        import numpy as np  # noqa: PLC0415

        with self._lock:
            arr = self._picam2.capture_array('main')
        # capture_array('main') with RGB888 returns BGR order from Picamera2.
        b, g, r = arr[..., 0], arr[..., 1], arr[..., 2]
        out = {}
        for name, chan in (('r', r), ('g', g), ('b', b)):
            hist, _ = np.histogram(chan, bins=bins, range=(0, 256))
            out[name] = hist.astype(int).tolist()
        # Fraction of pixels at/near the top of the 8-bit preview range.
        clip = {name: round(float(np.mean(chan >= 250)), 4) for name, chan in (('r', r), ('g', g), ('b', b))}
        out['clipping'] = clip
        out['bins'] = bins
        return out

    # --- live Macbeth finder ----------------------------------------------
    def detect_chart(self) -> dict:
        """Locate a Macbeth chart in the current preview frame (live finder).

        Returns {found, confidence, corners} where corners are 4 [x, y] points
        normalised to 0-1 in the frame, so the browser can map them onto the
        displayed preview.
        """
        from ctt.detection.macbeth import locate_chart  # noqa: PLC0415

        with self._lock:
            arr = self._picam2.capture_array('main')
        try:
            res = locate_chart(arr)
        except Exception:  # detection is best-effort; never fail the request
            res = None
        if res is None:
            return {'found': False, 'confidence': None, 'corners': None, 'small': False, 'saturated': False}
        corners, _centres, conf = res
        h, w = arr.shape[:2]
        norm = [[float(x / w), float(y / h)] for (x, y) in corners]
        return {
            'found': True,
            'confidence': round(conf, 3),
            'corners': norm,
            'small': self._chart_too_small(corners, w, h),
            'saturated': self._chart_saturated(arr, corners),
        }

    def chart_patches(self) -> dict:
        """Locate the chart and return the current frame plus patch centres.

        Used by the live colour-accuracy check: the caller samples the patch
        colours from the returned (ISP-processed, BGR-ordered) frame.
        """
        from ctt.detection.macbeth import locate_chart  # noqa: PLC0415

        with self._lock:
            arr = self._picam2.capture_array('main')
        try:
            res = locate_chart(arr)
        except Exception:  # detection is best-effort; never fail the request
            res = None
        if res is None:
            return {'found': False}
        corners, centres, conf = res
        return {'found': True, 'confidence': round(conf, 3), 'centres': centres, 'frame': arr}

    # Fraction of the chart region that may clip before we flag over-exposure. The
    # bright (white/grey) patches are a small part of the chart, so a low threshold
    # catches them; CTT itself rejects a capture if any patch saturates.
    _CLIP_FRACTION = 0.02

    def _chart_saturated(self, arr, corners) -> bool:
        """True if the detected chart region is clipping in the preview (over-exposed)."""
        xs = [int(c[0]) for c in corners]
        ys = [int(c[1]) for c in corners]
        x0, x1 = max(min(xs), 0), min(max(xs), arr.shape[1])
        y0, y1 = max(min(ys), 0), min(max(ys), arr.shape[0])
        crop = arr[y0:y1, x0:x1]
        if crop.size == 0:
            return False
        clipped = float((crop >= 250).any(axis=-1).mean())
        return clipped > self._CLIP_FRACTION

    # CTT samples a 16x16 px window per patch in each half-resolution Bayer channel
    # (image.get_patches, size=16). The 6x4 chart needs each patch comfortably larger
    # than that, so warn when the per-patch pitch falls below ~2x the sample window.
    _PATCH_SAMPLE = 16
    _MIN_PITCH = 2 * _PATCH_SAMPLE  # minimum comfortable half-res patch pitch (px)

    def _chart_too_small(self, corners, frame_w: int, frame_h: int) -> bool:
        xs = [float(c[0]) for c in corners]
        ys = [float(c[1]) for c in corners]
        # Chart extent as a fraction of the preview frame == fraction of the full-res
        # capture (same field of view); map to full-res pixels for the patch maths.
        cap_w, cap_h = self.resolution
        chart_w = (max(xs) - min(xs)) / frame_w * cap_w
        chart_h = (max(ys) - min(ys)) / frame_h * cap_h
        pitch_x = chart_w / 2 / 6  # half-res pitch across 6 patch columns
        pitch_y = chart_h / 2 / 4  # half-res pitch across 4 patch rows
        return min(pitch_x, pitch_y) < self._MIN_PITCH

    # --- DNG capture -------------------------------------------------------
    def capture_dng(self) -> tuple[bytes, dict]:
        with self._lock:
            request = self._picam2.capture_request()
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'capture.dng'
                    request.save_dng(str(path))
                    data = path.read_bytes()
                metadata = request.get_metadata()
            finally:
                request.release()
        meta = {
            'exposure': int(metadata.get('ExposureTime', 0)),
            'gain': round(float(metadata.get('AnalogueGain', 0.0)), 3),
            'colour_temp': int(metadata.get('ColourTemperature', 0)),
            'lux': round(float(metadata.get('Lux', 0.0)), 1),
        }
        return data, meta

    def capture_still(self, quality: int = 95) -> tuple[bytes, bytes, dict]:
        """Capture one full-resolution frame as BOTH a DNG and a JPEG."""
        return self.capture_burst(1, quality=quality)[0]

    def capture_burst(self, frames: int, quality: int = 95) -> list[tuple[bytes, bytes, dict]]:
        """Capture a burst of full-resolution frames, each as a DNG + JPEG.

        The running video config has a full-resolution raw stream but only a
        preview-size main stream, so full-res JPEGs need a still mode. We switch
        to a full-sensor still configuration (full main + full raw) once, grab
        the requests back to back, and produce each frame's DNG from its raw
        stream and JPEG from its main stream -- so the pairs match exactly --
        then switch back to the preview config. This costs a brief preview
        blip, like capture_png().
        """
        import cv2  # noqa: PLC0415

        frames = max(1, min(int(frames), 16))
        shots = []
        with self._lock:
            still = self._picam2.create_still_configuration(
                main={'size': self.resolution, 'format': 'RGB888'},
                raw={'size': self._raw_size, 'format': self._raw_format},
                sensor=self._sensor_config(),
                transform=self._transform(),
            )
            self._picam2.switch_mode(still)
            try:
                for _ in range(frames):
                    request = self._picam2.capture_request()
                    try:
                        with tempfile.TemporaryDirectory() as tmp:
                            path = Path(tmp) / 'capture.dng'
                            request.save_dng(str(path))
                            dng = path.read_bytes()
                        arr = request.make_array('main')  # full-res RGB888 (BGR-ordered = cv2 native)
                        metadata = request.get_metadata()
                    finally:
                        request.release()
                    shots.append((dng, arr, metadata))
            finally:
                self._picam2.switch_mode(self._video_config())
        out = []
        for dng, arr, metadata in shots:
            ok, buf = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise CameraError('Failed to encode capture JPEG')
            meta = {
                'exposure': int(metadata.get('ExposureTime', 0)),
                'gain': round(float(metadata.get('AnalogueGain', 0.0)), 3),
                'colour_temp': int(metadata.get('ColourTemperature', 0)),
                'lux': round(float(metadata.get('Lux', 0.0)), 1),
            }
            out.append((dng, buf.tobytes(), meta))
        return out

    def health(self) -> dict:
        return {
            'model': self.model,
            'resolution': list(self.resolution),
            'modes': [
                {'size': list(m['size']), 'bit_depth': m['bit_depth'], 'fps': m['fps']} for m in self.sensor_modes
            ],
            'controls': self.get_controls(),
            'tuning': os.path.basename(self.tuning_file) if self.tuning_file else None,
            'hflip': self._hflip,
            'vflip': self._vflip,
        }

    def close(self) -> None:
        # Take the lock so we never tear the pipeline down mid-capture.
        with self._lock:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:  # pragma: no cover - best-effort teardown
                logger.exception('Error closing camera')


_SHARED: Picamera2Camera | None = None
_SHARED_LOCK = threading.Lock()


def get_shared_camera() -> Picamera2Camera:
    """Return a process-wide singleton camera (one Picamera2 per process)."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            try:
                _SHARED = Picamera2Camera()
            except CameraError:
                raise
            except Exception as err:
                raise CameraError(f'Failed to start the camera: {err}') from err
        return _SHARED


def validate_tuning_file(path: str, timeout: float = 60.0) -> None:
    """Test-load a tuning file by bringing the camera up in a throwaway subprocess.

    A tuning the IPA rejects doesn't just fail the open: libcamera drops the
    camera from the process-global camera manager, which cannot be recreated,
    leaving this process camera-less until a restart. Hand-edited tunings are
    therefore brought up in a sacrificial subprocess first, so the main process
    only ever loads files known to work. The camera must be closed in this
    process while the canary runs. Raises CameraError when the tuning is bad.
    """
    script = (
        'import os, sys\n'
        'from picamera2 import Picamera2\n'
        'path = sys.argv[1]\n'
        'tuning = Picamera2.load_tuning_file(os.path.basename(path), dir=os.path.dirname(path))\n'
        'cam = Picamera2(tuning=tuning)\n'
        'cam.configure(cam.create_video_configuration())\n'
        'cam.start()\n'
        'cam.stop()\n'
        'cam.close()\n'
    )
    try:
        proc = subprocess.run(
            [sys.executable, '-c', script, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as err:
        raise CameraError(f'Tuning validation timed out for {os.path.basename(path)}') from err
    if proc.returncode != 0:
        # Prefer libcamera's ERROR lines (the actual complaint) over the tail
        # of the python traceback.
        lines = [ln for ln in f'{proc.stderr}\n{proc.stdout}'.splitlines() if ln.strip()]
        errors = [ln for ln in lines if 'ERROR' in ln]
        detail = (errors or lines)[-1] if (errors or lines) else f'exit code {proc.returncode}'
        raise CameraError(f'Tuning {os.path.basename(path)} failed to load: {detail}')


def reload_shared_camera(
    tuning_file: str | None = None, preview_max_width: int = 1920, validate: bool = False
) -> Picamera2Camera:
    """Restart the shared camera with a given tuning file (None = default tuning).

    Only one Picamera2 can be open per process, so the current instance is closed
    first. A no-op when the camera already runs the requested tuning, which keeps the
    capture page's "restore default" call cheap. validate=True canary-tests the
    tuning in a subprocess first (see validate_tuning_file) — use it for any
    tuning that isn't known-good, e.g. hand-edited custom files.
    """
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is not None and _SHARED.tuning_file == tuning_file:
            return _SHARED
        if _SHARED is not None:
            _SHARED.close()
            _SHARED = None
        try:
            if validate and tuning_file is not None:
                validate_tuning_file(tuning_file)
            _SHARED = Picamera2Camera(preview_max_width=preview_max_width, tuning_file=tuning_file)
        except Exception as err:
            # The requested tuning failed to load: reopen with the default
            # tuning so the camera isn't left dead, then report.
            if tuning_file is not None:
                with contextlib.suppress(Exception):
                    _SHARED = Picamera2Camera(preview_max_width=preview_max_width)
            if isinstance(err, CameraError):
                raise
            tuning_name = os.path.basename(tuning_file) if tuning_file else 'the default tuning'
            raise CameraError(f'Failed to start the camera with {tuning_name}: {err}') from err
        return _SHARED


def platform_target() -> str | None:
    """Map the running ISP platform to a CTT target ('pisp'/'vc4'), or None.

    A tuning file is platform-specific, so the live preview test must load the one
    matching this Pi's ISP.
    """
    try:
        from picamera2 import Picamera2  # noqa: PLC0415

        name = getattr(Picamera2.platform, 'name', str(Picamera2.platform)).lower()
    except Exception:  # pragma: no cover - depends on Pi environment
        return None
    if 'pisp' in name:
        return 'pisp'
    if 'vc4' in name:
        return 'vc4'
    return None
