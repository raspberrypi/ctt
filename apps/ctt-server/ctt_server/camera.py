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

import logging
import os
import tempfile
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# libcamera logs a lot at INFO — notably the one-time sensor-mode probe (reading
# Picamera2.sensor_modes configures the camera once per mode). Drop INFO so the server
# console stays readable; warnings/errors still show and an explicit user value wins.
os.environ.setdefault('LIBCAMERA_LOG_LEVELS', 'WARN')

# Full-resolution raw size is sensor-specific and constant, but reading
# Picamera2.sensor_modes probes every mode (slow + noisy). Cache it per model so the
# camera can be reopened (e.g. a tuning reload) without re-probing.
_RAW_SIZE_CACHE: dict[str, tuple[int, int]] = {}
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
        # Optionally start the pipeline with a specific tuning file (e.g. a freshly
        # generated CTT tuning, for the Results-page live preview test). None = the
        # camera's built-in default tuning.
        self.tuning_file = tuning_file
        if tuning_file:
            tuning = Picamera2.load_tuning_file(os.path.basename(tuning_file), dir=os.path.dirname(tuning_file))
            self._picam2 = Picamera2(tuning=tuning)
        else:
            self._picam2 = Picamera2()
        self.model = self._picam2.camera_properties.get('Model', 'unknown')
        # Full-resolution raw stream (full sensor field of view) → full-res DNGs.
        # Pick the largest available raw sensor mode (the full readout), rather than
        # sensor_resolution, since the pixel-array size isn't always a usable mode.
        # Reading sensor_modes probes every mode, so cache the result per model and
        # reuse it when the camera is reopened (e.g. a tuning reload).
        raw_size = _RAW_SIZE_CACHE.get(self.model)
        if raw_size is None:
            largest = max(self._picam2.sensor_modes, key=lambda m: m['size'][0] * m['size'][1])['size']
            raw_size = (int(largest[0]), int(largest[1]))
            _RAW_SIZE_CACHE[self.model] = raw_size
        sensor_w, sensor_h = raw_size
        self.resolution = (sensor_w, sensor_h)
        # Preview derived from the full-res frame, scaled down preserving aspect.
        prev_w = min(preview_max_width, sensor_w) & ~1
        prev_h = int(round(prev_w * sensor_h / sensor_w)) & ~1
        config = self._picam2.create_video_configuration(
            main={'size': (prev_w, prev_h), 'format': 'RGB888'},
            raw={'size': raw_size},
            buffer_count=4,
        )
        self._picam2.configure(config)
        self._picam2.start()
        # Allow auto-exposure to settle before the first frame.
        time.sleep(0.5)

    # --- controls ----------------------------------------------------------
    def get_controls(self) -> dict:
        md = self._picam2.capture_metadata()
        return {
            'exposure': int(md.get('ExposureTime', 0)),
            'gain': round(float(md.get('AnalogueGain', 0.0)), 3),
            'colour_temp': int(md.get('ColourTemperature', 0)),
            'lux': round(float(md.get('Lux', 0.0)), 1),
            'ev': round(self._ev, 2),
            'auto_exposure': self._auto,
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

    def health(self) -> dict:
        return {
            'model': self.model,
            'resolution': list(self.resolution),
            'controls': self.get_controls(),
            'tuning': os.path.basename(self.tuning_file) if self.tuning_file else None,
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
            _SHARED = Picamera2Camera()
        return _SHARED


def reload_shared_camera(tuning_file: str | None = None, preview_max_width: int = 1920) -> Picamera2Camera:
    """Restart the shared camera with a given tuning file (None = default tuning).

    Only one Picamera2 can be open per process, so the current instance is closed
    first. A no-op when the camera already runs the requested tuning, which keeps the
    capture page's "restore default" call cheap.
    """
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is not None and _SHARED.tuning_file == tuning_file:
            return _SHARED
        if _SHARED is not None:
            _SHARED.close()
            _SHARED = None
        _SHARED = Picamera2Camera(preview_max_width=preview_max_width, tuning_file=tuning_file)
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
