# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Image Engineering lightSTUDIO-S driver (a concrete ctt.devices.Lightbox).
#
# Non-root access on the Pi: install contrib/99-lightstudio.rules to
# /etc/udev/rules.d/ then `sudo udevadm control --reload && sudo udevadm trigger`.

from __future__ import annotations

import enum
import logging
import threading

from ..lightbox import Lightbox, LightboxError

logger = logging.getLogger(__name__)


class Illuminant(enum.StrEnum):
    """The lightSTUDIO-S illuminants, for typo-safe scripting.

    Any API that takes an illuminant also accepts these (they are plain strings),
    e.g. box.set_illuminant(Illuminant.D65).

    Members are deliberately named exactly like their string values (rather than
    PEP 8's UPPER_CASE) so the two spellings can never diverge.
    """

    F12 = 'F12'
    F11 = 'F11'
    D50 = 'D50'
    D65 = 'D65'
    Halogen10 = 'Halogen10'
    Halogen100 = 'Halogen100'
    Halogen400 = 'Halogen400'
    HalogenBF = 'HalogenBF'


# Fixed illuminant per channel on the lightSTUDIO-S (from the CLI manual).
CHANNEL_NAMES = {
    1: str(Illuminant.F12),
    2: str(Illuminant.F11),
    3: str(Illuminant.D50),
    4: str(Illuminant.D65),
    5: str(Illuminant.Halogen10),
    6: str(Illuminant.Halogen100),
    7: str(Illuminant.Halogen400),
    8: str(Illuminant.HalogenBF),
}

# Descriptive labels for UIs; also accepted as illuminant names (matching folds
# case, spacing and punctuation, so e.g. 'halogen (10 lux)' resolves channel 5).
CHANNEL_LABELS = {
    1: 'F12',
    2: 'F11',
    3: 'D50',
    4: 'D65',
    5: 'Halogen (10 lux)',
    6: 'Halogen (100 lux)',
    7: 'Halogen (400 lux)',
    8: 'Halogen + blue filter (400 lux)',
}

# Nominal correlated colour temperature (K) per channel: the standard CIE
# illuminant values (D50/D65/F11/F12) and Illuminant A for the halogen channels
# (same tungsten spectrum at all three lux levels). Channel 8 (blue-filtered
# halogen) has no single standard CCT, so it is omitted — consumers default it
# (the web UI seeds 6500 K).
CHANNEL_TEMPS = {
    1: 3000,  # F12
    2: 4000,  # F11
    3: 5000,  # D50
    4: 6500,  # D65
    5: 2856,  # Halogen (CIE Illuminant A)
    6: 2856,
    7: 2856,
}


class LightStudioS(Lightbox):
    """Image Engineering lightSTUDIO-S over USB.

    Intensity is expressed in percent (0–100), matching the official CLI. All public
    methods are serialised under a lock so concurrent callers can share one device.
    """

    # Identified from a real lightSTUDIO-S (serial LS-S00270) via `lsusb -v`:
    #   idVendor 0x20a0  idProduct 0x412d  (iManufacturer "www.image-engineering.de")
    #   bDeviceClass 255 (vendor-specific), low-speed, bMaxPacketSize0 = 8.
    # NB 0x20a0 is a shared "Clay Logic / Openmoko" vendor id used by many V-USB
    # devices, so we also match on the manufacturer string before trusting a hit.
    VENDOR_ID = 0x20A0
    PRODUCT_ID = 0x412D
    _MANUFACTURER = 'www.image-engineering.de'

    _CONFIGURATION = 1
    _INTERFACE = 0
    _TIMEOUT_MS = 1000

    # --- wire protocol ----
    # The device has a single vendor interface with no endpoints, so every command
    # is a vendor CONTROL transfer on EP0:
    #
    #   GET (bmRequestType 0xC0, bRequest 1): wValue = command, returns wLength bytes
    #       getChannel           wValue 0x0065  wIndex 0        -> 1 byte  (channel)
    #       getIntensity         wValue 0x0066  wIndex 0        -> u16 LE  (current ch.)
    #       getDefaultIntensity  wValue 0x0068  wIndex channel  -> u16 LE
    #   SET (bmRequestType 0x40, bRequest 2): no data stage
    #       setIntensity         wValue (channel<<8)|0x03  wIndex value
    #
    # Intensity is carried in per-mille (0..1000 == 0..100 %): e.g. 50 % -> 500,
    # and getDefaultIntensity for channel 1 (F12) returns 0x03E8 = 1000 = 100 %.
    _RT_IN = 0xC0  # vendor | device-to-host
    _RT_OUT = 0x40  # vendor | host-to-device
    _REQ_GET = 0x01
    _REQ_SET = 0x02
    _CMD_GET_CHANNEL = 0x0065
    _CMD_GET_INTENSITY = 0x0066
    _CMD_GET_DEFAULT_INTENSITY = 0x0068
    _SUBCMD_SET_INTENSITY = 0x03  # low byte of wValue; high byte carries the channel
    _PER_MILLE = 1000  # device intensity unit: 0..1000 maps to 0..100 %
    _CHANNELS = range(1, 9)  # 8 fixed illuminant channels (see CHANNEL_NAMES)

    def __init__(
        self,
        serial: str | None = None,
        vendor_id: int | None = None,
        product_id: int | None = None,
    ) -> None:
        try:
            import usb.core  # noqa: PLC0415 (lazy import)
            import usb.util  # noqa: PLC0415
        except ImportError as err:  # pragma: no cover - depends on host environment
            raise LightboxError(
                'pyusb is not available. Install the devices extra (pip install -e ".[devices]") '
                'and the libusb backend (sudo apt install libusb-1.0-0).'
            ) from err

        self._usb = usb
        self._lock = threading.Lock()
        self._channel = 0  # last channel selected; tracked since the device is write-mostly

        vid = vendor_id if vendor_id is not None else self.VENDOR_ID
        pid = product_id if product_id is not None else self.PRODUCT_ID
        if vid == 0 or pid == 0:
            raise LightboxError('lightSTUDIO-S USB id is not configured (vendor_id/product_id).')

        self._dev = self._find_device(usb, vid, pid, serial)
        if self._dev is None:
            looked = f'{vid:#06x}:{pid:#06x}'
            extra = f' serial {serial!r}' if serial else ''
            raise LightboxError(f'lightSTUDIO-S not found ({looked}{extra}). Is it plugged in and powered?')

        self._serial = self._read_string(self._dev.iSerialNumber)

        # On Linux a kernel driver rarely binds a vendor device, but detach
        # defensively so claim_interface can't fail with "resource busy".
        try:
            if self._dev.is_kernel_driver_active(self._INTERFACE):
                self._dev.detach_kernel_driver(self._INTERFACE)
        except Exception:  # pragma: no cover - platform dependent (NotImplementedError on some backends)
            pass

        self._dev.set_configuration(self._CONFIGURATION)
        usb.util.claim_interface(self._dev, self._INTERFACE)
        logger.info('lightSTUDIO-S opened (%#06x:%#06x serial=%s)', vid, pid, self._serial)

    # --- discovery ---------------------------------------------------------
    @classmethod
    def probe(cls, serial: str | None = None) -> LightStudioS | None:
        """Return an opened lightSTUDIO-S if one is attached, else None."""
        try:
            return cls(serial=serial)
        except LightboxError:
            return None

    def _find_device(self, usb, vid: int, pid: int, serial: str | None):
        """Pick the matching USB device: right vid/pid, manufacturer, and serial.

        0x20a0 is shared, so verify the manufacturer string when readable (a missing
        string — e.g. no udev permission — is treated leniently). When `serial` is
        given, only a device whose iSerialNumber matches is accepted.
        """
        for dev in usb.core.find(find_all=True, idVendor=vid, idProduct=pid):
            mfr = self._read_string(dev.iManufacturer, dev)
            if mfr is not None and self._MANUFACTURER not in mfr:
                continue
            if serial is not None:
                sn = self._read_string(dev.iSerialNumber, dev)
                if sn is None or (sn != serial and serial not in sn):
                    continue
            return dev
        return None

    def _read_string(self, index, dev=None) -> str | None:
        dev = dev if dev is not None else self._dev
        try:
            return self._usb.util.get_string(dev, index)
        except (ValueError, self._usb.core.USBError):  # pragma: no cover - permission/descriptor dependent
            return None

    # --- identity ----------------------------------------------------------
    @property
    def model(self) -> str:
        return 'lightSTUDIO-S'

    @property
    def serial(self) -> str | None:
        return self._serial

    @property
    def illuminants(self) -> dict[int, str]:
        return CHANNEL_NAMES

    @property
    def illuminant_labels(self) -> dict[int, str]:
        return CHANNEL_LABELS

    @property
    def illuminant_temps(self) -> dict[int, int]:
        return CHANNEL_TEMPS

    # --- control primitives (the base class resolves names and validates) ---
    def _set_intensity(self, channel: int, percent: float) -> None:
        """Set `channel` to `percent` (0–100 %). Also makes it the active channel.

        Mirrors the CLI's `--channel N --setIntensity P`; setting an intensity is how
        the device switches illuminant, so there is no separate "select" step.
        """
        self._validate_channel(channel)
        with self._lock:
            self._set_intensity_locked(channel, self._clamp_percent(percent))

    def _get_state(self) -> tuple[int, float]:
        """Read (channel, intensity) under one lock — the device only reports the
        *current* channel (CLI `--getChannel` / `--getIntensity`)."""
        with self._lock:
            return self._get_channel_locked(), self._get_intensity_locked()

    def _get_default_intensity(self, channel: int) -> float:
        """Return `channel`'s power-on default intensity (0–100 %)."""
        self._validate_channel(channel)
        with self._lock:
            return self._get_default_intensity_locked(channel)

    def _set_channel(self, channel: int) -> None:
        """Switch on `channel` at its stored default intensity (CLI `--setChannel`)."""
        self._validate_channel(channel)
        with self._lock:
            self._select_channel_locked(channel)

    # --- protocol (vendor control transfers on EP0) ------------------------
    # Callers already hold self._lock, so the *_locked helpers must not re-acquire it.

    def _set_intensity_locked(self, channel: int, percent: float) -> None:
        w_value = (channel << 8) | self._SUBCMD_SET_INTENSITY
        w_index = int(round(percent / 100.0 * self._PER_MILLE))
        self._dev.ctrl_transfer(self._RT_OUT, self._REQ_SET, w_value, w_index, None, self._TIMEOUT_MS)
        self._channel = channel

    def _get_intensity_locked(self) -> float:
        data = self._dev.ctrl_transfer(self._RT_IN, self._REQ_GET, self._CMD_GET_INTENSITY, 0, 2, self._TIMEOUT_MS)
        return self._decode_permille(data)

    def _get_channel_locked(self) -> int:
        data = self._dev.ctrl_transfer(self._RT_IN, self._REQ_GET, self._CMD_GET_CHANNEL, 0, 1, self._TIMEOUT_MS)
        self._channel = int(data[0])
        return self._channel

    def _get_default_intensity_locked(self, channel: int) -> float:
        data = self._dev.ctrl_transfer(
            self._RT_IN, self._REQ_GET, self._CMD_GET_DEFAULT_INTENSITY, channel, 2, self._TIMEOUT_MS
        )
        return self._decode_permille(data)

    def _select_channel_locked(self, channel: int) -> None:
        # The device has no standalone "select"; switching channel means setting it
        # to its stored default intensity, exactly as the CLI's --setChannel does.
        default_pct = self._get_default_intensity_locked(channel)
        self._set_intensity_locked(channel, default_pct)

    # --- helpers -----------------------------------------------------------
    def _validate_channel(self, channel: int) -> None:
        if channel not in self._CHANNELS:
            raise LightboxError(f'invalid channel {channel!r}; valid channels are {list(self._CHANNELS)}')

    @classmethod
    def _decode_permille(cls, data) -> float:
        """Decode a 2-byte little-endian per-mille reply into percent (0–100)."""
        raw = int(data[0]) | (int(data[1]) << 8)
        return raw / cls._PER_MILLE * 100.0

    @staticmethod
    def _clamp_percent(percent: float) -> float:
        return max(0.0, min(100.0, float(percent)))

    def close(self) -> None:
        try:
            self._usb.util.release_interface(self._dev, self._INTERFACE)
            self._usb.util.dispose_resources(self._dev)
        except Exception:  # pragma: no cover - best-effort teardown
            logger.exception('Error closing lightSTUDIO-S')
