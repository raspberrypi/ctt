# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Transport seam for the CL-70F driver.
#
# The device attaches as a vendor-class USB device (0x0a41:0x7003) driven over raw
# libusb bulk endpoints. Command frames are sent bare — the transfer boundary delimits
# a frame, and the device rejects a trailing CR/LF as a parameter error.
#
# A Transport moves bytes only — the ACK/data-response sequencing lives in the driver,
# which keeps this layer trivial and lets tests inject a fake transport.

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from ..lightmeter import LightmeterError, LightmeterTimeout

logger = logging.getLogger(__name__)


@runtime_checkable
class Transport(Protocol):
    """A byte pipe to the meter: send a command frame, receive one reply message."""

    def send(self, frame: bytes) -> None:
        """Write one command frame to the device."""

    def receive(self, timeout_ms: int) -> bytes:
        """Read one reply message (an ACK/NAK or a data response).

        Raises LightmeterTimeout when nothing arrives in time, LightmeterError on
        any other transport failure.
        """

    def close(self) -> None:
        """Release the device."""


class UsbTransport:
    """Raw-USB (libusb/pyusb) transport for the vendor-class device.

    pyusb is lazily imported so core CTT is unaffected when it is absent, and
    construction fails with LightmeterError (not a crash) when the backend or device is
    missing, so probe() can fall through to None.
    """

    # Vendor/product id and bulk endpoint addresses for the device.
    VENDOR_ID = 0x0A41
    PRODUCT_ID = 0x7003
    _CONFIGURATION = 1
    _INTERFACE = 0
    _EP_OUT = 0x02  # bulk OUT endpoint address
    _EP_IN = 0x81  # bulk IN endpoint address
    _MAX_REPLY = 4096  # largest reply is the full measurement result (~2.4 kB)

    def __init__(self, serial: str | None = None) -> None:
        try:
            import usb.core  # noqa: PLC0415 (lazy import)
            import usb.util  # noqa: PLC0415
        except ImportError as err:  # pragma: no cover - depends on host environment
            raise LightmeterError(
                'pyusb is not available. Install the devices extra (pip install -e ".[devices]") '
                'and the libusb backend (sudo apt install libusb-1.0-0).'
            ) from err

        self._usb = usb
        try:
            dev = usb.core.find(idVendor=self.VENDOR_ID, idProduct=self.PRODUCT_ID)
        except usb.core.NoBackendError as err:  # pragma: no cover - depends on host environment
            raise LightmeterError('no libusb backend available (sudo apt install libusb-1.0-0).') from err
        if dev is None:
            raise LightmeterError('CL-70F not found on USB. Is it plugged in and powered?')
        if serial is not None:
            sn = self._read_string(dev, dev.iSerialNumber)
            if sn is None or serial not in sn:
                raise LightmeterError(f'CL-70F with serial {serial!r} not found.')

        try:
            if dev.is_kernel_driver_active(self._INTERFACE):
                dev.detach_kernel_driver(self._INTERFACE)
        except Exception:  # pragma: no cover - platform dependent
            pass
        dev.set_configuration(self._CONFIGURATION)
        usb.util.claim_interface(dev, self._INTERFACE)
        self._dev = dev
        self._serial = self._read_string(dev, dev.iSerialNumber)
        logger.info('CL-70F opened over USB (%#06x:%#06x serial=%s)', self.VENDOR_ID, self.PRODUCT_ID, self._serial)

    def _read_string(self, dev, index) -> str | None:
        if not index:
            # The device carries no serial-number string descriptor.
            return None
        try:
            return self._usb.util.get_string(dev, index)
        except (ValueError, self._usb.core.USBError):  # pragma: no cover - permission/descriptor dependent
            return None

    @property
    def serial(self) -> str | None:
        return self._serial

    def send(self, frame: bytes) -> None:
        try:
            self._dev.write(self._EP_OUT, frame)
        except self._usb.core.USBError as err:
            raise LightmeterError(f'USB write to the CL-70F failed: {err}') from err

    def receive(self, timeout_ms: int) -> bytes:
        try:
            return bytes(self._dev.read(self._EP_IN, self._MAX_REPLY, timeout_ms))
        except self._usb.core.USBTimeoutError as err:
            raise LightmeterTimeout('timed out waiting for a reply from the CL-70F') from err
        except self._usb.core.USBError as err:
            raise LightmeterError(f'USB read from the CL-70F failed: {err}') from err

    def close(self) -> None:
        try:
            self._usb.util.release_interface(self._dev, self._INTERFACE)
            self._usb.util.dispose_resources(self._dev)
        except Exception:  # pragma: no cover - best-effort teardown
            logger.exception('Error closing CL-70F USB transport')


def open_transport(serial: str | None = None) -> Transport | None:
    """Open the CL-70F USB transport, or None if the device is not present.

    UsbTransport raises LightmeterError when the backend or device is absent; we
    swallow that and return None so probe() reports "not present" rather than failing.
    """
    try:
        return UsbTransport(serial=serial)
    except LightmeterError:
        return None
