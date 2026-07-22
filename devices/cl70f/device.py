# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Konica Minolta CL-70F driver (a concrete devices.LightMeter).
#
# Orchestrates the command/response protocol (protocol.py) over a byte transport
# (transport.py): enter remote mode, take a measurement, poll for completion and read
# the result. All public methods are serialised under a lock so callers can share one
# device.
#
# Non-root USB access on the Pi: install contrib/99-cl70f.rules to /etc/udev/rules.d/
# then `sudo udevadm control --reload && sudo udevadm trigger`.

from __future__ import annotations

import dataclasses
import logging
import threading
import time

from ..lightmeter import LightMeter, LightmeterError, LightmeterTimeout, Measurement, MeasurementLimits
from . import protocol
from .transport import Transport, open_transport

logger = logging.getLogger(__name__)


class CL70F(LightMeter):
    """Konica Minolta CL-70F CRI illuminance meter.

    Reports illuminance (lux/fc), correlated colour temperature, Δuv, full CIE
    colorimetry, CRI (Ra + R1–R15) and the 380–780 nm spectrum (see Measurement).
    """

    # Reply timeouts, with headroom: a quick acknowledgement and data response for most
    # commands, and a longer budget for the operation commands (measure / dark calibrate).
    _ACK_TIMEOUT_MS = 1000
    _DATA_TIMEOUT_MS = 2000
    _OPERATION_TIMEOUT_MS = 8000
    # Short read used to discard stale replies before resending a command.
    _DRAIN_TIMEOUT_MS = 100
    # A measurement completes asynchronously; poll ST until the busy bit clears.
    _MEASURE_TIMEOUT_S = 30.0
    _POLL_INTERVAL_S = 0.2

    # Rated measurement range (CL-70F spec): illuminance 1–200,000 lx, colour needs
    # >= 5 lx, CCT 1,563–100,000 K. Below range the meter returns sentinel values
    # (e.g. -100 lx), which we flag rather than trust.
    LIMITS = MeasurementLimits(
        illuminance_min=1.0,
        illuminance_max=200_000.0,
        colour_min_lux=5.0,
        cct_min=1563.0,
        cct_max=100_000.0,
    )

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self._lock = threading.Lock()
        # The transport reports the USB/serial descriptor serial where available.
        self._serial = getattr(transport, 'serial', None)

    # --- discovery ---------------------------------------------------------
    @classmethod
    def probe(cls, serial: str | None = None) -> CL70F | None:
        """Return an opened CL-70F if one is attached, else None."""
        try:
            transport = open_transport(serial)
        except LightmeterError:
            return None
        if transport is None:
            return None
        return cls(transport)

    # --- identity ----------------------------------------------------------
    @property
    def model(self) -> str:
        return 'CL-70F'

    @property
    def serial(self) -> str | None:
        return self._serial

    @property
    def limits(self) -> MeasurementLimits:
        return self.LIMITS

    # --- measurement -------------------------------------------------------
    def measure(self) -> Measurement:
        """Take a fresh measurement and return the result.

        Enters remote mode, starts a measurement, polls until it completes (raising on
        an error state, logging any out-of-range/temperature warnings) and reads the
        latest result.
        """
        with self._lock:
            self._ensure_remote_locked()
            self._transact_locked('RM', '0', timeout_ms=self._OPERATION_TIMEOUT_MS)
            self._poll_idle_locked()
            try:
                return self._read_result_locked()
            except _NoResult:
                raise LightmeterError('CL-70F returned no result after measurement') from None

    def read_latest(self) -> Measurement | None:
        """Return the meter's most recent stored result, or None if it holds none."""
        with self._lock:
            self._ensure_remote_locked()
            try:
                return self._read_result_locked()
            except _NoResult:
                return None

    def calibrate(self) -> None:
        """Run a dark calibration and wait for it to complete.

        Worth running before measuring after a temperature change or a filter move.
        """
        with self._lock:
            self._ensure_remote_locked()
            self._transact_locked('DC', timeout_ms=self._OPERATION_TIMEOUT_MS)
            self._poll_idle_locked()

    def equipment_serial(self) -> str | None:
        """Query the equipment serial number (SN command), independent of the USB
        descriptor serial. Returns None if the device does not report one."""
        with self._lock:
            resp = self._transact_locked('SN', 'r', timeout_ms=self._DATA_TIMEOUT_MS)
        text = resp.red.split(b'\x00', 1)[0].decode('ascii', 'replace').strip()
        return text or None

    def close(self) -> None:
        # Release remote mode so the device's buttons work again; best-effort.
        try:
            with self._lock:
                self._transact_locked('RT', '0', timeout_ms=self._DATA_TIMEOUT_MS)
        except LightmeterError:  # pragma: no cover - device may already be gone
            pass
        self._transport.close()

    # --- protocol orchestration (callers hold self._lock) ------------------
    def _transact_locked(self, cmd: str, *params: str, timeout_ms: int) -> protocol.DataResponse:
        """Send a command and return its parsed data response, validating the ACK.

        The meter silently swallows the first command it receives after sitting idle,
        so a command whose acknowledgement never arrives is resent once (after draining
        any stale replies). A swallowed command's reply can also turn up late, in front
        of the resent command's ACK — a reply that opens with the echoed command code
        where the ACK belongs is therefore accepted as the data response directly.
        """
        frame = protocol.build_command(cmd, *params)
        for attempt in range(2):
            self._transport.send(frame)
            try:
                reply = self._transport.receive(self._ACK_TIMEOUT_MS)
                break
            except LightmeterTimeout:
                if attempt:
                    raise
                self._drain_locked()
        if reply[:2] == frame[:2]:
            return protocol.parse_data_response(reply)
        protocol.parse_ack(reply)
        return protocol.parse_data_response(self._transport.receive(timeout_ms))

    def _drain_locked(self) -> None:
        """Discard any stale buffered replies until the device goes quiet."""
        while True:
            try:
                stale = self._transport.receive(self._DRAIN_TIMEOUT_MS)
            except LightmeterTimeout:
                return
            logger.debug('CL-70F: discarded stale reply %r', stale[:16])

    def _ensure_remote_locked(self) -> None:
        """Put the device into remote mode (required for all but RT/ST/MN/FV/FB)."""
        self._transact_locked('RT', '1', timeout_ms=self._DATA_TIMEOUT_MS)

    def _read_result_locked(self) -> Measurement:
        """Read and decode the latest measurement (NR). Raises _NoResult if empty.

        Flags the reading out of range (in_range False) when its illuminance falls
        outside the meter's rated limits — below range the device fills the fields
        with sentinels (e.g. -100 lx), which are not a real measurement.
        """
        resp = self._transact_locked('NR', timeout_ms=self._DATA_TIMEOUT_MS)
        if not resp.red:
            raise _NoResult
        m = protocol.decode_nr(resp.red)
        in_range = self.LIMITS.illuminance_min <= m.illuminance_lux <= self.LIMITS.illuminance_max
        return dataclasses.replace(m, in_range=in_range)

    def _poll_idle_locked(self) -> None:
        """Poll status (ST) until the busy bit clears, raising on an error state."""
        deadline = time.monotonic() + self._MEASURE_TIMEOUT_S
        while True:
            resp = self._transact_locked('ST', timeout_ms=self._DATA_TIMEOUT_MS)
            if resp.error:
                warning, error = self._read_codes_locked()
                raise LightmeterError(f'CL-70F entered an error state (warning={warning:#06x} error={error:#06x})')
            if not resp.busy:
                if resp.warning:
                    warning, _ = self._read_codes_locked()
                    logger.warning('CL-70F measurement warnings: %s', '; '.join(protocol.describe_warnings(warning)))
                return
            if time.monotonic() > deadline:
                raise LightmeterError('timed out waiting for the CL-70F measurement to complete')
            time.sleep(self._POLL_INTERVAL_S)

    def _read_codes_locked(self) -> tuple[int, int]:
        """Read the warning/error code words (GE command)."""
        resp = self._transact_locked('GE', timeout_ms=self._DATA_TIMEOUT_MS)
        return protocol.decode_ge(resp.red)


class _NoResult(Exception):
    """Internal: the device reported no stored measurement (NR with an empty payload)."""
