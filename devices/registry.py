# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Device driver registry + device-agnostic factories.
#
# This is the only module that imports concrete drivers, so the generic interfaces in
# lightbox.py and lightmeter.py stay dependency-free. Add a new device by appending its
# driver class to the relevant list here, or by calling the matching register function
# at runtime.

from __future__ import annotations

import logging
import threading

from .cl70f import CL70F
from .lightbox import Lightbox, LightboxError
from .lightmeter import LightMeter, LightmeterError
from .lightstudio_s import LightStudioS

logger = logging.getLogger(__name__)

# Known lightbox drivers, tried in order by get_lightbox().
LIGHTBOX_DRIVERS: list[type[Lightbox]] = [LightStudioS]

# Known light-meter drivers, tried in order by get_lightmeter().
LIGHTMETER_DRIVERS: list[type[LightMeter]] = [CL70F]


def register_lightbox_driver(driver: type[Lightbox]) -> None:
    """Register an additional lightbox driver (e.g. an out-of-tree device)."""
    if driver not in LIGHTBOX_DRIVERS:
        LIGHTBOX_DRIVERS.append(driver)


def get_lightbox(serial: str | None = None) -> Lightbox:
    """Return the first attached, supported lightbox (optionally matching `serial`).

    Walks the driver registry calling each driver's probe(); raises LightboxError if
    no supported device is present.
    """
    for driver in LIGHTBOX_DRIVERS:
        try:
            box = driver.probe(serial)
        except LightboxError:  # pragma: no cover - probe should return None, not raise
            continue
        if box is not None:
            return box
    raise LightboxError('no supported lightbox found (is one attached and powered?)')


_SHARED: Lightbox | None = None
_SHARED_LOCK = threading.Lock()


def get_shared_lightbox(serial: str | None = None) -> Lightbox:
    """Return a process-wide singleton lightbox (one device per process)."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = get_lightbox(serial)
        return _SHARED


def register_lightmeter_driver(driver: type[LightMeter]) -> None:
    """Register an additional light-meter driver (e.g. an out-of-tree device)."""
    if driver not in LIGHTMETER_DRIVERS:
        LIGHTMETER_DRIVERS.append(driver)


def get_lightmeter(serial: str | None = None) -> LightMeter:
    """Return the first attached, supported light meter (optionally matching `serial`).

    Walks the meter-driver registry calling each driver's probe(); raises LightmeterError if
    no supported device is present.
    """
    for driver in LIGHTMETER_DRIVERS:
        try:
            meter = driver.probe(serial)
        except LightmeterError:  # pragma: no cover - probe should return None, not raise
            continue
        if meter is not None:
            return meter
    raise LightmeterError('no supported light meter found (is one attached and powered?)')


_SHARED_METER: LightMeter | None = None
_SHARED_METER_LOCK = threading.Lock()


def get_shared_lightmeter(serial: str | None = None) -> LightMeter:
    """Return a process-wide singleton light meter (one device per process)."""
    global _SHARED_METER
    with _SHARED_METER_LOCK:
        if _SHARED_METER is None:
            _SHARED_METER = get_lightmeter(serial)
        return _SHARED_METER
