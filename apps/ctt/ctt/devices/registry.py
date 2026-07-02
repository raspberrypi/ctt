# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Lightbox driver registry + device-agnostic factory.
#
# This is the only module that imports concrete drivers, so the generic interface in
# lightbox.py stays dependency-free. Add a new device by appending its driver class
# to LIGHTBOX_DRIVERS here, or by calling register_lightbox_driver() at runtime.

from __future__ import annotations

import logging
import threading

from .lightbox import Lightbox, LightboxError
from .lightstudio_s import LightStudioS

logger = logging.getLogger(__name__)

# Known lightbox drivers, tried in order by get_lightbox().
LIGHTBOX_DRIVERS: list[type[Lightbox]] = [LightStudioS]


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
