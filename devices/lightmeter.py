# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Generic light-meter / illuminance-measurement device API.
#
# This is the measurement counterpart to the Lightbox control API: consumers program
# against the abstract LightMeter interface and the device-agnostic factory in
# devices (get_lightmeter / get_shared_lightmeter); concrete drivers (e.g.
# devices.cl70f.CL70F) implement this contract and are registered in
# devices.registry. Adding a new meter is a new driver package plus one registry
# entry — no change here or in any consumer.
#
# This module has no hardware dependencies (drivers import it, not vice-versa).

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Spectrum:
    """A spectral power distribution sampled on a regular wavelength grid.

    `values` are the (relative) spectral readings at `start_nm`, `start_nm + step_nm`,
    … in nanometres. Kept as a compact triple rather than a {wavelength: value} dict so
    it round-trips cleanly through JSON and is cheap to carry around.
    """

    start_nm: float
    step_nm: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class Measurement:
    """A single light-meter reading, in device-independent units.

    The headline fields (illuminance in lux and correlated colour temperature) are
    always present; every other field is optional so a minimal meter can populate just
    what it measures and leave the rest None. Colorimetry follows CIE conventions:
    `cie1931_xy` is (x, y), `cie1976_uv` is (u', v'), `cie1960_uv` is (u, v) and `duv`
    is the signed deviation from the Planckian locus (Δuv).
    """

    # --- always present ---
    illuminance_lux: float
    cct: float  # correlated colour temperature (K)

    # --- common colorimetry (optional) ---
    duv: float | None = None  # deviation from the Planckian locus (Δuv)
    cie1931_xy: tuple[float, float] | None = None
    cie1976_uv: tuple[float, float] | None = None  # (u', v')
    cie1960_uv: tuple[float, float] | None = None  # (u, v)

    # --- rich extras (driver-dependent) ---
    illuminance_fc: float | None = None  # foot-candles
    tristimulus_xyz: tuple[float, float, float] | None = None
    dominant_wavelength: float | None = None  # nm
    excitation_purity: float | None = None
    cri_ra: float | None = None  # general colour-rendering index (Ra)
    cri_ri: tuple[float, ...] | None = None  # special indices R1…R15
    spectrum_5nm: Spectrum | None = None  # 380–780 nm, 5 nm pitch
    spectrum_1nm: Spectrum | None = None  # 380–780 nm, 1 nm pitch

    def to_dict(self) -> dict:
        """A JSON-friendly dict, dropping None fields so the payload stays compact."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class LightmeterError(RuntimeError):
    """Raised when a light meter is unavailable or a command fails."""


class LightMeter(ABC):
    """A light-measurement device that reports illuminance and colour.

    Concrete drivers implement `probe`, `measure` and `close`; the public helpers
    (`info`, context-manager support) build on those. `measure()` triggers a fresh
    reading; `read_latest()` returns the device's last reading without re-triggering,
    where the hardware supports it.
    """

    # --- identity (drivers may override) -----------------------------------
    @property
    def model(self) -> str:
        return type(self).__name__

    @property
    def serial(self) -> str | None:
        return None

    # --- discovery ---------------------------------------------------------
    @classmethod
    @abstractmethod
    def probe(cls, serial: str | None = None) -> LightMeter | None:
        """Return an opened instance if this driver's hardware is attached.

        Match `serial` when given. Return None when no matching device is present —
        including when a required backend (e.g. pyusb) is unavailable. Must not raise
        for the merely-absent case, so the registry can try the next driver.
        """

    # --- measurement -------------------------------------------------------
    @abstractmethod
    def measure(self) -> Measurement:
        """Trigger a measurement and return the result. Raises LightmeterError on failure."""

    def read_latest(self) -> Measurement | None:
        """Return the device's most recent reading without triggering a new one.

        Defaults to None for meters that cannot recall a stored result; drivers that
        can read back their last reading override this.
        """
        return None

    @abstractmethod
    def close(self) -> None:
        """Release the device."""

    # --- introspection -----------------------------------------------------
    def info(self) -> dict:
        """An identity/capability snapshot for UIs and CLIs (no measurement taken)."""
        return {
            'model': self.model,
            'serial': self.serial,
        }

    # --- context manager ---------------------------------------------------
    def __enter__(self) -> LightMeter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
