# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Generic lightbox / illumination-device API.
#
# Consumers program against the abstract Lightbox interface and the device-agnostic
# factory in ctt.devices (get_lightbox / get_shared_lightbox); concrete drivers (e.g.
# ctt.devices.lightstudio_s.LightStudioS) implement this contract and are registered
# in ctt.devices.registry. Adding a new lightbox is a new driver package plus one
# registry entry — no change here or in any consumer.
#
# This module has no hardware dependencies (drivers import it, not vice-versa).

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LightboxError(RuntimeError):
    """Raised when a lightbox is unavailable or a command fails."""


class Lightbox(ABC):
    """A controllable illumination device with one or more illuminant channels.

    Intensity is always expressed in percent (0–100). Channels are 1-based and map
    to named illuminants via `illuminants`. Concrete drivers implement the abstract
    primitives plus `probe`; the convenience methods (`set_illuminant`, `off`,
    `info`) and context-manager support are built on top of those primitives.
    """

    # --- identity (drivers may override) -----------------------------------
    @property
    def model(self) -> str:
        return type(self).__name__

    @property
    def serial(self) -> str | None:
        return None

    @property
    @abstractmethod
    def illuminants(self) -> dict[int, str]:
        """Map of channel number → illuminant name."""

    @property
    def illuminant_temps(self) -> dict[int, int]:
        """Map of channel number → nominal colour temperature (K), where known.

        Optional: drivers that can't supply nominal CCTs return {} (the default).
        Channels absent from the map have no known CCT — consumers should fall back
        (the web UI seeds 6500 K).
        """
        return {}

    # --- discovery ---------------------------------------------------------
    @classmethod
    @abstractmethod
    def probe(cls, serial: str | None = None) -> Lightbox | None:
        """Return an opened instance if this driver's hardware is attached.

        Match `serial` when given. Return None when no matching device is present —
        including when a required backend (e.g. pyusb) is unavailable. Must not raise
        for the merely-absent case, so the registry can try the next driver.
        """

    # --- control primitives (drivers implement) ----------------------------
    @abstractmethod
    def set_intensity(self, channel: int, percent: float) -> None:
        """Set `channel` to `percent` (0–100 %); also makes it the active channel."""

    @abstractmethod
    def get_intensity(self) -> float:
        """Return the active channel's intensity (0–100 %)."""

    @abstractmethod
    def get_channel(self) -> int:
        """Return the active channel number."""

    @abstractmethod
    def set_channel(self, channel: int) -> None:
        """Switch to `channel` at its stored default intensity."""

    @abstractmethod
    def get_default_intensity(self, channel: int) -> float:
        """Return `channel`'s power-on default intensity (0–100 %)."""

    @abstractmethod
    def close(self) -> None:
        """Release the device."""

    # --- convenience (built on the primitives) ------------------------------
    def set_illuminant(self, name: str, percent: float | None = None) -> int:
        """Select the channel whose illuminant matches `name` (case-insensitive).

        With `percent` None the channel is switched on at its default intensity;
        otherwise it is set to `percent`. Returns the resolved channel number.
        """
        channel = self._resolve_illuminant(name)
        if percent is None:
            self.set_channel(channel)
        else:
            self.set_intensity(channel, percent)
        return channel

    def off(self) -> None:
        """Turn the active channel off (0 %)."""
        channel = self.get_channel()
        if channel in self.illuminants:
            self.set_intensity(channel, 0)

    def info(self) -> dict:
        """A small status snapshot for UIs/CLIs."""
        channel = self.get_channel()
        return {
            'model': self.model,
            'serial': self.serial,
            'channel': channel,
            'illuminant': self.illuminants.get(channel),
            'intensity': self.get_intensity(),
            'illuminant_temps': self.illuminant_temps,
        }

    def _resolve_illuminant(self, name: str) -> int:
        key = str(name).strip().lower()
        for channel, illuminant in self.illuminants.items():
            if illuminant.lower() == key:
                return channel
        raise LightboxError(f'unknown illuminant {name!r}; known: {sorted(self.illuminants.values())}')

    # --- context manager ---------------------------------------------------
    def __enter__(self) -> Lightbox:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
