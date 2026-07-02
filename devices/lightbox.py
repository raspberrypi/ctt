# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Generic lightbox / illumination-device API.
#
# Consumers program against the abstract Lightbox interface and the device-agnostic
# factory in devices (get_lightbox / get_shared_lightbox); concrete drivers (e.g.
# devices.lightstudio_s.LightStudioS) implement this contract and are registered
# in devices.registry. Adding a new lightbox is a new driver package plus one
# registry entry — no change here or in any consumer.
#
# This module has no hardware dependencies (drivers import it, not vice-versa).

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import NamedTuple

logger = logging.getLogger(__name__)


class LightboxState(NamedTuple):
    """The live device state — one channel is active at a time."""

    channel: int
    illuminant: str | None
    intensity: float


class LightboxError(RuntimeError):
    """Raised when a lightbox is unavailable or a command fails."""


def _fold(name: str) -> str:
    """Normalise an illuminant name for matching: drop case, spacing and punctuation."""
    return ''.join(ch for ch in str(name).lower() if ch.isalnum())


class Lightbox(ABC):
    """A controllable illumination device with one or more illuminant channels.

    Intensity is always expressed in percent (0–100). Channels are 1-based and map
    to named illuminants via `illuminants`; every public method accepts either the
    illuminant name (case-insensitive, e.g. 'D65') or the channel number. Concrete
    drivers implement the abstract per-channel primitives plus `probe`; the public
    methods and context-manager support are built on top of those primitives.

    Only one channel is active at a time, and selecting an intensity is what
    switches channels — intensities cannot be staged on an inactive channel (the
    only stored per-channel value is the device's power-on default).
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
        """Map of channel number → canonical illuminant name (short, e.g. 'D65')."""

    @property
    def illuminant_labels(self) -> dict[int, str]:
        """Map of channel number → human-readable description, for UIs.

        Defaults to the canonical names; drivers override where a fuller
        description helps (e.g. 'HalogenBF' → 'Halogen + blue filter (400 lux)').
        Labels are also accepted anywhere an illuminant name is.
        """
        return self.illuminants

    @property
    def illuminant_aliases(self) -> dict[int, set[str]]:
        """Map of channel number → extra accepted names (alternate spellings).

        Matching already ignores case, spacing and punctuation; this is for
        genuinely different names. Defaults to none.
        """
        return {}

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

    # --- control primitives (drivers implement, channel numbers only) -------
    @abstractmethod
    def _set_intensity(self, channel: int, percent: float) -> None:
        """Set `channel` to `percent` (0–100 %); also makes it the active channel."""

    @abstractmethod
    def _get_state(self) -> tuple[int, float]:
        """Return (active channel, its intensity 0–100 %), read atomically."""

    @abstractmethod
    def _set_channel(self, channel: int) -> None:
        """Switch to `channel` at its stored default intensity."""

    @abstractmethod
    def _get_default_intensity(self, channel: int) -> float:
        """Return `channel`'s power-on default intensity (0–100 %)."""

    @abstractmethod
    def close(self) -> None:
        """Release the device."""

    # --- public control (illuminant = name or channel number) ---------------
    def set_illuminant(self, illuminant: int | str, percent: float | None = None) -> int:
        """Switch to an illuminant, the single write entry point.

        `illuminant` is a name (case-insensitive, e.g. 'D65'), a channel number,
        or a driver enum member. With `percent` None the illuminant comes on at
        its stored default intensity, otherwise at `percent` (0–100 %). Returns
        the resolved channel number.

        Intensities cannot be staged on an inactive channel: setting one always
        switches to that channel (the only stored per-channel value is the
        device's power-on default).
        """
        channel = self._resolve(illuminant)
        if percent is None:
            self._set_channel(channel)
        else:
            self._set_intensity(channel, percent)
        return channel

    def get_state(self) -> LightboxState:
        """Return the live state: the active channel, its illuminant name and
        intensity — read atomically (channel and intensity change together)."""
        channel, intensity = self._get_state()
        return LightboxState(channel, self.illuminants.get(channel), intensity)

    def get_default_intensity(self, illuminant: int | str) -> float:
        """Return an illuminant's power-on default intensity (0–100 %)."""
        return self._get_default_intensity(self._resolve(illuminant))

    def off(self) -> None:
        """Turn the active channel off (0 %)."""
        channel, _ = self._get_state()
        if channel in self.illuminants:
            self._set_intensity(channel, 0)

    def info(self) -> dict:
        """The complete status snapshot for UIs/CLIs: identity, capabilities
        (illuminant maps) and live state (active channel + intensity)."""
        state = self.get_state()
        return {
            'model': self.model,
            'serial': self.serial,
            'channel': state.channel,
            'illuminant': state.illuminant,
            'intensity': state.intensity,
            'illuminants': self.illuminants,
            'illuminant_labels': self.illuminant_labels,
            'illuminant_temps': self.illuminant_temps,
        }

    def _resolve(self, illuminant: int | str) -> int:
        """Resolve an illuminant name or channel number to a channel number.

        Name matching ignores case, spacing and punctuation, and accepts the
        canonical names, the descriptive labels and any driver aliases; numeric
        strings are treated as channel numbers.
        """
        if isinstance(illuminant, int):
            if illuminant in self.illuminants:
                return illuminant
            raise LightboxError(f'invalid channel {illuminant!r}; valid channels are {sorted(self.illuminants)}')
        key = _fold(illuminant)
        if key.isdigit():
            return self._resolve(int(key))
        for channel, name in self.illuminants.items():
            names = {name, self.illuminant_labels.get(channel, '')} | set(self.illuminant_aliases.get(channel, ()))
            if key in {_fold(n) for n in names if n}:
                return channel
        raise LightboxError(f'unknown illuminant {illuminant!r}; known: {sorted(self.illuminants.values())}')

    # --- context manager ---------------------------------------------------
    def __enter__(self) -> Lightbox:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
