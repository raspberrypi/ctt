# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Base class for calibration algorithms

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.camera import Camera


class CalibrationAlgorithm(ABC):
    json_key: str = ''

    def __init__(self, camera: Camera, platform: object, **config: object) -> None:
        self.camera = camera
        self.platform = platform

    @abstractmethod
    def run(self) -> dict | None:
        """Return dict fragment to merge into json[json_key], or None to skip."""
        ...
