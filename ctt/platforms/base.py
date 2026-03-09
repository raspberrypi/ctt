# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Platform configuration base class

from __future__ import annotations

from pathlib import Path


class PlatformConfig:
    def __init__(self, target: str, grid_size: tuple, default_template: Path | None):
        self.target = target
        self.grid_size = grid_size
        self.default_template = default_template
