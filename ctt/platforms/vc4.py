# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# VC4 platform configuration

from importlib import resources

from .base import PlatformConfig


def get_config() -> PlatformConfig:
    data_dir = resources.files('ctt.data')
    return PlatformConfig(
        target='bcm2835',
        grid_size=(16, 12),
        default_template=data_dir / 'vc4_template.json',
    )
