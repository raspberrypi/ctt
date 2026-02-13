# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# PiSP platform configuration

from importlib import resources

from .base import PlatformConfig


def get_config() -> PlatformConfig:
    data_dir = resources.files('ctt.data')
    return PlatformConfig(
        target='pisp',
        grid_size=(32, 32),
        default_template=data_dir / 'pisp_template.json',
    )
