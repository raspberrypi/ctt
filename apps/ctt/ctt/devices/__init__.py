# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Generic, pluggable device interfaces for CTT (currently: controllable lightboxes).

from .lightbox import Lightbox, LightboxError, LightboxState
from .registry import DRIVERS, get_lightbox, get_shared_lightbox, register_driver

__all__ = [
    'DRIVERS',
    'Lightbox',
    'LightboxError',
    'LightboxState',
    'get_lightbox',
    'get_shared_lightbox',
    'register_driver',
]
