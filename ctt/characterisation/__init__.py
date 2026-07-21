# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Sensor characterisation: hardware-free analysis of raw Bayer bursts.
#
# One shared recipe (central ROI, single measurement channel, temporal and
# spatial statistics per operating point) feeds every experiment: dark/black
# level/DSNU, photon transfer (conversion gain + read noise), PRNU, and — with
# live sweep captures — linearity, full well and dynamic range. The engine is
# source-agnostic: a FrameSet can be built from DNGs on disk or from live raw
# arrays, and everything downstream is identical.

from .discover import CaptureGroup, scan_project
from .frames import FrameSet, MismatchedGroupError, centre_roi, frameset_from_dngs, gr_plane
from .ptc import PtcFit, PtcPoint, fit_ptc, ptc_point
from .stats import SpatialStats, TemporalStats, shading_fit, spatial_stats, temporal_stats

__all__ = [
    'CaptureGroup',
    'FrameSet',
    'MismatchedGroupError',
    'PtcFit',
    'PtcPoint',
    'SpatialStats',
    'TemporalStats',
    'centre_roi',
    'fit_ptc',
    'frameset_from_dngs',
    'gr_plane',
    'ptc_point',
    'scan_project',
    'shading_fit',
    'spatial_stats',
    'temporal_stats',
]
