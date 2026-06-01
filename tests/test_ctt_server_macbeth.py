# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the additive live-finder helper in ctt.detection.macbeth.

import numpy as np

from ctt.detection import macbeth


def test_locate_chart_returns_none_when_no_chart():
    # A uniform frame has no chart; locate_chart must return None (not raise).
    img = np.full((480, 640, 3), 40, dtype=np.uint8)
    assert macbeth.locate_chart(img) is None
