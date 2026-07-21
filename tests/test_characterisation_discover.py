# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for characterisation discovery: burst grouping by filename stem with
# exact EXIF operating-point verification. EXIF reads are stubbed, so these
# run without real DNGs.

import pytest

from ctt.characterisation import discover
from ctt.characterisation.discover import CaptureGroup, classify, group_key, scan_project

# A plausible operating point; tests override fields per file as needed.
BASE = {'exposure_us': 33333, 'iso': 100, 'width': 968, 'height': 548, 'sigbits': 16, 'blacklevel': 3840}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A fake project dir + a per-filename EXIF table the stub serves from."""
    exif: dict[str, dict] = {}

    def add(name, **overrides):
        (tmp_path / name).write_bytes(b'DNG')
        exif[name] = {**BASE, **overrides}

    def fake_read_exif(path):
        meta = exif[path.name]
        if meta.get('_raise'):
            raise KeyError('ExposureTime')
        return {k: v for k, v in meta.items() if not k.startswith('_')}

    monkeypatch.setattr(discover, '_read_exif', fake_read_exif)
    return tmp_path, add


def test_group_key_strips_trailing_index():
    assert group_key('dark_3.dng') == 'dark.dng'
    assert group_key('alsc_3500k_11.dng') == 'alsc_3500k.dng'
    assert group_key('imx662_2910k_1250l_0.dng') == 'imx662_2910k_1250l.dng'
    assert group_key('imx662_2910k_1250l.dng') == 'imx662_2910k_1250l.dng'  # index-free stays


def test_classify_kinds():
    assert classify('dark_0.dng') == 'dark'
    assert classify('alsc_3500k_0.dng') == 'flat'
    assert classify('cac_5000k_0.dng') is None
    assert classify('imx662_2910k_1250l_0.dng') == 'chart'


def test_scan_groups_bursts_and_reads_tags(workspace):
    root, add = workspace
    for i in range(3):
        add(f'dark_{i}.dng')
    for i in range(2):
        add(f'alsc_3500k_{i}.dng', exposure_us=133, iso=107, blacklevel=3200)
    add('imx662_2910k_1250l_0.dng', exposure_us=5000)

    groups = {g.label: g for g in scan_project(root)}
    assert set(groups) == {'dark', 'alsc_3500k', 'imx662_2910k_1250l'}

    dark = groups['dark']
    assert dark.kind == 'dark' and len(dark.paths) == 3
    assert dark.exposure_us == 33333 and dark.gain == 1.0
    assert dark.blacklevel_16 == 3840  # sigbits 16: no shift

    flat = groups['alsc_3500k']
    assert flat.kind == 'flat' and flat.gain == pytest.approx(1.07)
    assert flat.colour_temp == 3500 and flat.lux is None

    chart = groups['imx662_2910k_1250l']
    assert chart.kind == 'chart' and chart.colour_temp == 2910 and chart.lux == 1250


def test_scan_splits_mixed_operating_points(workspace):
    root, add = workspace
    add('dark_0.dng')
    add('dark_1.dng')
    add('dark_2.dng', exposure_us=66666)  # different exposure: its own group

    groups = sorted(scan_project(root), key=lambda g: g.label)
    assert [g.label for g in groups] == ['dark', 'dark#2']
    assert [len(g.paths) for g in groups] == [2, 1]
    assert any('operating points' in w for w in groups[0].warnings)


def test_scan_skips_excluded_and_cac(workspace):
    root, add = workspace
    add('dark_0.dng')
    add('dark_1.dng')
    add('cac_5000k_0.dng')

    groups = scan_project(root, excluded={'dark_1.dng'})
    assert len(groups) == 1
    assert [p.name for p in groups[0].paths] == ['dark_0.dng']


def test_scan_warns_and_skips_unreadable_metadata(workspace):
    root, add = workspace
    add('dark_0.dng')
    add('dark_1.dng', _raise=True)

    groups = scan_project(root)
    assert len(groups) == 1 and len(groups[0].paths) == 1
    assert any('unreadable metadata' in w for w in groups[0].warnings)


def test_native_blacklevel_scales_to_16bit_domain():
    g = CaptureGroup(
        label='x',
        kind='dark',
        paths=[],
        exposure_us=1,
        gain=1.0,
        width=1,
        height=1,
        sigbits=12,
        blacklevel=200,
    )
    assert g.blacklevel_16 == 200 << 4
