# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for ctt_server filename construction and validation.

import pytest

from ctt.core.camera import get_col_lux
from ctt_server import naming


def test_macbeth_roundtrips_through_ctt_parser():
    name = naming.build_filename('macbeth', 5858, lux=1344, label='D65')
    assert name == 'd65_5858k_1344l.dng'
    col, lux = get_col_lux(name)
    assert (col, lux) == (5858, 1344)
    assert naming.detect_type(name) == 'macbeth'


def test_alsc_roundtrips_and_detects():
    name = naming.build_filename('alsc', 5000, index=3)
    assert name == 'alsc_5000k_3.dng'
    col, _ = get_col_lux(name)
    assert col == 5000
    assert naming.detect_type(name) == 'alsc'


def test_cac_roundtrips_and_detects():
    name = naming.build_filename('cac', 4000, index=0)
    assert name == 'cac_4000k_0.dng'
    col, _ = get_col_lux(name)
    assert col == 4000
    assert naming.detect_type(name) == 'cac'


def test_macbeth_requires_lux():
    with pytest.raises(naming.NamingError):
        naming.build_filename('macbeth', 5000, label='d65')


def test_macbeth_label_must_not_collide_with_type_markers():
    # Labels containing alsc/cac would be misdetected as those image types.
    with pytest.raises(naming.NamingError):
        naming.build_filename('macbeth', 5000, lux=800, label='alsc_room')
    with pytest.raises(naming.NamingError):
        naming.build_filename('macbeth', 5000, lux=800, label='cacophony')


def test_macbeth_name_contains_neither_marker():
    name = naming.build_filename('macbeth', 5000, lux=800, label='Daylight65')
    assert 'alsc' not in name and 'cac' not in name


def test_colour_temp_precedes_lux():
    name = naming.build_filename('macbeth', 5858, lux=1344, label='d65')
    assert name.index('5858k') < name.index('1344l')


def test_sanitise_label():
    assert naming.sanitise_label('D 65!') == 'd65'
    assert naming.sanitise_label('F-11') == 'f11'


def test_invalid_inputs_raise():
    with pytest.raises(naming.NamingError):
        naming.build_filename('bogus', 5000)
    with pytest.raises(naming.NamingError):
        naming.build_filename('alsc', 0)


@pytest.mark.parametrize(
    'name,ok',
    [
        ('d65_5858k_1344l.dng', True),
        ('alsc_5000k_0.dng', True),
        ('cac_4000k_1.dng', True),
        ('d65_5858k.dng', False),  # macbeth missing lux
        ('noexif.dng', False),  # no colour temp
        ('photo.jpg', False),  # not a dng
    ],
)
def test_validate_filename(name, ok):
    valid, _ = naming.validate_filename(name)
    assert valid is ok


def test_next_index_increments():
    existing = ['alsc_5000k_0.dng', 'alsc_5000k_1.dng', 'alsc_3000k_0.dng']
    assert naming.next_index(existing, 'alsc', 5000) == 2
    assert naming.next_index(existing, 'alsc', 3000) == 1
    assert naming.next_index(existing, 'alsc', 4000) == 0


def test_parse_filename_macbeth():
    assert naming.parse_filename('d65_5858k_1344l.dng') == ('macbeth', 5858, 1344, 'd65')


def test_parse_filename_alsc():
    assert naming.parse_filename('alsc_3000k_2.dng') == ('alsc', 3000, None, None)


def test_parse_filename_cac():
    assert naming.parse_filename('cac_4500k_0.dng') == ('cac', 4500, None, None)


def test_parse_filename_rejects_untagged():
    with pytest.raises(naming.NamingError):
        naming.parse_filename('IMG_1234.dng')
