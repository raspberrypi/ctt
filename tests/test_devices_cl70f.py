# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Tests for the CL-70F driver (devices.cl70f). Hardware-free: the protocol layer is
# pure, and the driver is driven against a fake transport that returns canned replies.

import struct

import pytest

from devices import LightmeterError
from devices.cl70f import protocol
from devices.cl70f.device import CL70F
from devices.lightmeter import LightmeterTimeout

# --- shared reply builders ---------------------------------------------------
ACK_REPLY = bytes([protocol.ACK, 0x30])
IDLE = 0x40  # bit6 fixed 1, no busy/warning/error
BUSY = IDLE | protocol.STA1_BUSY
REMOTE = IDLE | protocol.STA1_REMOTE
WARN = IDLE | protocol.STA1_WARNING
ERROR = IDLE | protocol.STA1_ERROR


def _data_reply(cmd: str, sta1: int = IDLE, red: bytes = b'') -> bytes:
    """Build a data response: echoed RES + STA-1 + STA-2 + KEY + RED."""
    return cmd.encode('ascii') + bytes([sta1, IDLE, IDLE]) + red


def _pack(name: str, value: float) -> tuple[int, bytes]:
    off, size = protocol._NR_OFFSETS[name]
    fmt = f'{protocol.BYTE_ORDER}{"f" if size == 4 else "d"}'
    return off, struct.pack(fmt, value)


def _make_nr_red(**values: float) -> bytes:
    """A full-length NR RED payload with the named float/double fields set."""
    red = bytearray(protocol.NR_RED_LENGTH)
    for name, value in values.items():
        off, packed = _pack(name, value)
        red[off : off + len(packed)] = packed
    return bytes(red)


class FakeTransport:
    """Returns queued replies on each receive(), recording every frame sent.

    A queued exception instance is raised instead of returned, so tests can model
    reply timeouts.
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.closed = False

    def send(self, frame):
        self.sent.append(frame)

    def receive(self, timeout_ms):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def close(self):
        self.closed = True


# --- command building --------------------------------------------------------
def test_build_command():
    assert protocol.build_command('RT', '1') == b'RT1'
    assert protocol.build_command('ST') == b'ST'
    assert protocol.build_command('EM', '0', '0001') == b'EM0,0001'
    with pytest.raises(ValueError, match='2 characters'):
        protocol.build_command('RES')


# --- acknowledgement parsing -------------------------------------------------
def test_parse_ack_accepts_ack():
    assert protocol.parse_ack(bytes([protocol.ACK, 0x30])) is None


def test_parse_ack_rejects_nak_with_code():
    with pytest.raises(LightmeterError, match='parameter out of range'):
        protocol.parse_ack(bytes([protocol.NAK, ord('2')]))


def test_parse_ack_handles_empty_and_garbage():
    with pytest.raises(LightmeterError, match='empty'):
        protocol.parse_ack(b'')
    with pytest.raises(LightmeterError, match='unexpected'):
        protocol.parse_ack(bytes([0x99]))


# --- data-response parsing + status bits -------------------------------------
def test_parse_data_response_splits_fields():
    resp = protocol.parse_data_response(_data_reply('NR', sta1=BUSY, red=b'\x01\x02'))
    assert resp.res == b'NR'
    assert resp.red == b'\x01\x02'
    assert resp.busy is True
    assert resp.warning is False
    assert resp.error is False


def test_parse_data_response_status_bits():
    assert protocol.parse_data_response(_data_reply('ST', sta1=WARN)).warning is True
    assert protocol.parse_data_response(_data_reply('ST', sta1=ERROR)).error is True
    with pytest.raises(LightmeterError, match='truncated'):
        protocol.parse_data_response(b'NR')


# --- get-warning/error decode ------------------------------------------------
def test_decode_ge_and_describe():
    red = struct.pack(f'{protocol.BYTE_ORDER}H', 5) + b',' + struct.pack(f'{protocol.BYTE_ORDER}H', 0)
    warning, error = protocol.decode_ge(red)
    assert warning == 5  # bits 0 and 2 set
    assert error == 0
    described = protocol.describe_warnings(warning)
    assert any('temperature' in w for w in described)
    assert any('lower measurement limit' in w for w in described)


# --- NR layout + decode ------------------------------------------------------
def test_nr_layout_offsets():
    # Pin the absolute byte layout: these offsets are derived independently from the
    # field/separator sizes and must not drift.
    assert protocol.NR_RED_LENGTH == 2370
    assert protocol._NR_OFFSETS['colour_temperature'] == (45, 4)
    assert protocol._NR_OFFSETS['illuminance_lx'] == (266, 4)
    assert protocol._NR_OFFSETS['tristimulus_x'] == (276, 8)
    assert protocol._NR_OFFSETS['cie1931_x'] == (303, 4)
    assert protocol._NR_OFFSETS['ra'] == (343, 4)
    assert protocol._NR_OFFSETS['r15'] == (418, 4)
    assert protocol._NR_OFFSETS['spectrum_5nm'] == (423, 81 * 4)
    assert protocol._NR_OFFSETS['spectrum_1nm'] == (748, 401 * 4)


def test_decode_nr_reads_headline_and_colorimetry():
    red = _make_nr_red(
        illuminance_lx=234.5,
        colour_temperature=6543.0,
        deviation=0.0021,
        cie1931_x=0.3127,
        cie1931_y=0.3290,
        cie1976_u=0.1978,
        cie1976_v=0.4683,
        tristimulus_x=95.047,
        tristimulus_y=100.0,
        tristimulus_z=108.883,
        ra=98.5,
    )
    m = protocol.decode_nr(red)
    assert m.illuminance_lux == pytest.approx(234.5, abs=1e-2)
    assert m.cct == pytest.approx(6543.0)
    assert m.duv == pytest.approx(0.0021, abs=1e-6)
    assert m.cie1931_xy[0] == pytest.approx(0.3127, abs=1e-5)
    assert m.cie1976_uv[1] == pytest.approx(0.4683, abs=1e-5)
    assert m.tristimulus_xyz[1] == pytest.approx(100.0)  # an 8-byte double field
    assert m.cri_ra == pytest.approx(98.5)
    assert m.spectrum_5nm is not None and len(m.spectrum_5nm.values) == 81
    assert m.spectrum_1nm is not None and len(m.spectrum_1nm.values) == 401


def test_decode_nr_is_positional_not_comma_split():
    # A binary float can contain a 0x2C (',') byte; positional decode must still work.
    off, packed = _pack('illuminance_lx', 0.0)
    lux_with_comma = struct.unpack(f'{protocol.BYTE_ORDER}f', b'\x2c\x2c\x2c\x2c')[0]
    red = _make_nr_red(illuminance_lx=lux_with_comma, colour_temperature=5000.0)
    assert b',' in red[off : off + 4]  # the field really contains comma bytes
    m = protocol.decode_nr(red)
    assert m.illuminance_lux == pytest.approx(lux_with_comma)
    assert m.cct == pytest.approx(5000.0)


def test_decode_nr_too_short_raises():
    with pytest.raises(LightmeterError, match='too short'):
        protocol.decode_nr(b'\x00' * 8)


# --- driver orchestration over a fake transport ------------------------------
def test_measure_runs_the_full_sequence():
    nr_red = _make_nr_red(illuminance_lx=321.0, colour_temperature=4200.0)
    transport = FakeTransport(
        [
            ACK_REPLY,
            _data_reply('RT', sta1=REMOTE),  # enter remote mode
            ACK_REPLY,
            _data_reply('RM', sta1=BUSY),  # start measurement
            ACK_REPLY,
            _data_reply('ST', sta1=BUSY),  # poll: still busy
            ACK_REPLY,
            _data_reply('ST', sta1=REMOTE),  # poll: done
            ACK_REPLY,
            _data_reply('NR', sta1=REMOTE, red=nr_red),
        ]
    )
    meter = CL70F(transport)
    reading = meter.measure()
    assert reading.illuminance_lux == pytest.approx(321.0)
    assert reading.cct == pytest.approx(4200.0)
    assert reading.in_range is True  # 321 lx is within the rated range
    assert transport.sent == [b'RT1', b'RM0', b'ST', b'ST', b'NR']


def _measure_once(nr_red):
    """Drive one full measure() cycle returning the given NR payload."""
    transport = FakeTransport(
        [
            ACK_REPLY,
            _data_reply('RT', sta1=REMOTE),
            ACK_REPLY,
            _data_reply('RM', sta1=BUSY),
            ACK_REPLY,
            _data_reply('ST', sta1=REMOTE),
            ACK_REPLY,
            _data_reply('NR', sta1=REMOTE, red=nr_red),
        ]
    )
    return CL70F(transport).measure()


def test_measure_flags_out_of_range_reading():
    # Below the rated 1 lx floor (the meter's under-range sentinel) → in_range False.
    reading = _measure_once(_make_nr_red(illuminance_lx=-100.0, colour_temperature=0.0))
    assert reading.in_range is False


def test_limits_reported():
    limits = CL70F(FakeTransport([])).limits
    assert limits.illuminance_min == 1.0 and limits.illuminance_max == 200000.0
    assert limits.colour_min_lux == 5.0


def test_measure_raises_on_error_state():
    transport = FakeTransport(
        [
            ACK_REPLY,
            _data_reply('RT', sta1=REMOTE),
            ACK_REPLY,
            _data_reply('RM', sta1=BUSY),
            ACK_REPLY,
            _data_reply('ST', sta1=ERROR),  # poll: error
            ACK_REPLY,
            _data_reply('GE', red=struct.pack(f'{protocol.BYTE_ORDER}HcH', 0, b',', 1)),
        ]
    )
    with pytest.raises(LightmeterError, match='error state'):
        CL70F(transport).measure()


def test_read_latest_returns_none_when_empty():
    transport = FakeTransport(
        [
            ACK_REPLY,
            _data_reply('RT', sta1=REMOTE),
            ACK_REPLY,
            _data_reply('NR', sta1=REMOTE, red=b''),  # no stored result
        ]
    )
    assert CL70F(transport).read_latest() is None


def test_measure_raises_meter_error_on_empty_result():
    # A measurement that completes but yields an empty NR must surface as LightmeterError
    # (the public contract), not the private _NoResult.
    transport = FakeTransport(
        [
            ACK_REPLY,
            _data_reply('RT', sta1=REMOTE),
            ACK_REPLY,
            _data_reply('RM', sta1=BUSY),
            ACK_REPLY,
            _data_reply('ST', sta1=REMOTE),  # poll: done
            ACK_REPLY,
            _data_reply('NR', sta1=REMOTE, red=b''),  # but no result
        ]
    )
    with pytest.raises(LightmeterError, match='no result'):
        CL70F(transport).measure()


def test_transact_resends_a_swallowed_command():
    # The meter drops the first command after sitting idle: the ACK read times out,
    # the driver drains (another timeout) and resends the same frame once.
    transport = FakeTransport(
        [
            LightmeterTimeout('no ack'),  # RT1 swallowed
            LightmeterTimeout('quiet'),  # drain finds nothing stale
            ACK_REPLY,
            _data_reply('RT', sta1=REMOTE),  # resent RT1 succeeds
            ACK_REPLY,
            _data_reply('NR', sta1=REMOTE, red=b''),
        ]
    )
    assert CL70F(transport).read_latest() is None
    assert transport.sent == [b'RT1', b'RT1', b'NR']


def test_transact_gives_up_after_one_resend():
    transport = FakeTransport(
        [
            LightmeterTimeout('no ack'),
            LightmeterTimeout('quiet'),
            LightmeterTimeout('no ack again'),
        ]
    )
    with pytest.raises(LightmeterTimeout):
        CL70F(transport).read_latest()
    assert transport.sent == [b'RT1', b'RT1']


def test_transact_accepts_a_bare_data_response():
    # A swallowed command's reply can arrive late, in place of the next ACK; a reply
    # opening with the echoed command code is taken as the data response directly.
    transport = FakeTransport(
        [
            _data_reply('RT', sta1=REMOTE),  # data response with no preceding ACK
            ACK_REPLY,
            _data_reply('NR', sta1=REMOTE, red=b''),
        ]
    )
    assert CL70F(transport).read_latest() is None
    assert transport.sent == [b'RT1', b'NR']


def test_probe_returns_none_without_hardware():
    # No USB/serial transport is configured, so probing must report "not present".
    assert CL70F.probe() is None


def test_close_releases_remote_and_transport():
    transport = FakeTransport([ACK_REPLY, _data_reply('RT', sta1=IDLE)])
    CL70F(transport).close()
    assert transport.sent == [b'RT0']  # remote mode released
    assert transport.closed is True
