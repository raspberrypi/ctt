# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Konica Minolta CL-70F interface protocol — pure, I/O-free encode/decode.
#
# This module is the single place that knows the command frames and the byte layout of
# the responses. It has no transport dependency: build_command() returns the bytes to
# send and the parse_*/decode_* helpers turn raw reply bytes into Python values. The
# driver (device.py) wires these to a transport.
#
# Frames:
#   command       = ASCII CMD (2 chars) immediately followed by PRM (comma-separated)
#   ack reply     = ACK(06H)/NAK(15H) + ERR(1 byte)
#   data response = RES(2) + STA-1(1) + STA-2(1) + KEY(1) + RED(payload)
# No checksum or terminator is carried; a terminator, if ever needed, is the
# transport's concern, not this layer's.

from __future__ import annotations

import struct
from dataclasses import dataclass

from ..lightmeter import LightmeterError, Measurement, Spectrum

# --- byte order of binary numeric fields -------------------------------------
# Numeric fields are IEEE-754 single (4-byte) or double (8-byte) precision, big-endian.
BYTE_ORDER = '>'

# --- acknowledgement bytes ---------------------------------------------------
ACK = 0x06
NAK = 0x15

# Command-error codes carried in the ERR byte of a NAK.
COMMAND_ERRORS = {
    '1': 'unknown command code',
    '2': 'parameter out of range',
    '3': 'command not supported by this model',
    '4': 'command requires remote mode',
}

# --- STA-1 status bits. bit6 is fixed 1 so the byte stays printable. ---------
STA1_BUSY = 0x01
STA1_REMOTE = 0x02
STA1_ADJUST = 0x04
STA1_WARNING = 0x08
STA1_ERROR = 0x10

# --- STA-2 busy-detail bits --------------------------------------------------
STA2_INITIALISING = 0x01
STA2_COMMAND = 0x02
STA2_DARK_CALIBRATION = 0x04
STA2_LIGHT_MEASUREMENT = 0x08
STA2_CORDLESS_STANDBY = 0x10

# --- warning codes from the get-warning/error command ------------------------
WARNING_CODES = {
    0: 'temperature change since dark calibration exceeded the acceptable value',
    1: 'dark calibration incomplete (filter moved during calibration)',
    2: 'illuminance/intensity below the lower measurement limit',
    3: 'illuminance/intensity above the upper measurement limit',
}


@dataclass(frozen=True)
class DataResponse:
    """A decoded data response: the echoed command, status bytes and RED payload."""

    res: bytes  # echoed 2-char command code
    sta1: int
    sta2: int
    key: int
    red: bytes

    @property
    def busy(self) -> bool:
        return bool(self.sta1 & STA1_BUSY)

    @property
    def warning(self) -> bool:
        return bool(self.sta1 & STA1_WARNING)

    @property
    def error(self) -> bool:
        return bool(self.sta1 & STA1_ERROR)


# --- command building --------------------------------------------------------
def build_command(cmd: str, *params: str) -> bytes:
    """Build a command frame: the 2-char code immediately followed by comma-joined PRM.

    e.g. build_command('RT', '1') -> b'RT1'; build_command('ST') -> b'ST';
    build_command('EM', '0', '0001') -> b'EM0,0001'. No terminator is appended.
    """
    if len(cmd) != 2:
        raise ValueError(f'command code must be 2 characters, got {cmd!r}')
    return (cmd + ','.join(params)).encode('ascii')


# --- acknowledgement parsing -------------------------------------------------
def parse_ack(reply: bytes) -> None:
    """Validate an ACK/NAK reply. Raise LightmeterError on NAK or a malformed reply."""
    if not reply:
        raise LightmeterError('empty acknowledgement from CL-70F')
    if reply[0] == ACK:
        return
    if reply[0] == NAK:
        err = chr(reply[1]) if len(reply) > 1 else '?'
        raise LightmeterError(f'CL-70F rejected command (NAK, error {err!r}: {COMMAND_ERRORS.get(err, "unknown")})')
    raise LightmeterError(f'unexpected acknowledgement byte {reply[0]:#04x} from CL-70F')


# --- data-response parsing ---------------------------------------------------
def parse_data_response(reply: bytes) -> DataResponse:
    """Split a data response into RES + STA-1 + STA-2 + KEY + RED."""
    if len(reply) < 5:
        raise LightmeterError(f'truncated data response from CL-70F ({len(reply)} bytes)')
    return DataResponse(res=reply[0:2], sta1=reply[2], sta2=reply[3], key=reply[4], red=reply[5:])


def describe_warnings(warning_bits: int) -> list[str]:
    """Human-readable warnings set in a warning-code value."""
    return [text for bit, text in WARNING_CODES.items() if warning_bits & (1 << bit)]


def decode_ge(red: bytes) -> tuple[int, int]:
    """Decode a get-warning/error (GE) RED payload into (warning_bits, error_bits).

    Layout: RED-1 warning code (2 bytes), ',' separator, RED-2 error code (2 bytes).
    """
    if len(red) < 5:
        raise LightmeterError(f'truncated GE response from CL-70F ({len(red)} bytes)')
    warning = struct.unpack(f'{BYTE_ORDER}H', red[0:2])[0]
    error = struct.unpack(f'{BYTE_ORDER}H', red[3:5])[0]
    return warning, error


# --- NR (get latest measurement result) RED layout ---------------------------
# An ordered (name, size) table of the RED fields, with a 1-byte ',' separator between
# consecutive fields. Offsets are derived by walking this table so the byte positions
# can never drift out of step with the field sizes. Reserved fields are kept (named
# ext*) only to advance the offset correctly.
_NR_FIELDS: tuple[tuple[str, int], ...] = (
    ('viewing_angle', 1),
    ('memory_title', 16),
    ('consecutive_number', 4),
    ('ext4', 1),
    ('spectrum_yaxis', 2),
    ('ext6', 1),
    ('ext7', 4),
    ('ext8', 1),
    ('ext9', 4),
    ('ext10', 1),
    ('colour_temperature', 4),
    ('deviation', 4),
    ('ext13', 4),
    ('ext14', 4),
    ('ext15', 48),
    ('ext16', 48),
    ('ext17', 4),
    ('ext18', 48),
    ('ext19', 48),
    ('illuminance_lx', 4),
    ('illuminance_fc', 4),
    ('tristimulus_x', 8),
    ('tristimulus_y', 8),
    ('tristimulus_z', 8),
    ('cie1931_x', 4),
    ('cie1931_y', 4),
    ('cie1960_u', 4),
    ('cie1960_v', 4),
    ('cie1976_u', 4),
    ('cie1976_v', 4),
    ('dominant_wavelength', 4),
    ('excitation_purity', 4),
    ('ra', 4),
    *((f'r{i}', 4) for i in range(1, 16)),  # R1…R15
    ('spectrum_5nm', 81 * 4),  # 380–780 nm, 5 nm pitch
    ('spectrum_1nm', 401 * 4),  # 380–780 nm, 1 nm pitch
    ('ext51', 1),
    ('ext52', 1),
    ('ext53', 1),
    ('ext54', 1),
    ('ext55', 4),
    ('ext56', 4),
)


def _build_offsets(fields: tuple[tuple[str, int], ...]) -> dict[str, tuple[int, int]]:
    """Walk the field table into {name: (offset, size)}, accounting for 1-byte separators."""
    offsets: dict[str, tuple[int, int]] = {}
    pos = 0
    for index, (name, size) in enumerate(fields):
        offsets[name] = (pos, size)
        pos += size
        if index < len(fields) - 1:
            pos += 1  # ',' separator
    return offsets


_NR_OFFSETS = _build_offsets(_NR_FIELDS)
# Total RED length implied by the layout (fields plus the separators between them).
NR_RED_LENGTH = sum(size for _, size in _NR_FIELDS) + (len(_NR_FIELDS) - 1)


def _float(red: bytes, name: str) -> float | None:
    """Decode a 4-byte single- or 8-byte double-precision field by name, or None if absent."""
    offset, size = _NR_OFFSETS[name]
    if offset + size > len(red):
        return None
    fmt = f'{BYTE_ORDER}{"f" if size == 4 else "d"}'
    return struct.unpack(fmt, red[offset : offset + size])[0]


def _spectrum(red: bytes, name: str, start_nm: float, step_nm: float, count: int) -> Spectrum | None:
    offset, size = _NR_OFFSETS[name]
    if offset + size > len(red):
        return None
    values = struct.unpack(f'{BYTE_ORDER}{count}f', red[offset : offset + size])
    return Spectrum(start_nm=start_nm, step_nm=step_nm, values=values)


def _pair(red: bytes, a: str, b: str) -> tuple[float, float] | None:
    va, vb = _float(red, a), _float(red, b)
    return (va, vb) if va is not None and vb is not None else None


def decode_nr(red: bytes) -> Measurement:
    """Decode an NR (get latest measurement result) RED payload into a Measurement.

    Fields are read positionally from the fixed layout — never by splitting on the
    ',' separator, since a binary float can itself contain a 0x2C byte. Trailing
    fields absent from a short payload decode to None.
    """
    lux = _float(red, 'illuminance_lx')
    cct = _float(red, 'colour_temperature')
    if lux is None or cct is None:
        raise LightmeterError(f'CL-70F measurement result too short to contain lux/CCT ({len(red)} bytes)')

    cri_ri = tuple(_float(red, f'r{i}') for i in range(1, 16))
    xyz = tuple(_float(red, f'tristimulus_{c}') for c in 'xyz')
    return Measurement(
        illuminance_lux=lux,
        cct=cct,
        duv=_float(red, 'deviation'),
        cie1931_xy=_pair(red, 'cie1931_x', 'cie1931_y'),
        cie1976_uv=_pair(red, 'cie1976_u', 'cie1976_v'),
        cie1960_uv=_pair(red, 'cie1960_u', 'cie1960_v'),
        illuminance_fc=_float(red, 'illuminance_fc'),
        tristimulus_xyz=xyz if all(v is not None for v in xyz) else None,
        dominant_wavelength=_float(red, 'dominant_wavelength'),
        excitation_purity=_float(red, 'excitation_purity'),
        cri_ra=_float(red, 'ra'),
        cri_ri=cri_ri if all(v is not None for v in cri_ri) else None,
        spectrum_5nm=_spectrum(red, 'spectrum_5nm', 380.0, 5.0, 81),
        spectrum_1nm=_spectrum(red, 'spectrum_1nm', 380.0, 1.0, 401),
    )
