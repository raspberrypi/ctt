# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Build and validate CTT calibration-image filenames.
#
# CTT encodes the calibration tags in the filename itself (parsed by
# ctt.core.camera.get_col_lux and the image-type checks in Camera.add_imgs):
#
#   ALSC flat-field:  alsc_<K>k_<idx>.dng        e.g. alsc_5000k_0.dng
#   Macbeth chart:    <label>_<K>k_<lux>l.dng    e.g. d65_5858k_1344l.dng
#   Macbeth burst:    <label>_<K>k_<lux>l_<idx>.dng (frames CTT averages internally)
#   CAC dot chart:    cac_<K>k_<idx>.dng          e.g. cac_5000k_0.dng
#   Dark frame:       dark_<idx>.dng              e.g. dark_0.dng (no tags needed)
#
# We reuse get_col_lux as the single source of truth: every name we build is
# validated by round-tripping it back through the exact regex CTT uses.

import re

from ctt.core.camera import get_col_lux

IMAGE_TYPES = ('macbeth', 'alsc', 'cac', 'dark')

# Image type is detected by substring in CTT (Camera.add_imgs): a filename
# containing 'alsc' is ALSC, 'cac' is CAC, 'dark' is a dark frame, otherwise
# it is treated as Macbeth. A Macbeth label must contain none of these.
_RESERVED_SUBSTRINGS = ('alsc', 'cac', 'dark')


class NamingError(ValueError):
    """Raised when the requested tags cannot produce a valid CTT filename."""


def sanitise_label(label: str) -> str:
    """Reduce an illuminant label to lowercase alphanumerics (e.g. 'D65' -> 'd65')."""
    cleaned = re.sub(r'[^a-z0-9]', '', label.strip().lower())
    return cleaned


def build_filename(
    image_type: str,
    colour_temp: int | None = None,
    *,
    lux: int | None = None,
    label: str | None = None,
    index: int | None = None,
) -> str:
    """Build a CTT-compatible .dng filename for the given image type and tags.

    Raises NamingError if the tags are inconsistent (e.g. Macbeth without lux,
    or a label that collides with the alsc/cac/dark type markers), or if the
    result does not round-trip through CTT's own get_col_lux parser.
    """
    if image_type not in IMAGE_TYPES:
        raise NamingError(f'Unknown image type: {image_type!r} (expected one of {IMAGE_TYPES})')
    if image_type == 'dark':
        # Dark frames carry no tags: the capture conditions (no light) are not
        # encoded, and CTT reads exposure/gain from the DNG metadata.
        name = f'dark_{index or 0}.dng'
        _verify(name, image_type, None, None)
        return name
    if not isinstance(colour_temp, int) or colour_temp <= 0:
        raise NamingError(f'Colour temperature must be a positive integer Kelvin, got {colour_temp!r}')

    if image_type == 'alsc':
        name = f'alsc_{colour_temp}k_{index or 0}.dng'
    elif image_type == 'cac':
        name = f'cac_{colour_temp}k_{index or 0}.dng'
    else:  # macbeth
        if lux is None or lux <= 0:
            raise NamingError('Macbeth images require a positive lux value')
        clean = sanitise_label(label or 'mac')
        if not clean:
            raise NamingError(f'Macbeth label {label!r} is empty after sanitising')
        if any(s in clean for s in _RESERVED_SUBSTRINGS):
            raise NamingError(
                f'Macbeth label {clean!r} must not contain {_RESERVED_SUBSTRINGS} '
                '(it would be misdetected as an ALSC/CAC image)'
            )
        # Burst frames get an index suffix (CTT averages same-name groups
        # internally); single captures keep the index-free name, preserving the
        # capture-again-to-overwrite behaviour.
        suffix = f'_{index}' if index is not None else ''
        name = f'{clean}_{colour_temp}k_{lux}l{suffix}.dng'

    _verify(name, image_type, colour_temp, lux)
    return name


def _verify(name: str, image_type: str, colour_temp: int | None, lux: int | None) -> None:
    """Assert the generated name parses back to the intended tags via CTT's parser."""
    col, parsed_lux = get_col_lux(name)
    if image_type != 'dark' and col != colour_temp:
        raise NamingError(f'Generated name {name!r} does not encode colour temp {colour_temp} (got {col})')
    if image_type == 'macbeth' and parsed_lux != lux:
        raise NamingError(f'Generated name {name!r} does not encode lux {lux} (got {parsed_lux})')
    detected = detect_type(name)
    if detected != image_type:
        raise NamingError(f'Generated name {name!r} is detected as {detected!r}, expected {image_type!r}')


def detect_type(filename: str) -> str:
    """Mirror CTT's image-type detection (Camera.add_imgs) for a filename."""
    if 'alsc' in filename:
        return 'alsc'
    if 'cac' in filename:
        return 'cac'
    if 'dark' in filename:
        return 'dark'
    return 'macbeth'


def validate_filename(filename: str) -> tuple[bool, str]:
    """Validate an existing .dng filename. Returns (ok, message).

    ok=False means CTT would skip the file (missing/invalid tags for its type).
    """
    if not filename.lower().endswith('.dng'):
        return False, 'Not a .dng file'
    image_type = detect_type(filename)
    if image_type == 'dark':
        return True, 'OK'  # dark frames need no tags
    col, lux = get_col_lux(filename)
    if col is None:
        return False, 'No colour temperature in filename (expected e.g. 5000k)'
    if image_type == 'macbeth' and lux is None:
        return False, 'Macbeth image missing lux in filename (expected e.g. 5000k_800l)'
    return True, 'OK'


def parse_filename(filename: str) -> tuple[str, int | None, int | None, str | None]:
    """Parse a CTT-format filename into (image_type, colour_temp, lux, label).

    Used to auto-tag uploaded images from their filename. Raises NamingError if
    the name is not a valid CTT calibration filename. Dark frames have no tags:
    colour_temp, lux and label are all None.
    """
    ok, msg = validate_filename(filename)
    if not ok:
        raise NamingError(msg)
    image_type = detect_type(filename)
    if image_type == 'dark':
        return 'dark', None, None, None
    colour_temp, lux = get_col_lux(filename)
    label = None
    if image_type == 'macbeth':
        # The label is the prefix before the `_<K>k` tag, e.g. 'd65' in d65_5000k_800l.dng.
        label = filename.split(f'_{colour_temp}k', 1)[0] or None
    return image_type, colour_temp, lux, label


def next_index(
    existing: list[str],
    image_type: str,
    colour_temp: int | None = None,
    *,
    lux: int | None = None,
    label: str | None = None,
) -> int:
    """Return the next free replicate index for a name with these tags.

    ALSC/CAC/dark names are always indexed; Macbeth names only in burst mode,
    where the prefix includes the label and lux (e.g. d65_5000k_800l_).
    """
    if image_type == 'macbeth':
        prefix = f'{sanitise_label(label or "mac")}_{colour_temp}k_{lux}l_'
    elif image_type == 'dark':
        prefix = 'dark_'
    else:
        prefix = f'{image_type}_{colour_temp}k_'
    used = []
    for name in existing:
        if name.startswith(prefix) and name.endswith('.dng'):
            stem = name[len(prefix) : -len('.dng')]
            if stem.isdigit():
                used.append(int(stem))
    return (max(used) + 1) if used else 0
