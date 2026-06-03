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
#   CAC dot chart:    cac_<K>k_<idx>.dng          e.g. cac_5000k_0.dng
#
# We reuse get_col_lux as the single source of truth: every name we build is
# validated by round-tripping it back through the exact regex CTT uses.

import re

from ctt.core.camera import get_col_lux

IMAGE_TYPES = ('macbeth', 'alsc', 'cac')

# Image type is detected by substring in CTT (Camera.add_imgs): a filename
# containing 'alsc' is ALSC, 'cac' is CAC, otherwise it is treated as Macbeth.
# A Macbeth label must therefore contain neither substring.
_RESERVED_SUBSTRINGS = ('alsc', 'cac')


class NamingError(ValueError):
    """Raised when the requested tags cannot produce a valid CTT filename."""


def sanitise_label(label: str) -> str:
    """Reduce an illuminant label to lowercase alphanumerics (e.g. 'D65' -> 'd65')."""
    cleaned = re.sub(r'[^a-z0-9]', '', label.strip().lower())
    return cleaned


def build_filename(
    image_type: str,
    colour_temp: int,
    *,
    lux: int | None = None,
    label: str | None = None,
    index: int = 0,
) -> str:
    """Build a CTT-compatible .dng filename for the given image type and tags.

    Raises NamingError if the tags are inconsistent (e.g. Macbeth without lux,
    or a label that collides with the alsc/cac type markers), or if the result
    does not round-trip through CTT's own get_col_lux parser.
    """
    if image_type not in IMAGE_TYPES:
        raise NamingError(f'Unknown image type: {image_type!r} (expected one of {IMAGE_TYPES})')
    if not isinstance(colour_temp, int) or colour_temp <= 0:
        raise NamingError(f'Colour temperature must be a positive integer Kelvin, got {colour_temp!r}')

    if image_type == 'alsc':
        name = f'alsc_{colour_temp}k_{index}.dng'
    elif image_type == 'cac':
        name = f'cac_{colour_temp}k_{index}.dng'
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
        name = f'{clean}_{colour_temp}k_{lux}l.dng'

    _verify(name, image_type, colour_temp, lux)
    return name


def _verify(name: str, image_type: str, colour_temp: int, lux: int | None) -> None:
    """Assert the generated name parses back to the intended tags via CTT's parser."""
    col, parsed_lux = get_col_lux(name)
    if col != colour_temp:
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
    return 'macbeth'


def validate_filename(filename: str) -> tuple[bool, str]:
    """Validate an existing .dng filename. Returns (ok, message).

    ok=False means CTT would skip the file (missing/invalid tags for its type).
    """
    if not filename.lower().endswith('.dng'):
        return False, 'Not a .dng file'
    image_type = detect_type(filename)
    col, lux = get_col_lux(filename)
    if col is None:
        return False, 'No colour temperature in filename (expected e.g. 5000k)'
    if image_type == 'macbeth' and lux is None:
        return False, 'Macbeth image missing lux in filename (expected e.g. 5000k_800l)'
    return True, 'OK'


def parse_filename(filename: str) -> tuple[str, int, int | None, str | None]:
    """Parse a CTT-format filename into (image_type, colour_temp, lux, label).

    Used to auto-tag uploaded images from their filename. Raises NamingError if
    the name is not a valid CTT calibration filename.
    """
    ok, msg = validate_filename(filename)
    if not ok:
        raise NamingError(msg)
    image_type = detect_type(filename)
    colour_temp, lux = get_col_lux(filename)
    label = None
    if image_type == 'macbeth':
        # The label is the prefix before the `_<K>k` tag, e.g. 'd65' in d65_5000k_800l.dng.
        label = filename.split(f'_{colour_temp}k', 1)[0] or None
    return image_type, colour_temp, lux, label


def next_index(existing: list[str], image_type: str, colour_temp: int) -> int:
    """Return the next free replicate index for an ALSC/CAC name at this colour temp."""
    prefix = f'{image_type}_{colour_temp}k_'
    used = []
    for name in existing:
        if name.startswith(prefix) and name.endswith('.dng'):
            stem = name[len(prefix) : -len('.dng')]
            if stem.isdigit():
                used.append(int(stem))
    return (max(used) + 1) if used else 0
