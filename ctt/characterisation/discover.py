# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Discover and group a project's existing DNGs into characterisation inputs.
#
# Grouping is filename-first (a burst shares a stem and differs only in the
# trailing _<n> index — the capture naming convention), then EXIF-verified:
# frames of a group must agree exactly on exposure, ISO, geometry, container
# bits and black level. Disagreeing frames split into sub-groups keyed by that
# exact tuple; there is deliberately no tolerance clustering — ISO 100 and 107
# are different operating points, and the PTC slope depends on gain.

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import exifread

from ..core.camera import get_col_lux
from .frames import WHITE_16

logger = logging.getLogger(__name__)

_INDEX_RE = re.compile(r'_\d+(?=\.dng$)', re.IGNORECASE)


@dataclass
class CaptureGroup:
    """A burst of frames at one verified operating point."""

    label: str
    kind: str  # 'dark' | 'flat' | 'chart'
    paths: list[Path]
    exposure_us: int
    gain: float
    width: int
    height: int
    sigbits: int
    blacklevel: int  # as stored in the DNG (container-domain DN)
    colour_temp: int | None = None
    lux: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def blacklevel_16(self) -> float:
        return float(self.blacklevel << (16 - self.sigbits))


def group_key(name: str) -> str:
    """Burst stem: the filename with a trailing _<n> index removed.

    Generalises ctt.core.camera.burst_group_key (which only matches Macbeth
    names ending in a lux tag) to the dark_<n> and alsc_<K>k_<n> conventions.
    """
    return _INDEX_RE.sub('', name)


def classify(name: str) -> str | None:
    """The capture kind, using the same substring rules as CTT's image intake."""
    lower = name.lower()
    if 'dark' in lower:
        return 'dark'
    if 'alsc' in lower:
        return 'flat'
    if 'cac' in lower:
        return None  # dot charts: no characterisation use
    return 'chart'


def _read_exif(path: Path) -> dict:
    """The operating-point EXIF of one DNG, tolerant of writer layout differences.

    details=True is required: some writers (observed on PiDNG ALSC captures)
    hide BlackLevel from the fast details=False parse.
    """
    with open(path, 'rb') as f:
        tags = exifread.process_file(f, details=True)

    def tag(name):
        for prefix in ('EXIF SubIFD0', 'Image', 'EXIF'):
            value = tags.get(f'{prefix} {name}')
            if value is not None:
                return value
        raise KeyError(name)

    exp = tag('ExposureTime').values[0]
    try:
        white = int(tags['EXIF SubIFD0 Tag 0xC61D'].values[0])
    except KeyError:
        white = WHITE_16  # mono DNGs may omit WhiteLevel
    blacks = tag('BlackLevel').values
    try:
        black = int(blacks[0])
    except TypeError:
        black = int(blacks)
    return {
        'exposure_us': int(exp.num / exp.den * 1_000_000),
        'iso': int(tag('ISOSpeedRatings').values[0]),
        'width': int(tag('ImageWidth').values[0]),
        'height': int(tag('ImageLength').values[0]),
        'sigbits': white.bit_length(),
        'blacklevel': black,
    }


def scan_project(directory: Path | str, excluded: frozenset[str] | set[str] = frozenset()) -> list[CaptureGroup]:
    """Group a project directory's DNGs into verified operating-point bursts.

    Only the directory root is scanned (where CTT's own captures live).
    Excluded filenames are dropped before grouping; files with unreadable or
    incomplete EXIF are skipped with a warning attached to their group. Name
    groups whose frames disagree on the operating point split into sub-groups
    labelled `<stem>#2`, `#3`, ... in scan order.
    """
    directory = Path(directory)
    by_stem: dict[str, list[Path]] = {}
    for path in sorted(directory.glob('*.dng')):
        if path.name in excluded:
            continue
        if classify(path.name) is None:
            continue
        by_stem.setdefault(group_key(path.name), []).append(path)

    groups: list[CaptureGroup] = []
    for stem, paths in by_stem.items():
        kind = classify(stem)
        colour_temp, lux = get_col_lux(paths[0].name)
        by_point: dict[tuple, CaptureGroup] = {}
        warnings: list[str] = []
        for path in paths:
            try:
                meta = _read_exif(path)
            except Exception as err:
                warnings.append(f'{path.name}: unreadable metadata, skipped ({err.__class__.__name__}: {err})')
                continue
            key = tuple(meta.values())
            if key not in by_point:
                suffix = f'#{len(by_point) + 1}' if by_point else ''
                by_point[key] = CaptureGroup(
                    label=Path(stem).stem + suffix,
                    kind=kind,
                    paths=[],
                    exposure_us=meta['exposure_us'],
                    gain=meta['iso'] / 100.0,
                    width=meta['width'],
                    height=meta['height'],
                    sigbits=meta['sigbits'],
                    blacklevel=meta['blacklevel'],
                    colour_temp=colour_temp,
                    lux=lux,
                )
            by_point[key].paths.append(path)
        point_groups = list(by_point.values())
        if len(point_groups) > 1:
            summary = ', '.join(f'{g.label} ({len(g.paths)} frames)' for g in point_groups)
            warnings.append(f'{stem}: frames span {len(point_groups)} operating points — split into {summary}')
        for g in point_groups:
            g.warnings.extend(warnings)
        groups.extend(point_groups)
    return groups
