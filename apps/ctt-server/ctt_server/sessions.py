# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Project (capture session) model.
#
# A project is a directory under the workspace root that holds the captured
# .dng files (in its root, exactly where `ctt -i <dir>` expects them) plus a
# project.json sidecar recording per-capture metadata. CTT only reads .dng
# files from the directory root and ignores everything else, so project.json
# and the output/ subdirectory are invisible to it.

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ctt.core.camera import burst_group_key

from . import naming

_SIDECAR = 'project.json'
_OUTPUT_DIRNAME = 'output'


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec='seconds')


def default_workspace() -> Path:
    """Workspace root holding all projects. Override with $CTT_CAPTURE_WORKSPACE."""
    env = os.environ.get('CTT_CAPTURE_WORKSPACE')
    root = Path(env) if env else Path.home() / 'ctt-server-workspace'
    return root


def _safe_project_name(name: str) -> str:
    clean = re.sub(r'[^A-Za-z0-9._-]', '_', name.strip())
    clean = clean.strip('._-')
    if not clean:
        raise ValueError(f'Invalid project name: {name!r}')
    return clean


@dataclass
class Capture:
    """One captured image and its tags."""

    filename: str
    image_type: str
    colour_temp: int | None = None  # dark frames have no colour temperature
    lux: int | None = None
    label: str | None = None
    controls: dict = field(default_factory=dict)
    captured_at: str = field(default_factory=_now_iso)
    excluded: bool = False  # excluded from CTT runs; the file stays on disk


class Project:
    """A capture session backed by a directory of .dng files + project.json."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.captures: list[Capture] = []
        self.created_at = _now_iso()
        self.notes = ''
        if self.sidecar.exists():
            self._load()

    # --- paths -------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.path.name

    @property
    def sidecar(self) -> Path:
        return self.path / _SIDECAR

    @property
    def output_dir(self) -> Path:
        return self.path / _OUTPUT_DIRNAME

    # --- persistence -------------------------------------------------------
    def _load(self) -> None:
        data = json.loads(self.sidecar.read_text())
        self.created_at = data.get('created_at', self.created_at)
        self.notes = data.get('notes', '')
        # Heal any duplicate filenames from older saves (a re-capture at the same
        # colour temp + lux overwrites the DNG, so only one entry is meaningful).
        # Keep the original position but the most recent metadata (last wins).
        by_name: dict[str, Capture] = {}
        for c in (Capture(**c) for c in data.get('captures', [])):
            by_name[c.filename] = c
        self.captures = list(by_name.values())

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        data = {
            'name': self.name,
            'created_at': self.created_at,
            'notes': self.notes,
            'captures': [asdict(c) for c in self.captures],
        }
        self.sidecar.write_text(json.dumps(data, indent=2))

    # --- captures ----------------------------------------------------------
    def _dng_names(self) -> list[str]:
        return [p.name for p in self.path.glob('*.dng')]

    def add_capture(
        self,
        dng_bytes: bytes,
        image_type: str,
        colour_temp: int | None = None,
        *,
        lux: int | None = None,
        label: str | None = None,
        controls: dict | None = None,
        jpeg_bytes: bytes | None = None,
        indexed: bool = False,
    ) -> Capture:
        """Write a DNG with a CTT-correct filename and record its metadata.

        If jpeg_bytes is given, a full-resolution JPEG is written alongside the DNG
        under the same stem (e.g. foo.dng -> foo.jpg) for in-browser preview.
        indexed=True gives Macbeth captures a _<n> suffix (burst frames; CTT
        averages same-name groups internally); ALSC/CAC/dark are always indexed.
        """
        # Macbeth filenames are prefixed with the project (sensor) name by default.
        if image_type == 'macbeth' and not label:
            label = self.name
        if image_type == 'macbeth' and not indexed:
            index = None  # index-free name: capturing again overwrites
        else:
            index = naming.next_index(self._dng_names(), image_type, colour_temp, lux=lux, label=label)
        filename = naming.build_filename(image_type, colour_temp, lux=lux, label=label, index=index)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / filename).write_bytes(dng_bytes)
        jpg_path = (self.path / filename).with_suffix('.jpg')
        if jpeg_bytes is not None:
            jpg_path.write_bytes(jpeg_bytes)
        else:
            jpg_path.unlink(missing_ok=True)  # a re-capture without a JPEG must not leave a stale one
        capture = Capture(
            filename=filename,
            image_type=image_type,
            colour_temp=colour_temp,
            lux=lux,
            label=label,
            controls=controls or {},
        )
        # Re-capturing the same filename overwrites the DNG on disk (above), so
        # replace the existing metadata entry in place rather than duplicating it.
        existing = next((i for i, c in enumerate(self.captures) if c.filename == filename), None)
        if existing is not None:
            self.captures[existing] = capture
        else:
            self.captures.append(capture)
        self.save()
        return capture

    def import_capture(self, filename: str, data: bytes) -> Capture:
        """Register an uploaded image, renaming it to the project's naming scheme.

        Tags are parsed from the uploaded filename (which must follow the CTT
        convention; naming.NamingError otherwise), then the file is stored
        through add_capture so uploads are named exactly like captures —
        Macbeth labels become the project name, ALSC/CAC/dark get their plain
        canonical names — and the Images-tab burst grouping applies uniformly.
        """
        # Strip any path and normalise a .dng extension to lowercase so names
        # like FOO_5000K_800L.DNG still parse via CTT's (lowercase) extension regex.
        name = Path(filename).name
        stem, dot, ext = name.rpartition('.')
        if dot and ext.lower() == 'dng':
            name = f'{stem}.dng'

        image_type, colour_temp, lux, _label = naming.parse_filename(name)
        # A trailing _<n> on a Macbeth name marks a burst frame: keep it indexed
        # so multi-frame uploads get distinct names; plain names keep the
        # index-free overwrite semantics, exactly as captures do.
        indexed = image_type == 'macbeth' and burst_group_key(name) != name
        return self.add_capture(data, image_type, colour_temp, lux=lux, indexed=indexed)

    def delete_capture(self, filename: str) -> None:
        target = self.path / filename
        if target.exists() and target.suffix == '.dng':
            target.unlink()
            target.with_suffix('.jpg').unlink(missing_ok=True)  # remove the sibling preview JPEG
        self.captures = [c for c in self.captures if c.filename != filename]
        self.save()

    def set_excluded(self, filename: str, excluded: bool) -> Capture:
        """Mark a capture as excluded from (or included in) CTT runs; persists."""
        for c in self.captures:
            if c.filename == filename:
                c.excluded = excluded
                self.save()
                return c
        raise KeyError(filename)

    def has_saved_jpeg(self, filename: str) -> bool:
        """True if a sibling preview JPEG exists for this capture's DNG."""
        return filename.endswith('.dng') and (self.path / filename).with_suffix('.jpg').exists()

    def counts(self) -> dict[str, int]:
        out = {'macbeth': 0, 'alsc': 0, 'cac': 0, 'dark': 0}
        for c in self.captures:
            out[c.image_type] = out.get(c.image_type, 0) + 1
        return out


class Workspace:
    """Container for projects under a workspace root directory."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else default_workspace()
        self.root.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> list[Project]:
        projects = []
        for child in sorted(self.root.iterdir()):
            if child.is_dir() and (child / _SIDECAR).exists():
                projects.append(Project(child))
        return projects

    def get_project(self, name: str) -> Project:
        path = self.root / _safe_project_name(name)
        if not (path / _SIDECAR).exists():
            raise FileNotFoundError(f'No such project: {name!r}')
        return Project(path)

    def create_project(self, name: str) -> Project:
        path = self.root / _safe_project_name(name)
        if (path / _SIDECAR).exists():
            raise FileExistsError(f'Project already exists: {name!r}')
        project = Project(path)
        project.save()
        return project

    def delete_project(self, name: str) -> None:
        path = self.root / _safe_project_name(name)
        if path.exists():
            shutil.rmtree(path)
