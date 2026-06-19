# SPDX-License-Identifier: BSD-2-Clause
#
# Copyright (C) 2026, Raspberry Pi
#
# Run the CTT in-process and stream its progress.
#
# We call ctt.core.runner.run_ctt_targets directly (the same shared pipeline the
# CLI uses) rather than spawning `python -m ctt`. CTT routes all progress through the
# `logging` module, so we capture it by attaching a handler to the `ctt` logger
# and relaying each line to the browser over SSE. The calibration runs in a
# worker thread so the SSE generator can stream lines as they are produced.

import json
import logging
import os
import queue
import threading
from collections.abc import Iterator
from pathlib import Path

from .sessions import Project

VALID_TARGETS = ('pisp', 'vc4')
VALID_MODES = ('full', 'alsc-only', 'colour-only', 'blacklevel-only')

# Attaching a handler to the process-global `ctt` logger is not safe to do
# concurrently, and CTT is not designed for concurrent runs, so serialise.
_run_lock = threading.Lock()
_SENTINEL = object()


_CCM_MATRIX_SELECTIONS = ('average', 'maximum', 'patches')


def build_config(options: dict) -> dict:
    """Build a CTT config dict from UI options (schema mirrors config_example.json)."""
    alsc = options.get('alsc', {})
    awb = options.get('awb', {})
    macbeth = options.get('macbeth', {})
    ccm = options.get('ccm', {})
    lux = options.get('lux', {})
    matrix_selection = ccm.get('matrix_selection', 'average')
    if matrix_selection not in _CCM_MATRIX_SELECTIONS:
        matrix_selection = 'average'
    ccm_config: dict = {'matrix_selection': matrix_selection}
    # test_patches only matters for the 'patches' selection; pass it through when given.
    test_patches = ccm.get('test_patches')
    if test_patches:
        ccm_config['test_patches'] = [int(p) for p in test_patches]
    return {
        'disable': list(options.get('disable', [])),
        'plot': [],  # interactive matplotlib plots are not used in the web flow
        'alsc': {
            'do_alsc_colour': int(bool(alsc.get('do_alsc_colour', 1))),
            'luminance_strength': float(alsc.get('luminance_strength', 0.8)),
            'max_gain': float(alsc.get('max_gain', 8.0)),
        },
        'awb': {'greyworld': int(bool(awb.get('greyworld', 0)))},
        'blacklevel': int(options.get('blacklevel', -1)),
        'macbeth': {
            'small': int(bool(macbeth.get('small', 0))),
            'show': 0,  # never pop interactive windows from the server
        },
        'ccm': ccm_config,
        'lux': {
            'reference_target': int(lux.get('reference_target', 1000)),
            'reference_method': lux.get('reference_method', 'trimmed-mean'),
        },
    }


def write_config(project: Project, options: dict) -> Path:
    project.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = project.output_dir / 'config.json'
    config_path.write_text(json.dumps(build_config(options), indent=4))
    return config_path


class _QueueLogHandler(logging.Handler):
    """Logging handler that relays each (newline-split) message to a queue."""

    def __init__(self, q: queue.Queue) -> None:
        super().__init__(level=logging.INFO)
        self._q = q
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record: logging.LogRecord) -> None:
        for line in self.format(record).split('\n'):
            self._q.put(line)


def run_ctt_stream(
    project: Project,
    targets: list[str],
    mode: str = 'full',
    options: dict | None = None,
    update: bool = False,
) -> Iterator[str]:
    """Run CTT in-process and yield progress lines as they are produced.

    Yields each progress line (no trailing newline), then a final
    'CTT_EXIT <code>' line. <code> is non-zero if the run failed.

    When ``update`` is set, each target's existing tuning file
    ({name}_{target}.json) is updated in place rather than regenerated from the
    default template: the run re-calibrates only the sections selected by
    ``mode`` and preserves the rest. Targets with no existing tuning are skipped.
    """
    targets = [t for t in targets if t in VALID_TARGETS]
    if not targets:
        yield 'ERROR: no valid target selected (choose pisp and/or vc4)'
        yield 'CTT_EXIT 2'
        return
    if mode not in VALID_MODES:
        yield f'ERROR: invalid mode {mode!r}'
        yield 'CTT_EXIT 2'
        return

    if not _run_lock.acquire(blocking=False):
        yield 'ERROR: a calibration is already running; wait for it to finish'
        yield 'CTT_EXIT 2'
        return

    try:
        config_path = write_config(project, options or {})
        alsc_only = mode == 'alsc-only'
        colour_only = mode == 'colour-only'
        black_level_only = mode == 'blacklevel-only'
        # Honour per-capture exclusion: pass the include-list only when something
        # is excluded, so the default run still picks up stray on-disk DNGs that
        # aren't registered in project.json (exactly as before).
        has_excluded = any(c.excluded for c in project.captures)
        images = [c.filename for c in project.captures if not c.excluded] if has_excluded else None

        # In update mode each target is updated in place from its existing tuning
        # file; targets without one are skipped. A normal run regenerates all
        # selected targets from the default template.
        update_paths: dict[str, str] = {}
        run_targets = list(targets)
        skipped: list[str] = []
        if update:
            run_targets = []
            for t in targets:
                existing = project.output_dir / f'{project.name}_{t}.json'
                if existing.exists():
                    update_paths[t] = str(existing)
                    run_targets.append(t)
                else:
                    skipped.append(f'No existing {t} tuning to update — skipped')
            if not run_targets:
                for line in skipped:
                    yield line
                yield 'ERROR: no existing tuning to update for the selected target(s)'
                yield 'CTT_EXIT 2'
                return

        q: queue.Queue = queue.Queue()
        result: dict[str, int] = {}

        def worker() -> None:
            # Set a headless matplotlib backend before importing ctt (its
            # algorithm modules import matplotlib at import time).
            os.environ.setdefault('MPLBACKEND', 'Agg')
            from ctt.core.runner import run_ctt_targets
            from ctt.utils.errors import ArgError

            ctt_logger = logging.getLogger('ctt')
            saved = (ctt_logger.handlers[:], ctt_logger.level, ctt_logger.propagate)
            handler = _QueueLogHandler(q)
            ctt_logger.handlers = [handler]
            ctt_logger.setLevel(logging.INFO)
            ctt_logger.propagate = False
            code = 0
            try:
                if update:
                    # run_ctt_targets writes a single update_path, so update each
                    # target separately with its own existing tuning file.
                    for t in run_targets:
                        run_ctt_targets(
                            project.output_dir,
                            project.name,
                            str(project.path),
                            str(config_path),
                            [t],
                            alsc_only=alsc_only,
                            colour_only=colour_only,
                            black_level_only=black_level_only,
                            update_path=update_paths[t],
                            images=images,
                        )
                else:
                    run_ctt_targets(
                        project.output_dir,
                        project.name,
                        str(project.path),
                        str(config_path),
                        run_targets,
                        alsc_only=alsc_only,
                        colour_only=colour_only,
                        black_level_only=black_level_only,
                        images=images,
                    )
            except ArgError as err:
                code = 1
                for line in str(err).splitlines():
                    if line.strip():
                        q.put(line.strip())
            except Exception as err:
                code = 1
                q.put(f'ERROR: {type(err).__name__}: {err}')
            finally:
                ctt_logger.handlers, ctt_logger.level, ctt_logger.propagate = saved
                result['code'] = code
                q.put(_SENTINEL)

        header = f'$ ctt (in-process)  targets={",".join(run_targets)}  mode={mode}'
        if update:
            header += '  update=on'
        if images is not None:
            header += f'  excluded={sum(1 for c in project.captures if c.excluded)}'
        yield header
        for line in skipped:
            yield line
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is _SENTINEL:
                break
            yield item
        thread.join()
        # A successful run regenerates the originals, so hand edits no longer
        # apply: discard each run target's custom tuning file.
        if result.get('code', 1) == 0:
            for t in run_targets:
                custom = project.output_dir / f'{project.name}_{t}_custom.json'
                if custom.exists():
                    custom.unlink()
                    yield f'Discarded custom tuning edits for {t} (original regenerated)'
        yield f'CTT_EXIT {result.get("code", 1)}'
    finally:
        _run_lock.release()


def output_files(project: Project, targets: list[str]) -> dict[str, dict[str, str | float | None]]:
    """Return produced output paths + mtime keyed by target: {target: {json, log, mtime}}."""
    out: dict[str, dict[str, str | float | None]] = {}
    for target in targets:
        json_path = project.output_dir / f'{project.name}_{target}.json'
        log_path = project.output_dir / f'{project.name}_{target}.log'
        out[target] = {
            'json': str(json_path) if json_path.exists() else None,
            'log': str(log_path) if log_path.exists() else None,
            'mtime': json_path.stat().st_mtime if json_path.exists() else None,
        }
    return out
