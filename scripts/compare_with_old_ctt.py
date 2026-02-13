#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-2-Clause
#
# Compare new ctt calibration output with the old libcamera ctt.
# Runs both tools on the same input directories and compares full JSON
# (structure + all parameters). Reports numerical errors; CCM may differ.
#
# Usage:
#   python3 scripts/compare_with_old_ctt.py [options]
#
# Requires: old ctt at --old-ctt-dir (run from repo root, or set OLD_CTT_DIR).

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _algorithms_by_key(data: dict) -> dict[str, dict]:
    """Return algorithms as a dict keyed by algorithm name (e.g. rpi.alsc)."""
    out: dict[str, dict] = {}
    for algo in data.get('algorithms', []):
        for name, blob in algo.items():
            out[name] = blob
    return out


def _compare_value(
    path: str,
    old_v: Any,
    new_v: Any,
    struct_errors: list[str],
    numeric_diffs: list[tuple[str, float, float, float]],
) -> None:
    """Recursively compare old_v and new_v; append structure errors and numeric diffs."""
    if type(old_v) is not type(new_v):
        if isinstance(old_v, (int, float)) and isinstance(new_v, (int, float)):
            pass  # compare as numbers
        else:
            struct_errors.append(f'{path}: type mismatch {type(old_v).__name__} vs {type(new_v).__name__}')
            return
    if isinstance(old_v, dict):
        all_keys = set(old_v) | set(new_v)
        for k in sorted(all_keys):
            if k not in old_v:
                struct_errors.append(f'{path}: key {k!r} only in new')
                continue
            if k not in new_v:
                struct_errors.append(f'{path}: key {k!r} only in old')
                continue
            _compare_value(f'{path}.{k}', old_v[k], new_v[k], struct_errors, numeric_diffs)
        return
    if isinstance(old_v, list):
        if len(old_v) != len(new_v):
            struct_errors.append(f'{path}: list length {len(old_v)} vs {len(new_v)}')
            return
        for i, (a, b) in enumerate(zip(old_v, new_v, strict=True)):
            _compare_value(f'{path}[{i}]', a, b, struct_errors, numeric_diffs)
        return
    if isinstance(old_v, (int, float)) and isinstance(new_v, (int, float)):
        try:
            diff = abs(float(old_v) - float(new_v))
            numeric_diffs.append((path, float(old_v), float(new_v), diff))
        except (TypeError, ValueError):
            struct_errors.append(f'{path}: non-numeric value')
        return
    if old_v != new_v:
        struct_errors.append(f'{path}: value mismatch {old_v!r} vs {new_v!r}')


def compare_json_full(old_data: dict, new_data: dict) -> tuple[list[str], list[tuple[str, float, float, float]]]:
    """
    Compare two tuning JSONs. Compare algorithms by name (order-independent).
    Returns (structure_errors, numeric_diffs) where numeric_diffs are (path, old, new, abs_diff).
    """
    struct_errors: list[str] = []
    numeric_diffs: list[tuple[str, float, float, float]] = []

    # Top-level keys
    for key in ('version', 'target'):
        if key not in old_data or key not in new_data:
            if key in old_data and key not in new_data:
                struct_errors.append(f'{key!r} only in old')
            elif key not in old_data and key in new_data:
                struct_errors.append(f'{key!r} only in new')
            continue
        if old_data[key] != new_data[key]:
            struct_errors.append(f'{key}: {old_data[key]!r} vs {new_data[key]!r}')

    old_algos = _algorithms_by_key(old_data)
    new_algos = _algorithms_by_key(new_data)
    all_names = sorted(set(old_algos) | set(new_algos))
    for name in all_names:
        if name not in old_algos:
            struct_errors.append(f'algorithms.{name}: only in new')
            continue
        if name not in new_algos:
            struct_errors.append(f'algorithms.{name}: only in old')
            continue
        _compare_value(f'algorithms.{name}', old_algos[name], new_algos[name], struct_errors, numeric_diffs)

    return struct_errors, numeric_diffs


def get_ccm(data: dict) -> list:
    for algo in data.get('algorithms', []):
        if 'rpi.ccm' in algo:
            return algo['rpi.ccm'].get('ccms', [])
    return []


def run_old_ctt(old_ctt_dir: Path, input_dir: Path, output_dir: Path, sensors: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for s in sensors:
        for t in ('pisp', 'vc4'):
            out = output_dir / f'{s}_{t}.json'
            cmd = [
                sys.executable,
                'ctt.py',
                '-i',
                str(input_dir / s),
                '-o',
                str(out),
                '-t',
                t,
            ]
            result = subprocess.run(cmd, cwd=old_ctt_dir, capture_output=True, text=True)
            if not out.exists():
                print('Old ctt did not write output (it only writes JSON when check_imgs passes).', file=sys.stderr)
                if result.stdout:
                    print(result.stdout, file=sys.stderr)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                raise SystemExit(f'Old ctt failed to produce {out}')
            if result.returncode != 0:
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                raise SystemExit(f'Old ctt failed with exit code {result.returncode}')


def run_new_ctt(repo_root: Path, input_dir: Path, output_dir: Path, sensors: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for s in sensors:
        cmd = [
            sys.executable,
            '-m',
            'ctt',
            '-i',
            str(input_dir / s),
            '-o',
            str(output_dir),
            '-t',
            'pisp',
            '-t',
            'vc4',
            '--name',
            s,
        ]
        result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr or result.stdout, file=sys.stderr)
            raise SystemExit(f'New ctt failed with exit code {result.returncode}')


def compare_ccms(old_dir: Path, new_dir: Path, sensors: list[str]) -> bool:
    print('\nCCM comparison: old (libcamera ctt) vs new (ctt)\n')
    print(f'{"Sensor":<12} {"Target":<6} {"Max |old-new|":<14} Status')
    print('-' * 50)
    all_ok = True
    for s in sensors:
        for t in ('pisp', 'vc4'):
            old_path = old_dir / f'{s}_{t}.json'
            new_path = new_dir / f'{s}_{t}.json'
            if not old_path.exists():
                print(f'{s:<12} {t:<6} {"":<14} Missing: {old_path}')
                all_ok = False
                continue
            if not new_path.exists():
                print(f'{s:<12} {t:<6} {"":<14} Missing: {new_path}')
                all_ok = False
                continue
            with open(old_path) as f:
                old_ccms = get_ccm(json.load(f))
            with open(new_path) as f:
                new_ccms = get_ccm(json.load(f))
            if len(old_ccms) != len(new_ccms):
                print(f'{s:<12} {t:<6} {"":<14} length mismatch')
                all_ok = False
                continue
            max_err = 0.0
            for oc, nc in zip(old_ccms, new_ccms, strict=True):
                if oc['ct'] != nc['ct']:
                    print(f'{s:<12} {t:<6} {"":<14} ct mismatch')
                    all_ok = False
                    break
                errs = [abs(a - b) for a, b in zip(oc['ccm'], nc['ccm'], strict=True)]
                max_err = max(max_err, max(errs))
            else:
                status = 'OK' if max_err < 0.01 else 'LARGE'
                if max_err >= 0.01:
                    all_ok = False
                print(f'{s:<12} {t:<6} {max_err:<14.6f} {status}')
    print(
        f'\n{"All CCM differences < 0.01 — comparison OK." if all_ok else "Some differences >= 0.01 — review above."}'
    )
    return all_ok


def compare_json_files(
    old_path: Path,
    new_path: Path,
) -> tuple[list[str], list[tuple[str, float, float, float]]]:
    """Load two JSON files and return (structure_errors, numeric_diffs)."""
    with open(old_path) as f:
        old_data = json.load(f)
    with open(new_path) as f:
        new_data = json.load(f)
    return compare_json_full(old_data, new_data)


def compare_all_params(old_dir: Path, new_dir: Path, sensors: list[str]) -> bool:
    """Compare entire JSON files for all sensors/targets. Report structure and numerical errors."""
    print('\nFull JSON comparison: old vs new (all parameters)\n')
    all_ok = True
    for s in sensors:
        for t in ('pisp', 'vc4'):
            old_path = old_dir / f'{s}_{t}.json'
            new_path = new_dir / f'{s}_{t}.json'
            if not old_path.exists() or not new_path.exists():
                continue
            struct_errors, numeric_diffs = compare_json_files(old_path, new_path)
            label = f'{s} / {t}'
            if struct_errors:
                print(f'--- {label} ---')
                print('Structure errors:')
                for e in struct_errors:
                    print(f'  {e}')
                all_ok = False
            if numeric_diffs:
                diffs_only = [d[3] for d in numeric_diffs]
                max_err = max(diffs_only)
                n = len(numeric_diffs)
                mean_err = sum(diffs_only) / n if n else 0.0
                status = 'OK' if max_err < 0.01 else 'LARGE'
                if max_err >= 0.01:
                    all_ok = False
                print(f'{label:<22} Numeric: n={n}, max |old-new|={max_err:.6f}, mean={mean_err:.6f}  {status}')
            elif not struct_errors:
                print(f'{label:<22} (no numeric fields to compare)')
    if all_ok:
        print('All parameters: no structure errors and max numeric error < 0.01.')
    else:
        print('Some structure errors or numeric errors >= 0.01 — review above.')
    return all_ok


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_input = repo_root / 'tuning_files'
    default_old_ctt = os.environ.get('OLD_CTT_DIR')
    default_old_ctt = Path(default_old_ctt) if default_old_ctt else None

    parser = argparse.ArgumentParser(
        description='Run old and new ctt on the same inputs and compare full JSON (all parameters).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--old-ctt-dir',
        type=Path,
        default=default_old_ctt,
        help='Path to old libcamera ctt directory (contains ctt.py). Can set OLD_CTT_DIR env instead.',
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=default_input,
        help='Base directory containing one subdir per sensor (e.g. tuning_files)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/tmp/ctt_compare'),
        help='Directory for old/ and new/ outputs',
    )
    parser.add_argument(
        '--sensors',
        nargs='+',
        default=['imx219', 'imx477', 'imx708', 'imx708_wide'],
        help='Sensor names (subdirs under --input-dir)',
    )
    parser.add_argument(
        '--skip-run',
        action='store_true',
        help='Skip running the tools; only compare existing old/ and new/ outputs',
    )
    args = parser.parse_args()

    if not args.skip_run:
        if args.old_ctt_dir is None or not (args.old_ctt_dir / 'ctt.py').exists():
            parser.error(
                '--old-ctt-dir must point to a directory containing ctt.py '
                '(e.g. libcamera/utils/raspberrypi/ctt). Set OLD_CTT_DIR or pass --old-ctt-dir.'
            )
        if not args.input_dir.exists():
            parser.error(f'--input-dir not found: {args.input_dir}')
        for s in args.sensors:
            if not (args.input_dir / s).exists():
                parser.error(f'Sensor input dir not found: {args.input_dir / s}')

        print('Running old ctt...')
        t0 = time.perf_counter()
        run_old_ctt(args.old_ctt_dir, args.input_dir, args.output_dir / 'old', args.sensors)
        old_elapsed = time.perf_counter() - t0
        print(f'Old ctt finished in {old_elapsed:.2f}s')

        print('Running new ctt...')
        t0 = time.perf_counter()
        run_new_ctt(repo_root, args.input_dir, args.output_dir / 'new', args.sensors)
        new_elapsed = time.perf_counter() - t0
        print(f'New ctt finished in {new_elapsed:.2f}s')

    ok_ccm = compare_ccms(args.output_dir / 'old', args.output_dir / 'new', args.sensors)
    ok_full = compare_all_params(args.output_dir / 'old', args.output_dir / 'new', args.sensors)
    sys.exit(0 if (ok_ccm and ok_full) else 1)


if __name__ == '__main__':
    main()
