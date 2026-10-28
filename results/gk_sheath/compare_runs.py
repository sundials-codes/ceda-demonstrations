"""Compare every Gkeyll .gkyl output of two run directories.

Usage: python compare_runs.py DIR_A DIR_B [--prefix-a PA --prefix-b PB] [--rtol R --atol A]

Files are matched by name after stripping the given prefixes (the output prefix is the
executable name, so runs made from differently named executables can be compared). Each
file is loaded with postgkyl and the values compared elementwise. The script prints one
line per file (max absolute and relative difference) and a summary, and exits with status 1
if any file differs by more than the tolerances (default: must be bit-identical).

Daniel R. Reynolds
Math & Stat @ UMBC
"""
import argparse
import os
import sys

import numpy as np
import postgkyl as pg


def load_values(path):
    data = pg.GData(path)
    return np.asarray(data.get_values(), dtype=float)


def list_outputs(dirname, prefix):
    names = {}
    for f in sorted(os.listdir(dirname)):
        if f.endswith(".gkyl") and f.startswith(prefix):
            names[f[len(prefix):]] = os.path.join(dirname, f)
    return names


def compare_dirs(dir_a, dir_b, prefix_a="", prefix_b="", rtol=0.0, atol=0.0, verbose=True):
    """Return (num_compared, list of (name, reason) failures)."""
    files_a = list_outputs(dir_a, prefix_a)
    files_b = list_outputs(dir_b, prefix_b)
    failures = []
    only_a = sorted(set(files_a) - set(files_b))
    only_b = sorted(set(files_b) - set(files_a))
    for name in only_a:
        failures.append((name, "missing in B"))
    for name in only_b:
        failures.append((name, "missing in A"))
    common = sorted(set(files_a) & set(files_b))
    for name in common:
        try:
            va = load_values(files_a[name])
            vb = load_values(files_b[name])
        except Exception as exc:  # unreadable file: report, don't abort the whole comparison
            failures.append((name, f"load error: {exc}"))
            continue
        if va.shape != vb.shape:
            failures.append((name, f"shape {va.shape} vs {vb.shape}"))
            if verbose:
                print(f"{name:60s} shape mismatch {va.shape} vs {vb.shape}")
            continue
        diff = np.abs(va - vb)
        max_abs = float(np.nanmax(diff)) if diff.size else 0.0
        scale = float(np.nanmax(np.abs(va))) if va.size else 0.0
        max_rel = max_abs / scale if scale > 0 else max_abs
        nan_mismatch = bool(np.any(np.isnan(va) != np.isnan(vb)))
        ok = (not nan_mismatch) and (max_abs <= atol + rtol * scale)
        if verbose:
            print(f"{name:60s} max|a-b| = {max_abs:9.3e}  rel = {max_rel:9.3e}  {'ok' if ok else 'DIFFERS'}")
        if not ok:
            failures.append((name, f"max abs {max_abs:.3e}, rel {max_rel:.3e}"))
    return len(common), failures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--prefix-a", default="")
    ap.add_argument("--prefix-b", default="")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    n, failures = compare_dirs(args.dir_a, args.dir_b, args.prefix_a, args.prefix_b,
                               args.rtol, args.atol, verbose=not args.quiet)
    print(f"\n{n} files compared, {len(failures)} differ or are missing")
    for name, reason in failures:
        print(f"  {name}: {reason}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
