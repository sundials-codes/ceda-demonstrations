"""MPI correctness check: compare multi-rank runs of rt_gk_sheath_2x2v_p1 with the 1-rank run.

Usage: python mpi_check.py [--methods ...] [--layouts ...] [--jobs N]

For each method the input is built once; it is then run with each MPI layout (ranks and
-d decomposition cuts in z; the gyrokinetic app allows no x decomposition) in its own scratch directory under gkylsoft's mpirun, and every
.gkyl output is compared with the 1-rank run (compare_runs.py). The table of max absolute and
relative differences is printed and written to mpi_check.md. This is a correctness check at the
test's default resolution (4 x 8 cells), not a performance study.

Daniel R. Reynolds
Math & Stat @ UMBC
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from compare_runs import compare_dirs, list_outputs, load_values
from gk_series import SERIES, compare_with_ref, compile_input, edit_input, parse_log
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MPIRUN = os.path.join(REPO, "deps", "gkeyll-extsts", "gkylsoft", "openmpi", "bin", "mpirun")
INPUT = "rt_gk_sheath_2x2v_p1"
M = "GKYL_SUNDIALS_METHOD_"
RK = r"^( *)\.rk_method = .*$"
METHODS = {
    "native":          [(RK, r"\1.rk_method = " + M + "NONE,")],
    "opsplit":         [(RK, r"\1.rk_method = " + M + "RK_SSP_3_3 | " + M + "LSRK_RKL_2,")],
    "extsts_fixed":    [(RK, r"\1.rk_method = " + M + "MRI_EXTSTS | " + M + "LSRK_RKL_2, .fixed_step = true,")],
    "extsts_adaptive": [(RK, r"\1.rk_method = " + M + "MRI_EXTSTS | " + M + "LSRK_RKL_2,")],
}
LAYOUTS = {  # name: (ranks, extra args); this app only decomposes in z (-d), not x (-c)
    "np1":    (1, []),
    "np2_d2": (2, ["-M", "-d", "2"]),
    "np4_d4": (4, ["-M", "-d", "4"]),
    "np8_d8": (8, ["-M", "-d", "8"]),
}


def run_layout(method, layout, exe_path, workroot):
    ranks, extra = LAYOUTS[layout]
    rundir = os.path.join(workroot, method, layout)
    os.makedirs(rundir)
    cmd = [exe_path] + extra if ranks == 1 else [MPIRUN, "-np", str(ranks), exe_path] + extra
    with open(os.path.join(rundir, "run.log"), "w") as log:
        run = subprocess.run(cmd, cwd=rundir, stdout=log, stderr=subprocess.STDOUT)
    steps, wall, _ = parse_log(os.path.join(rundir, "run.log"))
    print(f"[{method}/{layout}] {'ok' if run.returncode == 0 else 'FAILED'}: {steps} steps, {wall} s")
    return method, layout, rundir, run.returncode == 0, steps, wall


def worst_diff(dir_a, dir_b, prefix):
    """Max absolute and relative difference over all common .gkyl files, and the worst file."""
    fa, fb = list_outputs(dir_a, prefix), list_outputs(dir_b, prefix)
    worst = (0.0, 0.0, "-")
    for name in sorted(set(fa) & set(fb)):
        va, vb = load_values(fa[name]), load_values(fb[name])
        if va.shape != vb.shape:
            return (float("inf"), float("inf"), f"{name} shape")
        d = float(np.nanmax(np.abs(va - vb))) if va.size else 0.0
        s = float(np.nanmax(np.abs(va))) if va.size else 0.0
        r = d / s if s > 0 else d
        if r > worst[1]:
            worst = (d, r, name)
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--layouts", nargs="+", default=list(LAYOUTS), choices=list(LAYOUTS))
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if "np1" not in args.layouts:
        sys.exit("the np1 layout is the comparison baseline and must be included")

    workroot = tempfile.mkdtemp(prefix="gk_mpi.")
    try:
        exes = {}
        for method in args.methods:
            bdir = os.path.join(workroot, method)
            os.makedirs(bdir)
            edit_input(os.path.join(HERE, INPUT + ".c"), METHODS[method], os.path.join(bdir, INPUT + ".c"))
            build = compile_input(HERE, bdir, INPUT)
            if build.returncode != 0:
                sys.exit(f"[{method}] build failed:\n{build.stderr}")
            exes[method] = os.path.join(bdir, INPUT)
        jobs = [(m, l) for m in args.methods for l in args.layouts]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(lambda j: run_layout(j[0], j[1], exes[j[0]], workroot), jobs))

        by = {(r[0], r[1]): r for r in results}
        lines = ["| method | layout | steps | wall (s) | files | max abs diff vs np1 | max rel diff | worst file | field E vs np1 | elc moms vs np1 |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        bad = False
        for method in args.methods:
            base = by[(method, "np1")]
            for layout in args.layouts:
                r = by[(method, layout)]
                if not r[3] or not base[3]:
                    lines.append(f"| {method} | {layout} | run failed | | | | | |"); bad = True; continue
                n, failures = compare_dirs(base[2], r[2], INPUT + "-", INPUT + "-", verbose=False)
                d, rel, worst = worst_diff(base[2], r[2], INPUT + "-")
                missing = [f for f in failures if "missing" in f[1] or "shape" in f[1]]
                if missing:
                    bad = True
                series = compare_with_ref(r[2], INPUT, base[2], INPUT)  # time series vs the np1 run
                lines.append(f"| {method} | {layout} | {r[4]} | {r[5]:.1f} | {n}{' (+%d missing/shape)' % len(missing) if missing else ''} | "
                             f"{d:.2e} | {rel:.2e} | {worst} | {series['field_energy']:.1e} | {series['elc_integrated_moms']:.1e} |")
    finally:
        shutil.rmtree(workroot, ignore_errors=True)

    table = "\n".join(lines)
    print("\n" + table)
    fork_commit = subprocess.run(["git", "-C", os.path.join(REPO, "deps", "gkeyll-extsts"), "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
    with open(os.path.join(HERE, "mpi_check.md"), "w") as f:
        f.write(f"# MPI check, `rt_gk_sheath_2x2v_p1` (default 4 x 8 cells)\n\n"
                f"Made by `mpi_check.py` on {subprocess.run(['date', '-I'], capture_output=True, text=True).stdout.strip()} "
                f"on {os.uname().nodename}, fork commit `{fork_commit}`. Each layout is compared file by file "
                f"with the 1-rank run of the same method; the max absolute and relative (to max|np1|) differences "
                f"over all .gkyl outputs are listed with the file that attains the max relative difference, followed by the "
                f"max relative difference of the field-energy and electron integrated-moment time series for t > 1e-6.\n\n"
                + table + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
