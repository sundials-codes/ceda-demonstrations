"""Temporal convergence study of the fixed-step methods on rt_gk_sheath_2x2v_p1.

Usage: python convergence.py [--methods native opsplit extsts] [--levels 1 0.5 0.25 0.125 0.0625]
                             [--ref-level 1/128] [--t-end 2e-6] [--jobs N]

Every method takes the CFL-set step of the explicit (SSP-RK) terms, which in this test is set by
the omega_H limit (cfl_frac_omegaH, default 1.7) rather than the advection CFL (cfl_frac). Each
level scales BOTH fractions by the same factor, so the step sequence of a level is the one of the
input file scaled by that factor (level 1 is the input file as is). Methods:
  native:  Gkeyll's SSP-RK3 (GKYL_SUNDIALS_METHOD_NONE), order 3;
  opsplit: SSP-RK3 + RKL operator split (first-order Lie splitting);
  extsts:  ExtSTS (MRI GKC21 outer + RKL inner) with fixed_step = true, order 2.
The reference is native SSP-RK3 at the (much finer) reference level. Errors are measured on the
final output frame (t = t_end) of the electron and ion distribution functions and of the field
(relative L2 norm), and on the final field energy; observed orders come from consecutive levels.
The table is written to convergence.md and the error plot to convergence.png.

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
from fractions import Fraction

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_runs import load_values
from gk_series import compile_input, edit_input, load_series, parse_log

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = "rt_gk_sheath_2x2v_p1"
M = "GKYL_SUNDIALS_METHOD_"
CFL_FRAC = 1.0  # values at level 1: the input file's cfl_frac and Gkeyll's default omega_H fraction
CFL_FRAC_OMEGAH = 1.7
METHODS = {
    "native": (M + "NONE,", 3),
    "opsplit": (M + "RK_SSP_3_3 | " + M + "LSRK_RKL_2,", 1),
    "extsts": (M + "MRI_EXTSTS | " + M + "LSRK_RKL_2, .fixed_step = true,", 2),
}
FRAMES = ["elc_1", "ion_1", "field_1"]


def run_one(method, level, t_end, workroot):
    """Build and run one (method, level) in scratch; return the run directory or None."""
    name = f"{method}_lvl{level}"
    rundir = os.path.join(workroot, name)
    os.makedirs(rundir)
    edits = [
        (r"^( *)\.rk_method = .*$", r"\1.rk_method = " + METHODS[method][0]),
        (r"^  double cfl_frac = 1\.0;", f"  double cfl_frac = {CFL_FRAC * level!r};"),
        (r"^//    \.cfl_frac_omegaH = 1e10,", f"    .cfl_frac_omegaH = {CFL_FRAC_OMEGAH * level!r},"),
        (r"^  double t_end = 6\.0e-6;", f"  double t_end = {t_end!r};"),
    ]
    edit_input(os.path.join(HERE, INPUT + ".c"), edits, os.path.join(rundir, INPUT + ".c"))
    build = compile_input(HERE, rundir, INPUT)
    if build.returncode != 0:
        print(f"[{name}] build failed:\n{build.stderr}")
        return None
    log = os.path.join(rundir, "run.log")
    with open(log, "w") as f:
        run = subprocess.run([f"./{INPUT}"], cwd=rundir, stdout=f, stderr=subprocess.STDOUT)
    if run.returncode != 0:
        print(f"[{name}] run failed, see {log}")
        return None
    steps, wall, _ = parse_log(log)
    print(f"[{name}] {steps} steps, {wall} s", flush=True)
    return rundir


def errors(rundir, refdir):
    """Relative L2 errors of the final frames and of the final field energy vs. the reference."""
    out = {}
    for frame in FRAMES:
        v = load_values(os.path.join(rundir, f"{INPUT}-{frame}.gkyl"))
        v_ref = load_values(os.path.join(refdir, f"{INPUT}-{frame}.gkyl"))
        out[frame] = float(np.linalg.norm(v - v_ref) / np.linalg.norm(v_ref))
    _, e = load_series(os.path.join(rundir, f"{INPUT}-field_energy.gkyl"))
    _, e_ref = load_series(os.path.join(refdir, f"{INPUT}-field_energy.gkyl"))
    out["field_energy"] = float(abs(e[-1, 0] - e_ref[-1, 0]) / abs(e_ref[-1, 0]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--levels", nargs="+", type=Fraction, default=[Fraction(1, 2**k) for k in range(5)],
                    help="step-size factors relative to the input file's CFL fractions")
    ap.add_argument("--ref-level", type=Fraction, default=Fraction(1, 128),
                    help="step-size factor of the native SSP-RK3 reference run")
    ap.add_argument("--t-end", type=float, default=2e-6)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    levels = sorted(args.levels, reverse=True)
    if args.ref_level >= min(levels):
        sys.exit("--ref-level must be finer than every level")

    jobs = [("native", args.ref_level)] + [(m, l) for m in args.methods for l in levels]
    workroot = tempfile.mkdtemp(prefix="gk_convergence.")
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            dirs = dict(zip(jobs, pool.map(lambda j: run_one(j[0], float(j[1]), args.t_end, workroot), jobs)))
        refdir = dirs[("native", args.ref_level)]
        if refdir is None:
            sys.exit("reference run failed")
        results = {}
        for m in args.methods:
            for l in levels:
                d = dirs[(m, l)]
                if d is None:
                    continue
                steps, wall, _ = parse_log(os.path.join(d, "run.log"))
                results[(m, l)] = dict(steps=steps, wall=wall, dt=args.t_end / steps, err=errors(d, refdir))
        ref_steps, ref_wall, _ = parse_log(os.path.join(refdir, "run.log"))
    finally:
        shutil.rmtree(workroot, ignore_errors=True)

    cols = FRAMES + ["field_energy"]
    lines = []
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.8), sharey=True)
    for m in args.methods:
        lines += [f"\n**{m}** (expected order {METHODS[m][1]})\n",
                  "| level | steps | mean dt | wall (s) | " + " | ".join(f"{c} err | order" for c in cols) + " |",
                  "|" + "---|" * (4 + 2 * len(cols))]
        prev = None
        for l in levels:
            r = results.get((m, l))
            if r is None:
                lines.append(f"| {l} | failed |")
                prev = None
                continue
            cells = []
            for c in cols:
                e = r["err"][c]
                order = "-" if prev is None or e <= 0 or prev["err"][c] <= 0 else \
                    f"{np.log(prev['err'][c] / e) / np.log(prev['dt'] / r['dt']):.2f}"
                cells.append(f"{e:.2e} | {order}")
            lines.append(f"| {l} | {r['steps']} | {r['dt']:.3e} | {r['wall']:.1f} | " + " | ".join(cells) + " |")
            prev = r
        for ax, c in zip(axes, cols):
            pts = [(results[(m, l)]["dt"], results[(m, l)]["err"][c]) for l in levels if (m, l) in results]
            if pts:
                ax.loglog(*zip(*pts), "o-", label=m)
    for ax, c in zip(axes, cols):
        dts = [r["dt"] for r in results.values()]
        if dts:
            d = np.array([min(dts), max(dts)])
            for p in (1, 2, 3):
                ax.loglog(d, 1e-3 * (d / d[1]) ** p, "k:", lw=0.8)
                ax.annotate(f"$h^{p}$", (d[0], 1e-3 * (d[0] / d[1]) ** p), fontsize=8)
        ax.set_title(c, fontsize=10); ax.set_xlabel("mean dt")
    axes[0].set_ylabel("relative L2 error vs reference"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "convergence.png"), dpi=120)

    fork_commit = subprocess.run(["git", "-C", os.path.join(HERE, "..", "..", "deps", "gkeyll-extsts"),
                                  "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(HERE, "convergence.md"), "w") as f:
        f.write(f"# Temporal convergence, `rt_gk_sheath_2x2v_p1`\n\n"
                f"Made by `convergence.py` on {subprocess.run(['date', '-I'], capture_output=True, text=True).stdout.strip()} "
                f"on {os.uname().nodename}, fork commit `{fork_commit}`; t_end = {args.t_end:g}, jobs = {args.jobs} "
                f"(runs were concurrent, so wall times are indicative only).\n\n"
                f"Level = factor applied to both `cfl_frac` ({CFL_FRAC:g}) and `cfl_frac_omegaH` ({CFL_FRAC_OMEGAH:g}); "
                f"the step is the CFL-set step of the explicit terms at that level. "
                f"Reference: native SSP-RK3 at level {args.ref_level} ({ref_steps} steps, {ref_wall:.1f} s). "
                f"Errors: relative L2 norm of the final frame (t = t_end) against the reference, and relative "
                f"error of the final field energy; order = log(err ratio)/log(dt ratio) between consecutive levels.\n"
                + table + "\n\n![convergence](convergence.png)\n")


if __name__ == "__main__":
    main()
