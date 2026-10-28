"""Adaptive ExtSTS tolerance sweep on rt_gk_sheath_2x2v_p1, with and without the CFL caps.

Usage: python tolerance_sweep.py [--rtols 1e-3 1e-4 1e-5] [--cases capped uncapped] [--jobs N]

Cases (the "capped" case is the input file as is; "uncapped" loosens both explicit CFL
limits, which in this test are set by the omega_H mode rather than the advection CFL):
  capped:   cfl_frac = 1,  cfl_frac_omegaH = 1 (default)
  uncapped: cfl_frac = 10, cfl_frac_omegaH = 17
For each case and relative tolerance the adaptive ExtSTS (RKL inner) run is built and run in
scratch with print_arkode_stats on, and compared with ref/ (native SSP-RK3): steps, step
attempts, error-test failures, wall time, and the max relative difference of the integrated
diagnostics for t > 1e-6. The table is printed and written to tolerance_sweep.md, and the
field-energy error curves to tolerance_sweep.png.

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gk_series import (SERIES, compare_with_ref, compile_input, edit_input, load_series,
                       parse_log, rel_diff)

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = "rt_gk_sheath_2x2v_p1"
REF = os.path.join(HERE, "ref")
M = "GKYL_SUNDIALS_METHOD_"
CASES = {
    "capped": [],
    "uncapped": [(r"^  double cfl_frac = 1\.0;", "  double cfl_frac = 10.0;"),
                 (r"^//    \.cfl_frac_omegaH = 1e10,", "    .cfl_frac_omegaH = 17.0,")],
}


def run_one(case, rtol, workroot):
    name = f"{case}_rtol{rtol:g}"
    rundir = os.path.join(workroot, name)
    os.makedirs(rundir)
    edits = CASES[case] + [
        (r"^( *)\.rk_method = .*$", r"\1.rk_method = " + M + "MRI_EXTSTS | " + M + "LSRK_RKL_2,"
         + r" .print_arkode_stats = true,"),
        (r"\.relative_tolerance = [^,]*,", f".relative_tolerance = {rtol:g},"),
    ]
    edit_input(os.path.join(HERE, INPUT + ".c"), edits, os.path.join(rundir, INPUT + ".c"))
    build = compile_input(HERE, rundir, INPUT)
    if build.returncode != 0:
        print(f"[{name}] build failed:\n{build.stderr}")
        return name, None
    log = os.path.join(rundir, "run.log")
    with open(log, "w") as f:
        run = subprocess.run([f"./{INPUT}"], cwd=rundir, stdout=f, stderr=subprocess.STDOUT)
    if run.returncode != 0:
        print(f"[{name}] run failed, see {log}")
        return name, None
    steps, wall, stats = parse_log(log)
    diffs = compare_with_ref(rundir, INPUT, REF)
    t_run, e_run = load_series(os.path.join(rundir, f"{INPUT}-field_energy.gkyl"))
    t_ref, e_ref = load_series(os.path.join(REF, "gk_sheath_2x2v_p1-field_energy.gkyl"))
    _, t_curve, curve = rel_diff(t_run, e_run, t_ref, e_ref)
    print(f"[{name}] {steps} steps, {wall} s, stats {stats}")
    return name, dict(case=case, rtol=rtol, steps=steps, wall=wall, stats=stats,
                      diffs=diffs, curve=(t_curve, curve))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rtols", nargs="+", type=float, default=[1e-3, 1e-4, 1e-5])
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if not os.path.isdir(REF):
        sys.exit("ref/ is missing; run makeref.sh first")

    jobs = [(c, r) for c in args.cases for r in args.rtols]
    workroot = tempfile.mkdtemp(prefix="gk_tolsweep.")
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(lambda j: run_one(j[0], j[1], workroot), jobs))
    finally:
        shutil.rmtree(workroot, ignore_errors=True)

    short = {"field_energy": "field E", "elc_integrated_moms": "elc moms", "ion_integrated_moms": "ion moms",
             "elc_bflux_ylower_integrated_HamiltonianMoments": "elc bflux", "ion_bflux_ylower_integrated_HamiltonianMoments": "ion bflux"}
    header = ("| case | rtol | steps | attempts | err fails | stab. lim. | acc. lim. | wall (s) | "
              + " | ".join(short[s] for s in SERIES) + " |")
    lines = [header, "|" + "---|" * (8 + len(SERIES))]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, r in results:
        if r is None:
            lines.append(f"| {name} | failed |")
            continue
        st = r["stats"]
        lines.append(f"| {r['case']} | {r['rtol']:g} | {r['steps']} | {st.get('Step attempts', '?')} | "
                     f"{st.get('Error test fails', '?')} | {st.get('Stability limited steps', '?')} | "
                     f"{st.get('Accuracy limited steps', '?')} | {r['wall']:.1f} | "
                     + " | ".join(f"{r['diffs'][s]:.1e}" for s in SERIES) + " |")
        ax.plot(*r["curve"], label=name)
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel("|field energy - ref| / max|ref|"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "tolerance_sweep.png"), dpi=120)

    fork_commit = subprocess.run(["git", "-C", os.path.join(HERE, "..", "..", "deps", "gkeyll-extsts"),
                                  "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(HERE, "tolerance_sweep.md"), "w") as f:
        f.write(f"# Adaptive ExtSTS tolerance sweep, `rt_gk_sheath_2x2v_p1`\n\n"
                f"Made by `tolerance_sweep.py` on {subprocess.run(['date', '-I'], capture_output=True, text=True).stdout.strip()} "
                f"on {os.uname().nodename}, fork commit `{fork_commit}`. ExtSTS = MRI GKC21 outer + RKL inner, "
                f"Soderlind controller; reference = native SSP-RK3 (`ref/`). Differences are max relative "
                f"differences of the integrated diagnostics for t > 1e-6.\n\n"
                f"Cases: capped = input file as is (cfl_frac = 1); uncapped = cfl_frac = 10, cfl_frac_omegaH = 17.\n\n"
                + table + "\n\n![field energy error](tolerance_sweep.png)\n")


if __name__ == "__main__":
    main()
