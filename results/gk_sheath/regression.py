"""Regression and determinism checks for the SUNDIALS steppers in Gkeyll.

Usage:
  python regression.py make  [--configs NAME ...] [--jobs N]   # run and store baseline/<config>/
  python regression.py check [--configs NAME ...] [--jobs N]   # rerun, compare with baseline/
  python regression.py list

Each configuration is a copy of an input file with its `.sundials_stepper` block edited (see
CONFIGS), compiled with the staging folder's `gk_compile` (common.sh) and run in its own
scratch directory. `make` copies the outputs to baseline/<config>/ (git-ignored); `check`
compares every .gkyl output with that baseline bit-for-bit (compare_runs.py).

Intended use: `make` before a change to the fork, `check` after it. The methods that existed
before the ExtSTS work (operator split, SSP-RK3 alone, LSRK SSP, LSRK RKL on the LBO test)
must stay bit-identical. `check` right after `make` on the same build is a run-to-run
determinism test.

Daniel R. Reynolds
Math & Stat @ UMBC
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FORK = os.path.join(REPO, "deps", "gkeyll-extsts")
BASELINE = os.path.join(HERE, "baseline")
SHEATH = os.path.join(HERE, "rt_gk_sheath_2x2v_p1.c")
LBO = os.path.join(FORK, "gyrokinetic", "creg", "rt_gk_lbo_relax_sundials_1x2v_p1.c")

# Each configuration: source input file and a list of (regex, replacement) edits. Every regex
# must match exactly once, so a changed input file fails loudly instead of silently running
# the wrong method. `rk_method` edits replace the active (uncommented) `.rk_method` line.
RK_LINE = r"^( *)\.rk_method = .*$"


def rk(method, extra=""):
    return (RK_LINE, r"\1.rk_method = " + method + "," + extra)


M = "GKYL_SUNDIALS_METHOD_"
CONFIGS = {
    # rt_gk_sheath_2x2v_p1: pre-existing methods (must stay bit-identical across code changes)
    "sheath_native":   (SHEATH, [rk(M + "NONE")]),
    "sheath_opsplit":  (SHEATH, [rk(M + "RK_SSP_3_3 | " + M + "LSRK_RKL_2")]),
    "sheath_ssprk3":   (SHEATH, [rk(M + "RK_SSP_3_3")]),
    "sheath_lsrkssp_opsplit": (SHEATH, [rk(M + "LSRK_SSP_S_3 | " + M + "LSRK_RKL_2")]),
    # ExtSTS (fixed step must stay bit-identical; adaptive is a determinism check)
    "sheath_extsts_fixed": (SHEATH, [rk(M + "MRI_EXTSTS | " + M + "LSRK_RKL_2",
                                        r" .fixed_step = true,")]),
    "sheath_extsts_adaptive": (SHEATH, [rk(M + "MRI_EXTSTS | " + M + "LSRK_RKL_2")]),
    # rt_gk_lbo_relax_sundials_1x2v_p1: STS alone, with Gkeyll's and SUNDIALS' eigenvalue estimate
    "lbo_rkl_dee_gkeyll":   (LBO, [rk(M + "LSRK_RKL_2")]),
    "lbo_rkl_dee_sundials": (LBO, [rk(M + "LSRK_RKL_2"),
                                   (r"\.dee_by_gkeyll = true", ".dee_by_gkeyll = false")]),
}


def edit_input(src, edits, dst):
    text = open(src).read()
    for pattern, repl in edits:
        text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
        if n != 1:
            raise RuntimeError(f"{os.path.basename(src)}: pattern {pattern!r} matched {n} times, expected 1")
    with open(dst, "w") as f:
        f.write(text)


def run_config(name, workroot):
    """Build and run one configuration in workroot/name; return (name, run dir, exe name, ok)."""
    src, edits = CONFIGS[name]
    exe = os.path.splitext(os.path.basename(src))[0]
    rundir = os.path.join(workroot, name)
    os.makedirs(rundir, exist_ok=True)
    edit_input(src, edits, os.path.join(rundir, exe + ".c"))
    build = subprocess.run(["bash", "-c", f'source "{HERE}/common.sh" && gk_compile "{exe}.c" "{exe}"'],
                           cwd=rundir, capture_output=True, text=True)
    if build.returncode != 0:
        print(f"[{name}] build failed:\n{build.stderr}")
        return name, rundir, exe, False
    with open(os.path.join(rundir, "run.log"), "w") as log:
        run = subprocess.run([f"./{exe}"], cwd=rundir, stdout=log, stderr=subprocess.STDOUT)
    ok = run.returncode == 0
    summary = subprocess.run(["grep", "-E", "Total simulation time|Time loop|time-step", os.path.join(rundir, "run.log")],
                             capture_output=True, text=True).stdout.strip().splitlines()
    steps = [l for l in summary if "time-step" in l]
    last_step = steps[-1].split()[2] if steps else "?"
    total = [l for l in summary if "Total simulation" in l or "Time loop" in l]
    print(f"[{name}] {'ok' if ok else 'FAILED'}: {last_step} steps, {total[-1] if total else 'no timing'}")
    return name, rundir, exe, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["make", "check", "list"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), help="subset of configurations")
    ap.add_argument("--jobs", type=int, default=8, help="configurations to run concurrently")
    args = ap.parse_args()
    if args.mode == "list":
        for name, (src, edits) in CONFIGS.items():
            print(f"{name:28s} {os.path.basename(src)}  {[e[1] for e in edits]}")
        return
    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        sys.exit(f"unknown configurations: {unknown}")

    workroot = tempfile.mkdtemp(prefix="gk_regression.")
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(lambda n: run_config(n, workroot), args.configs))
        failed = [r[0] for r in results if not r[3]]
        if args.mode == "make":
            for name, rundir, exe, ok in results:
                if not ok:
                    continue
                dst = os.path.join(BASELINE, name)
                shutil.rmtree(dst, ignore_errors=True)
                os.makedirs(dst)
                for f in os.listdir(rundir):
                    if f.endswith(".gkyl") or f == "run.log":
                        shutil.copy2(os.path.join(rundir, f), dst)
            print(f"\nbaseline stored in {BASELINE} for {len(results) - len(failed)} configurations")
        else:
            sys.path.insert(0, HERE)
            from compare_runs import compare_dirs
            report = []
            for name, rundir, exe, ok in results:
                if not ok:
                    report.append((name, "run failed"))
                    continue
                base = os.path.join(BASELINE, name)
                if not os.path.isdir(base):
                    report.append((name, "no baseline"))
                    continue
                n, failures = compare_dirs(base, rundir, exe + "-", exe + "-", verbose=False)
                report.append((name, f"{n} files bit-identical" if not failures else
                               f"{len(failures)} of {n} files differ: " + "; ".join(f"{f[0]} ({f[1]})" for f in failures[:5])))
            print("\nRegression check against baseline/:")
            bad = 0
            for name, msg in report:
                flag = "ok " if "bit-identical" in msg else "BAD"
                bad += flag == "BAD"
                print(f"  {flag} {name:28s} {msg}")
            if bad:
                sys.exit(1)
        if failed:
            sys.exit(1)
    finally:
        shutil.rmtree(workroot, ignore_errors=True)


if __name__ == "__main__":
    main()
