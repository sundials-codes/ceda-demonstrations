"""Helpers for Gkeyll integrated-diagnostic time series (shared by the staging scripts).

Daniel R. Reynolds
Math & Stat @ UMBC
"""
import os
import re
import subprocess

import numpy as np
import postgkyl as pg

T_SKIP = 1e-6  # ignore the initial transient when measuring differences
NEGLIGIBLE = 1e-12  # components smaller than this fraction of the largest one are skipped

# Time series compared against the reference run (file suffix after "<prefix>-").
SERIES = [
    "field_energy",
    "elc_integrated_moms",
    "ion_integrated_moms",
    "elc_bflux_ylower_integrated_HamiltonianMoments",
    "ion_bflux_ylower_integrated_HamiltonianMoments",
]


def load_series(fname):
    """Return (times, values[nt, ncomp]) of an integrated-diagnostic file."""
    data = pg.GData(fname)
    t = np.asarray(data.get_grid()[0])
    vals = np.asarray(data.get_values())
    return t, vals


def rel_diff(t_run, v_run, t_ref, v_ref, t_skip=T_SKIP):
    """Max over t > t_skip and components of |run - ref| / max|ref| (per component, skipping
    components that are zero up to roundoff), after interpolating the run onto the reference
    times; also return the per-time curve of the max over components."""
    mask = t_ref > t_skip
    curve = np.zeros(mask.sum())
    worst = 0.0
    scales = np.max(np.abs(v_ref), axis=0)
    for c in range(v_ref.shape[1]):
        scale = scales[c]
        if scale <= NEGLIGIBLE * scales.max():
            continue  # component is zero up to roundoff (e.g. the parallel momentum here)
        interp = np.interp(t_ref[mask], t_run, v_run[:, c])
        d = np.abs(interp - v_ref[mask, c]) / scale
        curve = np.maximum(curve, d)
        worst = max(worst, float(d.max()))
    return worst, t_ref[mask], curve


def compare_with_ref(rundir, prefix, refdir, refprefix="gk_sheath_2x2v_p1"):
    """Max relative difference of each SERIES entry between a run and the reference."""
    out = {}
    for name in SERIES:
        t_run, v_run = load_series(os.path.join(rundir, f"{prefix}-{name}.gkyl"))
        t_ref, v_ref = load_series(os.path.join(refdir, f"{refprefix}-{name}.gkyl"))
        out[name] = rel_diff(t_run, v_run, t_ref, v_ref)[0]
    return out


def edit_input(src, edits, dst):
    """Copy an input file applying (regex, replacement) edits; each must match exactly once."""
    text = open(src).read()
    for pattern, repl in edits:
        text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
        if n != 1:
            raise RuntimeError(f"{os.path.basename(src)}: pattern {pattern!r} matched {n} times, expected 1")
    with open(dst, "w") as f:
        f.write(text)


def compile_input(here, rundir, exe):
    """Compile rundir/exe.c with common.sh's gk_compile; return the CompletedProcess."""
    return subprocess.run(["bash", "-c", f'source "{here}/common.sh" && gk_compile "{exe}.c" "{exe}"'],
                          cwd=rundir, capture_output=True, text=True)


def parse_log(logfile):
    """Gkeyll step count, wall time, and the last ARKODE statistics block (if printed)."""
    steps, wall, stats = None, None, {}
    for line in open(logfile):
        if line.startswith("Taking time-step"):
            steps = int(line.split()[2])
        elif line.startswith("Total simulation time"):
            wall = float(line.split()[-2])
        elif " = " in line and not line.startswith(" "):
            key, _, val = line.partition(" = ")
            key = key.strip()
            if key in ("Steps", "Step attempts", "Error test fails",
                       "Stability limited steps", "Accuracy limited steps"):
                stats[key] = int(val)
    return steps, wall, stats
