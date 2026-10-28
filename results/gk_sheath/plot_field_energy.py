"""Plot the field energy of a run against a reference run and report their difference.

Usage: python plot_field_energy.py run-field_energy.gkyl ref-field_energy.gkyl [label]

The relative difference is measured after interpolating the run onto the reference times,
for t > 1e-6 (after the initial transient), normalized by max |reference|.

Daniel R. Reynolds
Math & Stat @ UMBC
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gk_series import load_series, rel_diff, T_SKIP


def main():
    run_file, ref_file = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else ""
    t_run, e_run = load_series(run_file)
    t_ref, e_ref = load_series(ref_file)
    worst, t_curve, curve = rel_diff(t_run, e_run, t_ref, e_ref)
    print(f"run: {len(t_run)} samples, t in [{t_run[0]:.3e}, {t_run[-1]:.3e}]")
    print(f"ref: {len(t_ref)} samples, t in [{t_ref[0]:.3e}, {t_ref[-1]:.3e}]")
    print(f"max relative field-energy difference (t > {T_SKIP:g}): {worst:.3e}")

    fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax[0].plot(t_ref, e_ref[:, 0], "k-", label="ref (native SSP-RK3)")
    ax[0].plot(t_run, e_run[:, 0], "r--", label=f"run {label}".strip())
    ax[0].set_ylabel("field energy")
    ax[0].legend()
    ax[1].plot(t_curve, curve, "b-")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("|run - ref| / max|ref|")
    out = f"field_energy{('_' + label) if label else ''}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
