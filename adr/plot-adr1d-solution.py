#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Plot 1D advection-diffusion-reaction solution output.
# -----------------------------------------------------------------------------

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_metadata(filename):
    """Read metadata written in comment headers by OpenOutput."""
    metadata = {}

    with open(filename, "r", encoding="utf-8") as output_file:
        for line in output_file:
            line = line.strip()
            if not line.startswith("#"):
                break

            fields = line[1:].strip().split(maxsplit=1)
            if len(fields) == 2:
                metadata[fields[0]] = fields[1]

    return metadata


def metadata_value(metadata, name, default=None, value_type=float):
    if name not in metadata:
        return default

    return value_type(metadata[name])


def infer_grid_size(nvalues, metadata, nx_arg=None):
    nx = nx_arg if nx_arg is not None else metadata_value(metadata, "nx", None, int)

    if nx is not None:
        if 3 * nx != nvalues:
            raise ValueError(
                f"Grid size nx={nx} does not match {nvalues} solution values per row"
            )
        return nx

    if nvalues % 3 != 0:
        raise ValueError(
            "Expected each row to contain t followed by equal-sized u, v, and w data"
        )

    return nvalues // 3


def snapshot_indices(nt, nsnapshots):
    if nsnapshots < 1:
        raise ValueError("The number of snapshots must be at least 1")
    if nsnapshots > nt:
        raise ValueError(
            f"Requested {nsnapshots} snapshots, but the file only contains {nt}"
        )

    if nsnapshots == 1:
        return np.array([nt - 1], dtype=int)

    return np.rint(np.linspace(0, nt - 1, nsnapshots)).astype(int)


def parse_frame_indices(frames):
    frame_list = frames.strip()
    delimiters = {"(": ")", "{": "}"}
    if (
        len(frame_list) < 2
        or frame_list[0] not in delimiters
        or frame_list[-1] != delimiters[frame_list[0]]
    ):
        raise argparse.ArgumentTypeError(
            "frames must be enclosed in parentheses or braces, e.g. '(0, 1, 2, 10)'"
        )

    frame_list = frame_list[1:-1].strip()
    if not frame_list:
        raise argparse.ArgumentTypeError("frames must include at least one index")

    try:
        return np.array(
            [int(frame.strip()) for frame in frame_list.split(",")], dtype=int
        )
    except ValueError as err:
        raise argparse.ArgumentTypeError("frames must contain integer indices") from err


def selected_snapshot_indices(nt, args):
    if args.frames is None:
        return snapshot_indices(nt, args.snapshots)

    if np.any(args.frames < 0) or np.any(args.frames >= nt):
        raise ValueError(
            f"Frame indices must be between 0 and {nt - 1}; received {args.frames.tolist()}"
        )

    return args.frames


def build_plot(data, metadata, args):
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 4:
        raise ValueError(
            "Expected each row to contain t followed by u, v, and w data"
        )

    nx = infer_grid_size(data.shape[1] - 1, metadata, args.nx)
    indices = selected_snapshot_indices(data.shape[0], args)
    nsnapshots = len(indices)

    xl = args.xl if args.xl is not None else metadata_value(metadata, "xl", 0.0, float)
    xu = args.xu if args.xu is not None else metadata_value(metadata, "xu", 1.0, float)
    x = np.linspace(xl, xu, nx)

    fig, axes = plt.subplots(
        nsnapshots,
        3,
        figsize=(13, 3.5 * nsnapshots),
        constrained_layout=True,
        squeeze=False,
    )

    for row, data_index in enumerate(indices):
        t = data[data_index, 0]
        u = data[data_index, 1 : 1 + nx]
        v = data[data_index, 1 + nx : 1 + 2 * nx]
        w = data[data_index, 1 + 2 * nx : 1 + 3 * nx]

        for col, (name, field) in enumerate((("u", u), ("v", v), ("w", w))):
            ax = axes[row, col]
            ax.plot(x, field)
            ax.set_title(f"{name}, t = {t:.6g}")
            ax.set_xlabel("x")
            ax.set_ylabel(name)
            ax.grid(linestyle="--", linewidth=0.5)

    fig.suptitle(args.title)
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 1D ADR solution output written by WriteOutput."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="advection_diffusion_reaction.out",
        help="solution output file to load with numpy.loadtxt",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=1,
        help="number of evenly-spaced time snapshots to plot",
    )
    parser.add_argument(
        "--frames",
        type=parse_frame_indices,
        help="specific zero-based snapshot indices to plot, e.g. '(0, 1, 2, 10)'  (overrides --snapshots)",
    )
    parser.add_argument(
        "--title",
        default=" ",
        help="overall figure title",
    )
    parser.add_argument("--nx", type=int, help="number of grid points")
    parser.add_argument("--xl", type=float, help="x-domain lower boundary")
    parser.add_argument("--xu", type=float, help="x-domain upper boundary")
    parser.add_argument("--save", help="save the figure to this file")
    return parser.parse_args()


def main():
    args = parse_args()
    filename = Path(args.filename)
    metadata = read_metadata(filename)
    data = np.loadtxt(filename)
    fig = build_plot(data, metadata, args)

    if args.save:
        fig.savefig(args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
