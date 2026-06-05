#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Plot 2D advection-diffusion-reaction solution output.
# -----------------------------------------------------------------------------

import argparse
import math
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


def infer_grid_shape(nvalues, metadata, nx_arg=None, ny_arg=None):
    nx = nx_arg if nx_arg is not None else metadata_value(metadata, "nx", None, int)
    ny = ny_arg if ny_arg is not None else metadata_value(metadata, "ny", None, int)

    if nx is not None and ny is not None:
        if 2 * nx * ny != nvalues:
            raise ValueError(
                f"Grid dimensions nx={nx}, ny={ny} do not match "
                f"{nvalues} solution values per row"
            )
        return nx, ny

    npoints = nvalues // 2
    nx = math.isqrt(npoints)
    if nx * nx == npoints:
        return nx, nx

    raise ValueError(
        "Could not infer grid dimensions from the output file. "
        "Use --nx and --ny, or include '# nx' and '# ny' header lines."
    )


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


def plot_field(ax, xgrid, ygrid, field, plot_type):
    if plot_type == "surface":
        surface = ax.plot_surface(xgrid, ygrid, field, cmap="viridis", linewidth=0)
        ax.set_zlabel("value")
        return surface

    contour = ax.contourf(xgrid, ygrid, field, levels=32, cmap="viridis")
    ax.set_aspect("equal", adjustable="box")
    return contour


def build_plot(data, metadata, args):
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3 or (data.shape[1] - 1) % 2 != 0:
        raise ValueError(
            "Expected each row to contain t followed by equal-sized u and v data"
        )

    nx, ny = infer_grid_shape(data.shape[1] - 1, metadata, args.nx, args.ny)
    npoints = nx * ny
    indices = selected_snapshot_indices(data.shape[0], args)
    nsnapshots = len(indices)

    xl = metadata_value(metadata, "xl", 0.0, float)
    xu = metadata_value(metadata, "xu", 1.0, float)
    yl = metadata_value(metadata, "yl", 0.0, float)
    yu = metadata_value(metadata, "yu", 1.0, float)
    xgrid, ygrid = np.meshgrid(np.linspace(xl, xu, nx), np.linspace(yl, yu, ny))

    subplot_kwargs = {}
    if args.plotstyle == "surface":
        subplot_kwargs["projection"] = "3d"

    fig, axes = plt.subplots(
        nsnapshots,
        2,
        figsize=(10, 4 * nsnapshots),
        subplot_kw=subplot_kwargs,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(nsnapshots, 2)

    for row, data_index in enumerate(indices):
        t = data[data_index, 0]
        u = data[data_index, 1 : 1 + npoints].reshape(ny, nx)
        v = data[data_index, 1 + npoints : 1 + 2 * npoints].reshape(ny, nx)

        for col, (name, field) in enumerate((("u", u), ("v", v))):
            ax = axes[row, col]
            plot = plot_field(ax, xgrid, ygrid, field, args.plotstyle)
            ax.set_title(f"{name}, t = {t:.6g}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(plot, ax=ax, shrink=0.85)

    fig.suptitle(args.title)
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 2D ADR solution output written by WriteOutput."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="solution.dat",
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
        "--plotstyle",
        choices=("contour", "surface"),
        default="contour",
        help="plot style for each 2D field",
    )
    parser.add_argument(
        "--title",
        default=" ",
        help="overall figure title",
    )
    parser.add_argument("--nx", type=int, help="number of grid points in x")
    parser.add_argument("--ny", type=int, help="number of grid points in y")
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
