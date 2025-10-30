#!/usr/bin/env python
#------------------------------------------------------------
# Programmer(s):  Mustafa Aggul @ SMU
#------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------

# rtol vs fail rate for RKC and RKL methods and eigsafeties

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D  # <-- Added for custom legend handles

# Set a global default font size for all text elements
plt.rcParams['font.size'] = 14

# Set specific global font sizes for titles and axis labels
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20

# For tick labels specifically
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# Load data
df = pd.read_excel("results_gk_diffusion_1x1v_p1_adaptive.xlsx", sheet_name="Sheet1")

# Use Accuracy column as error metric
error_col = "rtol"

# Filter only valid (finite) errors
df = df[pd.to_numeric(df[error_col], errors='coerce').notnull()]
df = df[df[error_col] < 1e20]  # remove blow-ups

# Remove SSP and normtype != 2 results if present
df = df[~df["method"].str.contains("SSP", case=False, na=False)]
df = df[df['normtype'] == 2]
df = df[~df['user_dom_eig']]

# Unique methods, k values, and eigsafety options
methods = df["method"].unique()
k_values = df["k"].unique()
eigsafety_opts = df["eigsafety"].unique()

# Define a set of colors, markers and line styles to avoid overlap confusion
colors = ['blue', 'orange', 'green', 'red']
markers = ["s", "D", "o", "^", "v", "<", ">", "p", "*", "X"]
linestyles = ["-", "--", "-.", ":"]

user_dom_eig_vals = df["user_dom_eig"].unique()

for user_dom_eig in user_dom_eig_vals:
    df_subset = df[df["user_dom_eig"] == user_dom_eig]

    for k_val in sorted(k_values):
        plt.figure(figsize=(10,6))
        df_k_subset = df_subset[df_subset["k"] == k_val]

        for i, method in enumerate(methods):
            df_method_subset = df_k_subset[df_k_subset["method"] == method]

            for j, eigsafety in enumerate(eigsafety_opts):
                df_method = df_method_subset[df_method_subset["eigsafety"] == eigsafety]
                marker = markers[i % len(markers)]
                linestyle = linestyles[i % len(linestyles)]
                color = colors[j % len(colors)]
                failure_rate = df_method["Fails"] / df_method["Steps"]
                plt.semilogx(
                    df_method[error_col],
                    failure_rate,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.5,
                    markersize=6,
                    color=color,
                    label=f"{method}, eigsafety={eigsafety}"
                )
                plt.ylim(-0.1, 0.6)

        plt.ylabel("Failure Rate")
        plt.xlabel("rtol")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()

        method_handles = [
            Line2D([0], [0],
                   color='black',
                   marker=markers[i % len(markers)],
                   linestyle=linestyles[i % len(linestyles)],
                   linewidth=1.5,
                   markersize=6,
                   label=method)
            for i, method in enumerate(methods)
        ]

        eigsafety_handles = [
            Line2D([0], [0],
                   color=colors[j % len(colors)],
                   marker='o',
                   linestyle='',
                   markersize=8,
                   label=f"eigsafety={eigsafety}")
            for j, eigsafety in enumerate(eigsafety_opts)
        ]

        first_legend = plt.legend(handles=method_handles, loc='upper left')
        plt.gca().add_artist(first_legend)
        plt.legend(handles=eigsafety_handles, loc='upper right')

        # Save the plot for this combination of k and user_dom_eig
        filename = f"eigensafety_rtol_vs_FR_k_{k_val}.pdf"
        plt.savefig(filename, dpi=300)
        print(f"Plot saved as {filename}")