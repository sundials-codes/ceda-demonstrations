#!/usr/bin/env python
#------------------------------------------------------------
# Programmer(s):  Mustafa Aggul @ SMU
#------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------

# Error vs relative tolerance for the power iteration 
# and the user provided dominant eigenvalue estimation

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
error_col = "Accuracy"

# Filter only valid (finite) errors
df = df[pd.to_numeric(df[error_col], errors='coerce').notnull()]
df = df[df[error_col] < 1e20]  # remove blow-ups

# Remove SSP results if present
df = df[~df["method"].str.contains("SSP", case=False, na=False)]

# Unique methods, k values, and eigsafety options
methods = df["method"].unique()
k_values = df["k"].unique()
user_dom_opts = df["user_dom_eig"].unique()
normtype_opts = df["normtype"].unique()

# Define a set of colors, markers and line styles to avoid overlap confusion
colors = ['blue', 'orange', 'green', 'red']
markers = ["s", "D", "o", "^", "v", "<", ">", "p", "*", "X"]
linestyles = ["-", "--", "-.", ":"]

for k_val in sorted(k_values):
    for normtype in normtype_opts:
        plt.figure(figsize=(10,6))
        df_subset = df[(df["k"] == k_val) & (df["normtype"] == normtype)]

        for i, method in enumerate(methods):
            df_method_subset = df_subset[df_subset["method"] == method]
            for j, user_dom_eig in enumerate(user_dom_opts):
                df_user_dom_eig = df_method_subset[df_method_subset["user_dom_eig"] == user_dom_eig]
                marker = markers[i % len(markers)]
                linestyle = linestyles[i % len(linestyles)]
                color = colors[j % len(colors)]
                plt.loglog(
                    df_user_dom_eig["Runtime"],
                    df_user_dom_eig[error_col],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.5,
                    markersize=6,
                    color=color,
                    label=f"{method}, user_dom_eig={user_dom_eig}"
                )
                plt.ylim(1.0e-7, 1.0e-2)
        plt.xlabel("Runtime")
        plt.ylabel("Error (Accuracy)")
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

        user_dom_eig_handles = [
            Line2D([0], [0],
                   color=colors[j % len(colors)],
                   marker='o',
                   linestyle='',
                   markersize=8,
                   label=f"user_dom_eig={user_dom_eig}")
            for j, user_dom_eig in enumerate(user_dom_opts)
        ]

        first_legend = plt.legend(handles=method_handles, loc='lower left')
        plt.gca().add_artist(first_legend)
        plt.legend(handles=user_dom_eig_handles, loc='lower center')

        # Save separate plot for each combination of k and normtype
        filename = f"error_vs_Runtime_k_{k_val}_normtype_{normtype}.pdf"
        plt.savefig(filename, dpi=300)
        print(f"Plot saved as {filename}")