#!/usr/bin/env python
#------------------------------------------------------------
# Programmer(s):  Mustafa Aggul @ SMU
#------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------

# Error vs runtime for SSP4 and RKL methods

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend
import matplotlib.pyplot as plt

# Set a global default font size for all text elements
plt.rcParams['font.size'] = 14

# Set specific global font sizes for titles and axis labels
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20

# For tick labels specifically
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Load data
df = pd.read_excel("results_gk_diffusion_1x1v_p1_adaptive.xlsx", sheet_name="Sheet1")

# Use Accuracy column as error metric
error_col = "Accuracy"

# Filter only valid (finite) errors
df = df[pd.to_numeric(df[error_col], errors='coerce').notnull()]
df = df[df[error_col] < 1e20]  # remove blow-ups

# Remove SSP2, SSP3 and RKC results if present
df = df[~df["method"].str.contains("SSP2", case=False, na=False)]
df = df[~df["method"].str.contains("SSP3", case=False, na=False)]
df = df[~df["method"].str.contains("RKC", case=False, na=False)]

# Unique methods
methods = df["method"].unique()
k_values = df["k"].unique()
user_dom_opts = df["user_dom_eig"].unique()

# Define a set of markers and line styles to avoid overlap confusion
markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]
linestyles = ["-", "--", "-.", ":"]

for k_val in sorted(k_values):
    for user_dom in user_dom_opts:
        plt.figure(figsize=(10,6))
        df_subset = df[(df["k"] == k_val) & (df["user_dom_eig"] == user_dom)]

        for i, method in enumerate(methods):
            df_method = df_subset[df_subset["method"] == method]
            for j, (normtype, group) in enumerate(df_method.groupby("normtype")):
                marker = markers[(i * 5 + j) % len(markers)]
                linestyle = linestyles[(i * 3 + j) % len(linestyles)]
                plt.loglog(
                    group["Runtime"],
                    group[error_col],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.5,
                    markersize=6,
                    label=f"{method}, normtype={normtype}"
                )
                plt.ylim(1.0e-14, 1.0e-1)
        plt.xlabel("Runtime")
        plt.ylabel("Error (Accuracy)")
        plt.legend(loc='lower left')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()

        # Save separate plot for each combination of k and user_dom_eig
        filename = f"error_vs_Runtime_k_{k_val}_userdom_{user_dom}.png"
        plt.savefig(filename, dpi=300)
        print(f"Plot saved as {filename}")