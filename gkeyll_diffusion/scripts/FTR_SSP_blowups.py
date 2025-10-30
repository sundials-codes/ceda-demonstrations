#!/usr/bin/env python
#------------------------------------------------------------
# Programmer(s):  Mustafa Aggul @ SMU
#------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------

# Error vs fixed step size h Plots for SSP and STS methods

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
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# Load data
df = pd.read_excel("results_gk_diffusion_1x1v_p1_fixed.xlsx", sheet_name="Sheet1")

# Use Accuracy column as error metric
error_col = "Accuracy"

# Filter only valid (finite) errors
df = df[pd.to_numeric(df[error_col], errors='coerce').notnull()]
df = df[df[error_col] < 1e20]  # remove blow-ups

# Unique methods, k values, normtypes and dom_eig options
methods = df["method"].unique()
k_values = df["k"].unique()
norm_types = df["normtype"].unique()
user_dom_opts = df["user_dom_eig"].unique()

# Define a set of markers and line styles to avoid overlap confusion
markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]
linestyles = ["-", "--", "-.", ":"]

for norm_type in sorted(norm_types):
    for k_val in sorted(k_values):
        for user_dom in user_dom_opts:
            plt.figure(figsize=(10,6))
            df_subset = df[(df["k"] == k_val) & (df["user_dom_eig"] == user_dom)]

            for i, method in enumerate(methods):
                df_method = df_subset[df_subset["method"] == method]
                for j, (eigsafety, group) in enumerate(df_method.groupby("eigsafety")):
                    marker = markers[(i + j) % len(markers)]
                    linestyle = linestyles[(i + j) % len(linestyles)]
                    plt.loglog(
                        group["h"],
                        group[error_col],
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=1.5,
                        markersize=6,
                        label=f"{method}"
                    )
                    plt.ylim(1.0e-13, 1.0e-3)
            plt.xlabel("h")
            plt.ylabel("Error (Accuracy)")
            plt.legend(loc='upper left')
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.tight_layout()

            # Save separate plot for each combination of k and user_dom_eig
            filename = f"error_vs_h_k_{k_val}_userdom_{user_dom}.pdf"
            plt.savefig(filename, dpi=300)
            print(f"Plot saved as {filename}")