#!/usr/bin/env python
#------------------------------------------------------------
# Programmer(s):  Mustafa Aggul @ SMU
#------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------

# This script generates error vs. runtime plots for STS methods
# The aim is to compare user provided eigenvalue estimates vs the approximate eigenvalue estimates
# obtained by SUNDIALS' power method implementation.

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D  # <-- Added for custom legend handles

# Set a global default font size for all text elements
plt.rcParams['font.size'] = 24

# Set specific global font sizes for titles and axis labels
plt.rcParams['axes.titlesize'] = 34
plt.rcParams['axes.labelsize'] = 34

# For tick labels specifically
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24

# Load data
df = pd.read_excel("../full_results_gk_diffusion_1x1v_p1_adaptive.xlsx", sheet_name="Sheet1")

# Use Accuracy column as error metric
error_col = "Accuracy"

# Filter only valid (finite) errors
df = df[pd.to_numeric(df[error_col], errors='coerce').notnull()]
df = df[df[error_col] < 1e5]  # remove blow-ups

# Remove SSP and normtype 1 results if present
df = df[~df["method"].str.contains("SSP", case=False, na=False)]
df = df[(df["eigsafety"] == 1.1) | (df["method"].str.startswith("SSP"))]
df = df[df["normtype"] == 2]

# Unique methods, k values, and eigsafety options
methods = df["method"].unique()
k_values = df["k"].unique()
user_dom_opts = ["$\\lambda_{approx}$", "$\\lambda_{user}$"]

# Define a set of colors, markers and line styles to avoid overlap confusion
colors = ['blue', 'orange', 'green', 'red']
markers = ["s", "D", "o", "^", "v", "<", ">", "p", "*", "X"]
linestyles = ["-", "--", "-.", ":"]

for idk, k_val in enumerate(k_values):
    for normtype in df["normtype"].unique():
        plt.figure(figsize=(10,6))
        df_subset = df[(df["k"] == k_val) & (df["normtype"] == normtype)]

        for i, method in enumerate(methods):
            df_method_subset = df_subset[df_subset["method"] == method]
            for j, user_dom_eig in enumerate(df["user_dom_eig"].unique()):
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
                )
                plt.ylim(1.0e-7, 1.0e-2)
        plt.xlabel("Runtime")
        plt.ylabel("Error (Accuracy)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()

        if idk == 2:
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
                    label=f"{user_dom_eig}")
                for j, user_dom_eig in enumerate(user_dom_opts)
            ]

            first_legend = plt.legend(handles=method_handles, loc='upper right')
            plt.gca().add_artist(first_legend)
            plt.legend(handles=user_dom_eig_handles, loc='upper center')

        # Save separate plot for each combination of k and normtype
        filename = f"error_vs_Runtime_k_{k_val}_normtype_{normtype}.pdf"
        plt.savefig(filename, dpi=300)
        print(f"Plot saved as {filename}")