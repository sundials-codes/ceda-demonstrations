import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel("results_gk_diffusion_1x1v_p1_fixed.xlsx", sheet_name="Sheet1")

# Use Accuracy column as error metric
error_col = "Accuracy"

# Filter only valid (finite) errors
df = df[pd.to_numeric(df[error_col], errors='coerce').notnull()]
df = df[df[error_col] < 1e20]  # remove blow-ups

# Remove SSP results if present
df = df[~df["method"].str.contains("SSP", case=False, na=False)]
df = df[~df["method"].str.contains("RKL", case=False, na=False)]

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
            for j, (eigsafety, group) in enumerate(df_method.groupby("eigsafety")):
                marker = markers[(i * 5 + j) % len(markers)]
                linestyle = linestyles[(i * 3 + j) % len(linestyles)]
                plt.loglog(
                    group["h"],
                    group[error_col],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.5,
                    markersize=6,
                    label=f"{method}, eigsafety={eigsafety}"
                )

        plt.xlabel("h (step size)")
        plt.ylabel("Error (Accuracy)")
        plt.title(f"Error vs Step Size (k={k_val}, user_dom_eig={user_dom})")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()

        # Save separate plot for each combination of k and user_dom_eig
        filename = f"error_vs_h_k_{k_val}_userdom_{user_dom}.png"
        plt.savefig(filename, dpi=300)
        print(f"Plot saved as {filename}")
