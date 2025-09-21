import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel("results_gk_diffusion_1x1v_p1_adaptive.xlsx", sheet_name="Sheet1")

# Use Accuracy column as error metric
error_col = "Accuracy"

# Filter only valid (finite) errors
df = df[pd.to_numeric(df[error_col], errors='coerce').notnull()]
df = df[df[error_col] < 1e20]  # remove blow-ups

# Remove SSP results if present
#df = df[~df["method"].str.contains("SSP", case=False, na=False)]
#df = df[~df["method"].str.contains("RKC", case=False, na=False)]

# Unique methods
methods = df["method"].unique()
k_values = df["k"].unique()
normtype_opts = df["normtype"].unique()
df = df[~df["method"].str.contains("SSP", case=False, na=False)]

# Define a set of markers and line styles to avoid overlap confusion
markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]
linestyles = ["-", "--", "-.", ":"]

for k_val in sorted(k_values):
    for normtype in normtype_opts:
        plt.figure(figsize=(10,6))
        df_subset = df[(df["k"] == k_val) & (df["normtype"] == normtype)]

        for i, method in enumerate(methods):
            df_method = df_subset[df_subset["method"] == method]
            for j, (user_dom_eig, group) in enumerate(df_method.groupby("user_dom_eig")):
                marker = markers[(i * 5 + j) % len(markers)]
                linestyle = linestyles[(i * 3 + j) % len(linestyles)]
                plt.loglog(
                    group["rtol"],
                    group[error_col],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.5,
                    markersize=6,
                    label=f"{method}, user_dom_eig={user_dom_eig}"
                )

        plt.xlabel("rtol")
        plt.ylabel("Error (Accuracy)")
        plt.title(f"Error vs rtol (k={k_val}, normtype={normtype})")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()

        # Save separate plot for each combination of k and normtype
        filename = f"error_vs_rtol_k_{k_val}_normtype_{normtype}.png"
        plt.savefig(filename, dpi=300)
        print(f"Plot saved as {filename}")