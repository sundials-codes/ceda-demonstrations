import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Load data
df = pd.read_excel("results_gk_diffusion_1x1v_p1_adaptive.xlsx", sheet_name="Sheet1")

# Use Accuracy column as error metric
error_col = "rtol"

# Filter only valid (finite) errors
df = df[pd.to_numeric(df[error_col], errors='coerce').notnull()]
df = df[df[error_col] < 1e20]  # remove blow-ups

# Remove SSP results if present
df = df[~df["method"].str.contains("SSP2", case=False, na=False)]
df = df[~df["method"].str.contains("SSP3", case=False, na=False)]
df = df[~df["method"].str.contains("SSP4", case=False, na=False)]

# Unique methods, k values, and normtypes
methods = df["method"].unique()
k_values = df["k"].unique()
normtype_opts = df["normtype"].unique()

# Define a set of markers and line styles to avoid overlap confusion
markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]
linestyles = ["-", "--", "-.", ":"]

user_dom_eig_vals = df["user_dom_eig"].unique()

for user_dom_eig in user_dom_eig_vals:
    df_subset = df[df["user_dom_eig"] == user_dom_eig]
    
    for k_val in sorted(k_values):
        plt.figure(figsize=(10,6))
        df_k_subset = df_subset[df_subset["k"] == k_val]


        for i, method in enumerate(methods):
            df_method_subset = df_k_subset[df_k_subset["method"] == method]

            for normtype in normtype_opts:
                df_method = df_method_subset[df_method_subset["normtype"] == normtype]
                marker = markers[i % len(markers)]
                linestyle = linestyles[i % len(linestyles)]
                plt.semilogx(
                    df_method[error_col],
                    df_method["Fails"],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.5,
                    markersize=6,
                    label=f"{method}, normtype={normtype}"
                )
                # Force y-axis ticks to be integers
                ax = plt.gca()
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.ylabel("Fails")
        plt.xlabel("rtol")
        plt.title(f"rtol vs Fails (k={k_val}, user_dom_eig={user_dom_eig})")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()

        # Save the plot for this combination of k and user_dom_eig
        filename = f"rtol_vs_Fails_user_dom_eig_{user_dom_eig}_k_{k_val}.png"
        plt.savefig(filename, dpi=300)
        print(f"Plot saved as {filename}")