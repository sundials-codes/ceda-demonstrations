import pandas as pd

filename = "results_gk_diffusion_1x1v_p1_adaptive.xlsx"

df_a = pd.read_excel(filename, engine='openpyxl')

df_filtered_a = df_a[(df_a['eigsafety'] == 1.1) | (df_a["method"].str.contains("SSP", case=False, na=False))]

df_filtered_a.to_excel(filename, index=False)

filename = "results_gk_diffusion_1x1v_p1_fixed.xlsx"

df_f = pd.read_excel(filename, engine='openpyxl')

df_filtered_f = df_f[(df_f['eigsafety'] == 1.1) | (df_f["method"].str.contains("SSP", case=False, na=False))]

df_filtered_f.to_excel(filename, index=False)