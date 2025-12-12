#!/usr/bin/env python
#------------------------------------------------------------
# Programmer(s):  Mustafa Aggul @ SMU
#------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------

# Filter datasets to include only 
# STS methods with eigsafety = 1.1
# SSP methods 
# normtypes 1 and 2 only for adaptive runs

import pandas as pd

filename = "full_results_gk_diffusion_1x1v_p1_fixed.xlsx"
filtered_filename = "results_gk_diffusion_1x1v_p1_adaptive.xlsx"

df_a = pd.read_excel(filename, engine='openpyxl')

df_filtered_a = df_a[(df_a['eigsafety'] == 1.1) | (df_a["method"].str.contains("SSP", case=False, na=False))]

df_filtered_a = df_filtered_a[df_filtered_a['normtype'] != 3]

df_filtered_a.to_excel(filtered_filename, index=False)

filename = "full_results_gk_diffusion_1x1v_p1_fixed.xlsx"
filtered_filename = "results_gk_diffusion_1x1v_p1_fixed.xlsx"

df_f = pd.read_excel(filename, engine='openpyxl')

df_filtered_f = df_f[(df_f['eigsafety'] == 1.1) | (df_f["method"].str.contains("SSP", case=False, na=False))]

df_filtered_f.to_excel(filtered_filename, index=False)