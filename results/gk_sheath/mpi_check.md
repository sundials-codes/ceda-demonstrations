# MPI check, `rt_gk_sheath_2x2v_p1` (default 4 x 8 cells)

Made by `mpi_check.py` on 2026-10-27 on math28, fork commit `2eeb5908a`. Each layout is compared file by file with the 1-rank run of the same method; the max absolute and relative (to max|np1|) differences over all .gkyl outputs are listed with the file that attains the max relative difference, followed by the max relative difference of the field-energy and electron integrated-moment time series for t > 1e-6.

| method | layout | steps | wall (s) | files | max abs diff vs np1 | max rel diff | worst file | field E vs np1 | elc moms vs np1 |
|---|---|---|---|---|---|---|---|---|---|
| native | np1 | 2691 | 14.7 | 117 | 0.00e+00 | 0.00e+00 | - | 0.0e+00 | 0.0e+00 |
| native | np2_d2 | 2691 | 24.4 | 117 | 2.09e-07 | 1.35e-13 | field_energy_dot.gkyl | 4.3e-16 | 5.3e-16 |
| native | np4_d4 | 2691 | 4.7 | 117 | 1.79e-07 | 1.16e-13 | field_energy_dot.gkyl | 4.3e-16 | 5.3e-16 |
| native | np8_d8 | 2691 | 3.0 | 117 | 1.79e-07 | 1.16e-13 | field_energy_dot.gkyl | 4.3e-16 | 5.3e-16 |
| opsplit | np1 | 1360 | 17.4 | 117 | 0.00e+00 | 0.00e+00 | - | 0.0e+00 | 0.0e+00 |
| opsplit | np2_d2 | 1360 | 30.9 | 117 | 1.66e+00 | 1.08e-06 | field_energy_dot.gkyl | 4.3e-08 | 7.6e-09 |
| opsplit | np4_d4 | 1360 | 5.3 | 117 | 1.61e+00 | 1.05e-06 | field_energy_dot.gkyl | 3.5e-08 | 9.2e-09 |
| opsplit | np8_d8 | 1360 | 3.7 | 117 | 2.21e+00 | 1.44e-06 | field_energy_dot.gkyl | 4.4e-08 | 1.1e-08 |
| extsts_fixed | np1 | 1360 | 14.3 | 117 | 0.00e+00 | 0.00e+00 | - | 0.0e+00 | 0.0e+00 |
| extsts_fixed | np2_d2 | 1360 | 26.7 | 117 | 8.94e-08 | 5.81e-14 | field_energy_dot.gkyl | 4.3e-16 | 5.3e-16 |
| extsts_fixed | np4_d4 | 1360 | 4.1 | 117 | 1.81e+13 | 1.82e-10 | elc_M1_1.gkyl | 2.3e-13 | 1.1e-14 |
| extsts_fixed | np8_d8 | 1360 | 2.4 | 117 | 1.81e+13 | 1.82e-10 | elc_M1_1.gkyl | 2.3e-13 | 1.1e-14 |
| extsts_adaptive | np1 | 1735 | 18.1 | 117 | 0.00e+00 | 0.00e+00 | - | 0.0e+00 | 0.0e+00 |
| extsts_adaptive | np2_d2 | 1736 | 24.0 | 117 | 6.74e-11 | 1.46e-02 | dt.gkyl | 2.7e-05 | 1.7e-05 |
| extsts_adaptive | np4_d4 | 1733 | 5.2 | 117 | 6.98e-11 | 1.51e-02 | dt.gkyl | 1.7e-05 | 4.5e-05 |
| extsts_adaptive | np8_d8 | 1730 | 3.1 | 117 | 8.47e-11 | 1.84e-02 | dt.gkyl | 2.0e-05 | 3.9e-05 |
