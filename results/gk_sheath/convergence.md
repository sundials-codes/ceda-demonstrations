# Temporal convergence, `rt_gk_sheath_2x2v_p1`

Made by `convergence.py` on 2026-10-28 on math28, fork commit `85ed9cb86`; t_end = 2e-06, jobs = 16 (runs were concurrent, so wall times are indicative only).

Level = factor applied to both `cfl_frac` (1) and `cfl_frac_omegaH` (1.7); the step is the CFL-set step of the explicit terms at that level. Reference: native SSP-RK3 at level 1/128 (116787 steps, 574.1 s). Errors: relative L2 norm of the final frame (t = t_end) against the reference, and relative error of the final field energy; order = log(err ratio)/log(dt ratio) between consecutive levels.

**native** (expected order 3)

| level | steps | mean dt | wall (s) | elc_1 err | order | ion_1 err | order | field_1 err | order | field_energy err | order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 913 | 2.191e-09 | 5.2 | 7.72e-06 | - | 1.08e-06 | - | 1.31e-05 | - | 2.29e-05 | - |
| 1/2 | 1826 | 1.095e-09 | 10.3 | 1.07e-06 | 2.85 | 1.89e-07 | 2.51 | 1.89e-06 | 2.79 | 3.24e-06 | 2.82 |
| 1/4 | 3650 | 5.479e-10 | 20.0 | 1.46e-07 | 2.89 | 3.98e-08 | 2.25 | 1.69e-07 | 3.49 | 2.92e-07 | 3.48 |
| 1/8 | 7300 | 2.740e-10 | 38.7 | 8.57e-08 | 0.76 | 1.70e-08 | 1.22 | 9.95e-08 | 0.76 | 2.65e-08 | 3.46 |
| 1/16 | 14599 | 1.370e-10 | 74.6 | 5.84e-08 | 0.55 | 6.88e-09 | 1.31 | 5.26e-08 | 0.92 | 5.69e-09 | 2.22 |

**opsplit** (expected order 1)

| level | steps | mean dt | wall (s) | elc_1 err | order | ion_1 err | order | field_1 err | order | field_energy err | order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 438 | 4.566e-09 | 6.5 | 3.78e-04 | - | 9.55e-06 | - | 9.14e-05 | - | 1.63e-04 | - |
| 1/2 | 875 | 2.286e-09 | 9.0 | 1.90e-04 | 0.99 | 4.11e-06 | 1.22 | 3.45e-05 | 1.41 | 2.84e-05 | 2.52 |
| 1/4 | 1749 | 1.144e-09 | 13.7 | 9.63e-05 | 0.98 | 1.98e-06 | 1.05 | 1.85e-05 | 0.90 | 1.62e-06 | 4.14 |
| 1/8 | 3497 | 5.719e-10 | 26.2 | 4.90e-05 | 0.97 | 1.02e-06 | 0.96 | 9.73e-06 | 0.93 | 1.65e-06 | -0.03 |
| 1/16 | 6994 | 2.860e-10 | 49.6 | 2.46e-05 | 0.99 | 5.11e-07 | 1.00 | 4.93e-06 | 0.98 | 1.23e-06 | 0.43 |

**extsts** (expected order 2)

| level | steps | mean dt | wall (s) | elc_1 err | order | ion_1 err | order | field_1 err | order | field_energy err | order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 438 | 4.566e-09 | 4.8 | 1.16e-04 | - | 1.13e-05 | - | 1.87e-04 | - | 3.41e-04 | - |
| 1/2 | 875 | 2.286e-09 | 9.4 | 2.34e-05 | 2.31 | 2.71e-06 | 2.06 | 3.99e-05 | 2.23 | 7.12e-05 | 2.26 |
| 1/4 | 1749 | 1.144e-09 | 18.7 | 5.18e-06 | 2.18 | 6.80e-07 | 2.00 | 8.65e-06 | 2.21 | 1.57e-05 | 2.19 |
| 1/8 | 3497 | 5.719e-10 | 36.4 | 1.12e-06 | 2.21 | 1.82e-07 | 1.90 | 1.97e-06 | 2.14 | 3.48e-06 | 2.17 |
| 1/16 | 6994 | 2.860e-10 | 69.1 | 2.62e-07 | 2.10 | 4.88e-08 | 1.90 | 4.30e-07 | 2.19 | 7.38e-07 | 2.24 |

![convergence](convergence.png)
