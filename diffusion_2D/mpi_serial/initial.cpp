/* -----------------------------------------------------------------------------
 * Programmer(s): David J. Gardner @ LLNL
 * -----------------------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2024, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------------------
 * Serial solution and derivative functions
 * ---------------------------------------------------------------------------*/

#include "diffusion_2D.hpp"

// Compute the exact solution
int Initial(sunrealtype t, N_Vector u, UserData* udata)
{
  // Initialize u to zero (handles boundary conditions)
  N_VConst(ZERO, u);

  // Iterate over domain interior
  sunindextype istart = (udata->HaveNbrW) ? 0 : 1;
  sunindextype iend   = (udata->HaveNbrE) ? udata->nx_loc : udata->nx_loc - 1;

  sunindextype jstart = (udata->HaveNbrS) ? 0 : 1;
  sunindextype jend   = (udata->HaveNbrN) ? udata->ny_loc : udata->ny_loc - 1;

  sunrealtype* uarray = N_VGetArrayPointer(u);
  if (check_flag((void*)uarray, "N_VGetArrayPointer", 0)) { return -1; }

  for (sunindextype j = jstart; j < jend; j++)
  {
    for (sunindextype i = istart; i < iend; i++)
    {
      const sunrealtype x = (udata->is + i) * udata->dx;
      const sunrealtype y = (udata->js + j) * udata->dy;

      uarray[IDX(i, j, udata->nx_loc)] = (1.0 + 0.3*sin(2.0*x))/sqrt(5.5*M_PI)*exp(-y*y/5.5);
    }
  }

  return 0;
}
