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
 * Serial diffusion functions
 * ---------------------------------------------------------------------------*/

#include "diffusion_2D.hpp"

// Diffusion function
int laplacian(sunrealtype t, N_Vector u, N_Vector f, UserData* udata)
{
  SUNDIALS_CXX_MARK_FUNCTION(udata->prof);

  int flag;
  sunindextype i, j;

  // Start exchange
  flag = udata->start_exchange(u);
  if (check_flag(&flag, "SendData", 1)) { return -1; }

  // Shortcuts to local number of nodes
  sunindextype nx_loc = udata->nx_loc;
  sunindextype ny_loc = udata->ny_loc;

  // Determine iteration range excluding the overall domain boundary
  sunindextype istart = (udata->HaveNbrW) ? 0 : 1;
  sunindextype iend   = (udata->HaveNbrE) ? nx_loc : nx_loc - 1;
  sunindextype jstart = (udata->HaveNbrS) ? 0 : 1;
  sunindextype jend   = (udata->HaveNbrN) ? ny_loc : ny_loc - 1;

  // Constants for computing diffusion term
  sunrealtype cx = udata->kx / (udata->dx * udata->dx);
  sunrealtype cy = udata->ky / (udata->dy * udata->dy);
  sunrealtype cc = -TWO * (cx + cy);

  // Access data arrays
  sunrealtype* uarray = N_VGetArrayPointer(u);
  if (check_flag((void*)uarray, "N_VGetArrayPointer", 0)) { return -1; }

  sunrealtype* farray = N_VGetArrayPointer(f);
  if (check_flag((void*)farray, "N_VGetArrayPointer", 0)) { return -1; }

  // Initialize rhs vector to zero (handles boundary conditions)
  N_VConst(ZERO, f);

  // Iterate over subdomain interior and add rhs diffusion term
  for (j = 1; j < ny_loc - 1; j++)
  {
    for (i = 1; i < nx_loc - 1; i++)
    {
      farray[IDX(i, j, nx_loc)] +=
        cc * uarray[IDX(i, j, nx_loc)] +
        cx * (uarray[IDX(i - 1, j, nx_loc)] + uarray[IDX(i + 1, j, nx_loc)]) +
        cy * (uarray[IDX(i, j - 1, nx_loc)] + uarray[IDX(i, j + 1, nx_loc)]);
    }
  }

  // Wait for exchange receives
  flag = udata->end_exchange();
  if (check_flag(&flag, "UserData::end_excahnge", 1)) { return -1; }

  // Iterate over subdomain boundaries and add rhs diffusion term
  sunrealtype* Warray = udata->Wrecv;
  sunrealtype* Earray = udata->Erecv;
  sunrealtype* Sarray = udata->Srecv;
  sunrealtype* Narray = udata->Nrecv;

  // West face (updates south-west and north-west corners if necessary)
  if (udata->HaveNbrW)
  {
    i = 0;
    if (udata->HaveNbrS) // South-West corner
    {
      j = 0;
      farray[IDX(i, j, nx_loc)] +=
        cc * uarray[IDX(i, j, nx_loc)] +
        cx * (Warray[j] + uarray[IDX(i + 1, j, nx_loc)]) +
        cy * (Sarray[i] + uarray[IDX(i, j + 1, nx_loc)]);
    }

    for (j = 1; j < ny_loc - 1; j++)
    {
      farray[IDX(i, j, nx_loc)] +=
        cc * uarray[IDX(i, j, nx_loc)] +
        cx * (Warray[j] + uarray[IDX(i + 1, j, nx_loc)]) +
        cy * (uarray[IDX(i, j - 1, nx_loc)] + uarray[IDX(i, j + 1, nx_loc)]);
    }

    if (udata->HaveNbrN) // North-West corner
    {
      j = ny_loc - 1;
      farray[IDX(i, j, nx_loc)] +=
        cc * uarray[IDX(i, j, nx_loc)] +
        cx * (Warray[j] + uarray[IDX(i + 1, j, nx_loc)]) +
        cy * (uarray[IDX(i, j - 1, nx_loc)] + Narray[i]);
    }
  }

  // East face (updates south-east and north-east corners if necessary)
  if (udata->HaveNbrE)
  {
    i = nx_loc - 1;
    if (udata->HaveNbrS) // South-East corner
    {
      j = 0;
      farray[IDX(i, j, nx_loc)] +=
        cc * uarray[IDX(i, j, nx_loc)] +
        cx * (uarray[IDX(i - 1, j, nx_loc)] + Earray[j]) +
        cy * (Sarray[i] + uarray[IDX(i, j + 1, nx_loc)]);
    }

    for (j = 1; j < ny_loc - 1; j++)
    {
      farray[IDX(i, j, nx_loc)] +=
        cc * uarray[IDX(i, j, nx_loc)] +
        cx * (uarray[IDX(i - 1, j, nx_loc)] + Earray[j]) +
        cy * (uarray[IDX(i, j - 1, nx_loc)] + uarray[IDX(i, j + 1, nx_loc)]);
    }

    if (udata->HaveNbrN) // North-East corner
    {
      j = ny_loc - 1;
      farray[IDX(i, j, nx_loc)] +=
        cc * uarray[IDX(i, j, nx_loc)] +
        cx * (uarray[IDX(i - 1, j, nx_loc)] + Earray[j]) +
        cy * (uarray[IDX(i, j - 1, nx_loc)] + Narray[i]);
    }
  }

  // South face (excludes corners)
  if (udata->HaveNbrS)
  {
    j = 0;
    for (i = 1; i < nx_loc - 1; i++)
    {
      farray[IDX(i, j, nx_loc)] +=
        cc * uarray[IDX(i, j, nx_loc)] +
        cx * (uarray[IDX(i - 1, j, nx_loc)] + uarray[IDX(i + 1, j, nx_loc)]) +
        cy * (Sarray[i] + uarray[IDX(i, j + 1, nx_loc)]);
    }
  }

  // North face (excludes corners)
  if (udata->HaveNbrN)
  {
    j = udata->ny_loc - 1;
    for (i = 1; i < nx_loc - 1; i++)
    {
      farray[IDX(i, j, nx_loc)] +=
        cc * uarray[IDX(i, j, nx_loc)] +
        cx * (uarray[IDX(i - 1, j, nx_loc)] + uarray[IDX(i + 1, j, nx_loc)]) +
        cy * (uarray[IDX(i, j - 1, nx_loc)] + Narray[i]);
    }
  }

  // Return success
  return 0;
}
