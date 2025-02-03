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

  // Start exchange
  int flag = udata->start_exchange(u);
  if (check_flag(&flag, "SendData", 1)) { return -1; }

  // Shortcuts to local number of nodes
  const sunindextype nx_loc = udata->nx_loc;
  const sunindextype ny_loc = udata->ny_loc;

  // Determine iteration range excluding the overall domain boundary
  const sunindextype istart = (udata->HaveNbrW) ? 0 : 1;
  const sunindextype iend   = (udata->HaveNbrE) ? nx_loc : nx_loc - 1;
  const sunindextype jstart = (udata->HaveNbrS) ? 0 : 1;
  const sunindextype jend   = (udata->HaveNbrN) ? ny_loc : ny_loc - 1;

  // Constants for computing diffusion term
  const sunrealtype cx = udata->kx / (udata->dx * udata->dx);
  const sunrealtype cy = udata->ky / (udata->dy * udata->dy);
  const sunrealtype cc = -TWO * (cx + cy);

  // Access data arrays
  sunrealtype* uarray = N_VGetArrayPointer(u);
  if (check_flag((void*)uarray, "N_VGetArrayPointer", 0)) { return -1; }

  sunrealtype* farray = N_VGetArrayPointer(f);
  if (check_flag((void*)farray, "N_VGetArrayPointer", 0)) { return -1; }

  // Initialize rhs vector to zero (handles boundary conditions)
  N_VConst(ZERO, f);

  // Iterate over subdomain interior and add rhs diffusion term
  for (sunindextype j = 1; j < ny_loc - 1; j++)
  {
    const sunrealtype Dy_s = Diffusion_Coeff_Y((udata->js+j) * udata->dy, udata)
                             / (udata->dy * udata->dy);
    const sunrealtype Dy_n = Diffusion_Coeff_Y((udata->js+j+1) * udata->dy, udata)
                             / (udata->dy * udata->dy);

    for (sunindextype i = 1; i < nx_loc - 1; i++)
    {
      const sunrealtype Dx_w = Diffusion_Coeff_X((udata->is+i) * udata->dx, udata)
                               / (udata->dx * udata->dx);
      const sunrealtype Dx_e = Diffusion_Coeff_X((udata->is+i+1) * udata->dx, udata)
                               / (udata->dx * udata->dx);

      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * uarray[IDX(i - 1, j, nx_loc)] +
        Dx_e * uarray[IDX(i + 1, j, nx_loc)] +
        Dy_s * uarray[IDX(i, j - 1, nx_loc)] +
        Dy_n * uarray[IDX(i, j + 1, nx_loc)];
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
    sunindextype i = 0;
    const sunrealtype Dx_w = Diffusion_Coeff_X((udata->is+i) * udata->dx, udata)
                             / (udata->dx * udata->dx);
    const sunrealtype Dx_e = Diffusion_Coeff_X((udata->is+i+1) * udata->dx, udata)
                             / (udata->dx * udata->dx);
    if (udata->HaveNbrS) // South-West corner
    {
      sunindextype j = 0;
      const sunrealtype Dy_s = Diffusion_Coeff_Y((udata->js+j) * udata->dy, udata)
                               / (udata->dy * udata->dy);
      const sunrealtype Dy_n = Diffusion_Coeff_Y((udata->js+j+1) * udata->dy, udata)
                               / (udata->dy * udata->dy);
      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * Warray[j] +
        Dx_e * uarray[IDX(i + 1, j, nx_loc)] +
        Dy_s * Sarray[i] +
        Dy_n * uarray[IDX(i, j + 1, nx_loc)];
    }

    for (sunindextype j = 1; j < ny_loc - 1; j++)
    {
      const sunrealtype Dy_s = Diffusion_Coeff_Y((udata->js+j) * udata->dy, udata)
                               / (udata->dy * udata->dy);
      const sunrealtype Dy_n = Diffusion_Coeff_Y((udata->js+j+1) * udata->dy, udata)
                               / (udata->dy * udata->dy);
      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * Warray[j] +
        Dx_e * uarray[IDX(i + 1, j, nx_loc)] +
        Dy_s * uarray[IDX(i, j - 1, nx_loc)] +
        Dy_n * uarray[IDX(i, j + 1, nx_loc)];
    }

    if (udata->HaveNbrN) // North-West corner
    {
      sunindextype j = ny_loc - 1;
      const sunrealtype Dy_s = Diffusion_Coeff_Y((udata->js+j) * udata->dy, udata)
                               / (udata->dy * udata->dy);
      const sunrealtype Dy_n = Diffusion_Coeff_Y((udata->js+j+1) * udata->dy, udata)
                               / (udata->dy * udata->dy);
      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * Warray[j] +
        Dx_e * uarray[IDX(i + 1, j, nx_loc)] +
        Dy_s * uarray[IDX(i, j - 1, nx_loc)] +
        Dy_n * Narray[i];
    }
  }

  // East face (updates south-east and north-east corners if necessary)
  if (udata->HaveNbrE)
  {
    sunindextype i = nx_loc - 1;
    const sunrealtype Dx_w = Diffusion_Coeff_X((udata->is+i) * udata->dx, udata)
                             / (udata->dx * udata->dx);
    const sunrealtype Dx_e = Diffusion_Coeff_X((udata->is+i+1) * udata->dx, udata)
                             / (udata->dx * udata->dx);
    if (udata->HaveNbrS) // South-East corner
    {
      sunindextype j = 0;
      const sunrealtype Dy_s = Diffusion_Coeff_Y((udata->js+j) * udata->dy, udata)
                               / (udata->dy * udata->dy);
      const sunrealtype Dy_n = Diffusion_Coeff_Y((udata->js+j+1) * udata->dy, udata)
                               / (udata->dy * udata->dy);
      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * uarray[IDX(i - 1, j, nx_loc)] +
        Dx_e * Earray[j] +
        Dy_s * Sarray[i] +
        Dy_n * uarray[IDX(i, j + 1, nx_loc)];
    }

    for (sunindextype j = 1; j < ny_loc - 1; j++)
    {
      const sunrealtype Dy_s = Diffusion_Coeff_Y((udata->js+j) * udata->dy, udata)
                               / (udata->dy * udata->dy);
      const sunrealtype Dy_n = Diffusion_Coeff_Y((udata->js+j+1) * udata->dy, udata)
                               / (udata->dy * udata->dy);
      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * uarray[IDX(i - 1, j, nx_loc)] +
        Dx_e * Earray[j] +
        Dy_s * uarray[IDX(i, j - 1, nx_loc)] +
        Dy_n * uarray[IDX(i, j + 1, nx_loc)];
    }

    if (udata->HaveNbrN) // North-East corner
    {
      sunindextype j = ny_loc - 1;
      const sunrealtype Dy_s = Diffusion_Coeff_Y((udata->js+j) * udata->dy, udata)
                               / (udata->dy * udata->dy);
      const sunrealtype Dy_n = Diffusion_Coeff_Y((udata->js+j+1) * udata->dy, udata)
                               / (udata->dy * udata->dy);
      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * uarray[IDX(i - 1, j, nx_loc)] +
        Dx_e * Earray[j] +
        Dy_s * uarray[IDX(i, j - 1, nx_loc)] +
        Dy_n * Narray[i];
    }
  }

  // South face (excludes corners)
  if (udata->HaveNbrS)
  {
    sunindextype j = 0;
    const sunrealtype Dy_s = Diffusion_Coeff_Y((udata->js+j) * udata->dy, udata)
                             / (udata->dy * udata->dy);
    const sunrealtype Dy_n = Diffusion_Coeff_Y((udata->js+j+1) * udata->dy, udata)
                             / (udata->dy * udata->dy);
    for (sunindextype i = 1; i < nx_loc - 1; i++)
    {
      const sunrealtype Dx_w = Diffusion_Coeff_X((udata->is+i) * udata->dx, udata)
                               / (udata->dx * udata->dx);
      const sunrealtype Dx_e = Diffusion_Coeff_X((udata->is+i+1) * udata->dx, udata)
                               / (udata->dx * udata->dx);
      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * uarray[IDX(i - 1, j, nx_loc)] +
        Dx_e * uarray[IDX(i + 1, j, nx_loc)] +
        Dy_s * Sarray[i] +
        Dy_n * uarray[IDX(i, j + 1, nx_loc)];
    }
  }

  // North face (excludes corners)
  if (udata->HaveNbrN)
  {
    sunindextype j = udata->ny_loc - 1;
    const sunrealtype Dy_s = Diffusion_Coeff_Y((udata->js+j) * udata->dy, udata)
                             / (udata->dy * udata->dy);
    const sunrealtype Dy_n = Diffusion_Coeff_Y((udata->js+j+1) * udata->dy, udata)
                             / (udata->dy * udata->dy);
    for (sunindextype i = 1; i < nx_loc - 1; i++)
    {
      const sunrealtype Dx_w = Diffusion_Coeff_X((udata->is+i) * udata->dx, udata)
                               / (udata->dx * udata->dx);
      const sunrealtype Dx_e = Diffusion_Coeff_X((udata->is+i+1) * udata->dx, udata)
                               / (udata->dx * udata->dx);
      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * uarray[IDX(i - 1, j, nx_loc)] +
        Dx_e * uarray[IDX(i + 1, j, nx_loc)] +
        Dy_s * uarray[IDX(i, j - 1, nx_loc)] +
        Dy_n * Narray[i];
    }
  }

  // Return success
  return 0;
}
