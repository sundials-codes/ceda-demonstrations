/* -----------------------------------------------------------------------------
 * Programmer(s): David J. Gardner @ LLNL
 *                Major revisions by Daniel R. Reynolds @ UMBC
 * ---------------------------------------------------------------------------*/

#include "diffusion_2D.hpp"

// Diffusion function
int laplacian(sunrealtype t, N_Vector u, N_Vector f, UserData* udata)
{
  SUNDIALS_CXX_MARK_FUNCTION(udata->prof);

  // Start exchange
  int flag = udata->start_exchange(u);
  if (check_flag(&flag, "SendData", 1)) { return -1; }

  // Shortcuts variables
  const sunindextype nx_loc = udata->nx_loc;
  const sunindextype ny_loc = udata->ny_loc;
  const sunrealtype dx      = udata->dx;
  const sunrealtype dy      = udata->dy;

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
    const sunrealtype ylo  = udata->yl + (udata->js + j - 0.5) * dy;
    const sunrealtype yhi  = udata->yl + (udata->js + j + 0.5) * dy;
    const sunrealtype Dy_s = Diffusion_Coeff_Y(ylo, udata) / (dy * dy);
    const sunrealtype Dy_n = Diffusion_Coeff_Y(yhi, udata) / (dy * dy);

    for (sunindextype i = 1; i < nx_loc - 1; i++)
    {
      const sunrealtype xlo  = udata->xl + (udata->is + i - 0.5) * dx;
      const sunrealtype xhi  = udata->xl + (udata->is + i + 0.5) * dx;
      const sunrealtype Dx_w = Diffusion_Coeff_X(xlo, udata) / (dx * dx);
      const sunrealtype Dx_e = Diffusion_Coeff_X(xhi, udata) / (dx * dx);

      farray[IDX(i, j, nx_loc)] += -((Dx_w + Dx_e) + (Dy_s + Dy_n)) *
                                     uarray[IDX(i, j, nx_loc)] +
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
    sunindextype i         = 0;
    const sunrealtype xlo  = udata->xl + (udata->is + i - 0.5) * dx;
    const sunrealtype xhi  = udata->xl + (udata->is + i + 0.5) * dx;
    const sunrealtype Dx_w = Diffusion_Coeff_X(xlo, udata) / (dx * dx);
    const sunrealtype Dx_e = Diffusion_Coeff_X(xhi, udata) / (dx * dx);
    if (udata->HaveNbrS) // South-West corner
    {
      sunindextype j         = 0;
      const sunrealtype ylo  = udata->yl + (udata->js + j - 0.5) * dy;
      const sunrealtype yhi  = udata->yl + (udata->js + j + 0.5) * dy;
      const sunrealtype Dy_s = Diffusion_Coeff_Y(ylo, udata) / (dy * dy);
      const sunrealtype Dy_n = Diffusion_Coeff_Y(yhi, udata) / (dy * dy);
      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * Warray[j] + Dx_e * uarray[IDX(i + 1, j, nx_loc)] +
        Dy_s * Sarray[i] + Dy_n * uarray[IDX(i, j + 1, nx_loc)];
    }

    for (sunindextype j = 1; j < ny_loc - 1; j++)
    {
      const sunrealtype ylo  = udata->yl + (udata->js + j - 0.5) * dy;
      const sunrealtype yhi  = udata->yl + (udata->js + j + 0.5) * dy;
      const sunrealtype Dy_s = Diffusion_Coeff_Y(ylo, udata) / (dy * dy);
      const sunrealtype Dy_n = Diffusion_Coeff_Y(yhi, udata) / (dy * dy);
      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * Warray[j] + Dx_e * uarray[IDX(i + 1, j, nx_loc)] +
        Dy_s * uarray[IDX(i, j - 1, nx_loc)] +
        Dy_n * uarray[IDX(i, j + 1, nx_loc)];
    }

    if (udata->HaveNbrN) // North-West corner
    {
      sunindextype j         = ny_loc - 1;
      const sunrealtype ylo  = udata->yl + (udata->js + j - 0.5) * dy;
      const sunrealtype yhi  = udata->yl + (udata->js + j + 0.5) * dy;
      const sunrealtype Dy_s = Diffusion_Coeff_Y(ylo, udata) / (dy * dy);
      const sunrealtype Dy_n = Diffusion_Coeff_Y(yhi, udata) / (dy * dy);
      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * Warray[j] + Dx_e * uarray[IDX(i + 1, j, nx_loc)] +
        Dy_s * uarray[IDX(i, j - 1, nx_loc)] + Dy_n * Narray[i];
    }
  }

  // East face (updates south-east and north-east corners if necessary)
  if (udata->HaveNbrE)
  {
    sunindextype i         = nx_loc - 1;
    const sunrealtype xlo  = udata->xl + (udata->is + i - 0.5) * dx;
    const sunrealtype xhi  = udata->xl + (udata->is + i + 0.5) * dx;
    const sunrealtype Dx_w = Diffusion_Coeff_X(xlo, udata) / (dx * dx);
    const sunrealtype Dx_e = Diffusion_Coeff_X(xhi, udata) / (dx * dx);
    if (udata->HaveNbrS) // South-East corner
    {
      sunindextype j         = 0;
      const sunrealtype ylo  = udata->yl + (udata->js + j - 0.5) * dy;
      const sunrealtype yhi  = udata->yl + (udata->js + j + 0.5) * dy;
      const sunrealtype Dy_s = Diffusion_Coeff_Y(ylo, udata) / (dy * dy);
      const sunrealtype Dy_n = Diffusion_Coeff_Y(yhi, udata) / (dy * dy);
      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * uarray[IDX(i - 1, j, nx_loc)] + Dx_e * Earray[j] +
        Dy_s * Sarray[i] + Dy_n * uarray[IDX(i, j + 1, nx_loc)];
    }

    for (sunindextype j = 1; j < ny_loc - 1; j++)
    {
      const sunrealtype ylo  = udata->yl + (udata->js + j - 0.5) * dy;
      const sunrealtype yhi  = udata->yl + (udata->js + j + 0.5) * dy;
      const sunrealtype Dy_s = Diffusion_Coeff_Y(ylo, udata) / (dy * dy);
      const sunrealtype Dy_n = Diffusion_Coeff_Y(yhi, udata) / (dy * dy);
      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * uarray[IDX(i - 1, j, nx_loc)] + Dx_e * Earray[j] +
        Dy_s * uarray[IDX(i, j - 1, nx_loc)] +
        Dy_n * uarray[IDX(i, j + 1, nx_loc)];
    }

    if (udata->HaveNbrN) // North-East corner
    {
      sunindextype j         = ny_loc - 1;
      const sunrealtype ylo  = udata->yl + (udata->js + j - 0.5) * dy;
      const sunrealtype yhi  = udata->yl + (udata->js + j + 0.5) * dy;
      const sunrealtype Dy_s = Diffusion_Coeff_Y(ylo, udata) / (dy * dy);
      const sunrealtype Dy_n = Diffusion_Coeff_Y(yhi, udata) / (dy * dy);
      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * uarray[IDX(i - 1, j, nx_loc)] + Dx_e * Earray[j] +
        Dy_s * uarray[IDX(i, j - 1, nx_loc)] + Dy_n * Narray[i];
    }
  }

  // South face (excludes corners)
  if (udata->HaveNbrS)
  {
    sunindextype j         = 0;
    const sunrealtype ylo  = udata->yl + (udata->js + j - 0.5) * dy;
    const sunrealtype yhi  = udata->yl + (udata->js + j + 0.5) * dy;
    const sunrealtype Dy_s = Diffusion_Coeff_Y(ylo, udata) / (dy * dy);
    const sunrealtype Dy_n = Diffusion_Coeff_Y(yhi, udata) / (dy * dy);
    for (sunindextype i = 1; i < nx_loc - 1; i++)
    {
      const sunrealtype xlo  = udata->xl + (udata->is + i - 0.5) * dx;
      const sunrealtype xhi  = udata->xl + (udata->is + i + 0.5) * dx;
      const sunrealtype Dx_w = Diffusion_Coeff_X(xlo, udata) / (dx * dx);
      const sunrealtype Dx_e = Diffusion_Coeff_X(xhi, udata) / (dx * dx);
      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * uarray[IDX(i - 1, j, nx_loc)] +
        Dx_e * uarray[IDX(i + 1, j, nx_loc)] + Dy_s * Sarray[i] +
        Dy_n * uarray[IDX(i, j + 1, nx_loc)];
    }
  }

  // North face (excludes corners)
  if (udata->HaveNbrN)
  {
    sunindextype j         = udata->ny_loc - 1;
    const sunrealtype ylo  = udata->yl + (udata->js + j - 0.5) * dy;
    const sunrealtype yhi  = udata->yl + (udata->js + j + 0.5) * dy;
    const sunrealtype Dy_s = Diffusion_Coeff_Y(ylo, udata) / (dy * dy);
    const sunrealtype Dy_n = Diffusion_Coeff_Y(yhi, udata) / (dy * dy);
    for (sunindextype i = 1; i < nx_loc - 1; i++)
    {
      const sunrealtype xlo  = udata->xl + (udata->is + i - 0.5) * dx;
      const sunrealtype xhi  = udata->xl + (udata->is + i + 0.5) * dx;
      const sunrealtype Dx_w = Diffusion_Coeff_X(xlo, udata) / (dx * dx);
      const sunrealtype Dx_e = Diffusion_Coeff_X(xhi, udata) / (dx * dx);
      farray[IDX(i, j, nx_loc)] +=
        -((Dx_w + Dx_e) + (Dy_s + Dy_n)) * uarray[IDX(i, j, nx_loc)] +
        Dx_w * uarray[IDX(i - 1, j, nx_loc)] +
        Dx_e * uarray[IDX(i + 1, j, nx_loc)] +
        Dy_s * uarray[IDX(i, j - 1, nx_loc)] + Dy_n * Narray[i];
    }
  }

  // Return success
  return 0;
}
