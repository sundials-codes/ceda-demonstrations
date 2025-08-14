/* -----------------------------------------------------------------------------
 * Programmer(s): David J. Gardner @ LLNL
 *                Major revisions by Daniel R. Reynolds @ SMU
 * ---------------------------------------------------------------------------*/

#include "diffusion_2D.hpp"

// Preconditioner setup routine
int PSetup(sunrealtype t, N_Vector u, N_Vector f, sunbooleantype jok,
           sunbooleantype* jcurPtr, sunrealtype gamma, void* user_data)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  SUNDIALS_CXX_MARK_FUNCTION(udata->prof);

  // Iterate over subdomain and set all entries of udata->diag to the inverse of
  // the diagonal of the Jacobian
  sunrealtype* darray = N_VGetArrayPointer(udata->diag);
  if (check_flag((void*)darray, "N_VGetArrayPointer", 0)) { return -1; }
  for (sunindextype j = 0; j < udata->ny_loc; j++)
  {
    const sunrealtype Dy_s = Diffusion_Coeff_Y((udata->js + j) * udata->dy,
                                               udata) /
                             (udata->dy * udata->dy);
    const sunrealtype Dy_n = Diffusion_Coeff_Y((udata->js + j + 1) * udata->dy,
                                               udata) /
                             (udata->dy * udata->dy);

    for (sunindextype i = 0; i < udata->nx_loc; i++)
    {
      const sunrealtype Dx_w = Diffusion_Coeff_X((udata->is + i) * udata->dx,
                                                 udata) /
                               (udata->dx * udata->dx);
      const sunrealtype Dx_e =
        Diffusion_Coeff_X((udata->is + i + 1) * udata->dx, udata) /
        (udata->dx * udata->dx);

      const sunrealtype diag           = -((Dx_w + Dx_e) + (Dy_s + Dy_n));
      darray[IDX(i, j, udata->nx_loc)] = ONE / (ONE - gamma * diag);
    }
  }

  // Return success
  return 0;
}

// Preconditioner solve routine for Pz = r
int PSolve(sunrealtype t, N_Vector u, N_Vector f, N_Vector r, N_Vector z,
           sunrealtype gamma, sunrealtype delta, int lr, void* user_data)
{
  // Access user_data structure
  UserData* udata = (UserData*)user_data;

  SUNDIALS_CXX_MARK_FUNCTION(udata->prof);

  // Perform Jacobi iteration
  N_VProd(udata->diag, r, z);

  // Return success
  return 0;
}
