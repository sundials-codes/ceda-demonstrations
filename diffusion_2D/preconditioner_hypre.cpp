/* -----------------------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ UMBC
 * ---------------------------------------------------------------------------*/

#include "diffusion_2D.hpp"

// Utility routines
int SetupHypre(UserData& udata);
static int Jac(UserData& udata);
static int ScaleAddI(UserData& udata, sunrealtype gamma);

// Preconditioner setup routine
int PSetup(sunrealtype t, N_Vector u, N_Vector f, sunbooleantype jok,
           sunbooleantype* jcurPtr, sunrealtype gamma, void* user_data)
{
  int flag;

  // Access problem data
  UserData* udata = (UserData*)user_data;

  // --------------
  // Fill Jacobian
  // --------------

  flag = Jac(*udata);
  if (flag != 0)
  {
    cerr << "Error in Jac = " << flag << endl;
    return -1;
  }

  flag = HYPRE_StructMatrixAssemble(udata->Jmatrix);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructMatrixAssemble = " << flag << endl;
    return -1;
  }

  // Fill matrix A = I - gamma * J
  flag = ScaleAddI(*udata, gamma);
  if (flag != 0) { return -1; }

  // Assemble matrix
  flag = HYPRE_StructMatrixAssemble(udata->Amatrix);
  if (flag != 0) { return -1; }

  // Indicate that the jacobian is current
  *jcurPtr = SUNTRUE;

  // -----------
  // Setup PFMG
  // -----------

  // Set rhs/solution vectors as all zero for now
  flag = HYPRE_StructVectorSetConstantValues(udata->bvec, ZERO);
  if (flag != 0) { return -1; }

  flag = HYPRE_StructVectorAssemble(udata->bvec);
  if (flag != 0) { return -1; }

  flag = HYPRE_StructVectorSetConstantValues(udata->xvec, ZERO);
  if (flag != 0) { return -1; }

  flag = HYPRE_StructVectorAssemble(udata->xvec);
  if (flag != 0) { return -1; }

  // Free the existing preconditioner if necessary
  if (udata->precond) { HYPRE_StructPFMGDestroy(udata->precond); }

  // Create the new preconditioner
  flag = HYPRE_StructPFMGCreate(udata->comm_c, &(udata->precond));
  if (flag != 0) { return -1; }

  // Signal that the initial guess is zero
  flag = HYPRE_StructPFMGSetZeroGuess(udata->precond);
  if (flag != 0) { return -1; }

  // tol <= 0.0 means do the max number of iterations
  flag = HYPRE_StructPFMGSetTol(udata->precond, ZERO);
  if (flag != 0) { return -1; }

  // Use one v-cycle
  flag = HYPRE_StructPFMGSetMaxIter(udata->precond, 1);
  if (flag != 0) { return -1; }

  // Use non-Galerkin coarse grid operator
  flag = HYPRE_StructPFMGSetRAPType(udata->precond, 1);
  if (flag != 0) { return -1; }

  // Set the relaxation type
  flag = HYPRE_StructPFMGSetRelaxType(udata->precond, udata->pfmg_relax);
  if (flag != 0) { return -1; }

  // Set the number of pre and post relaxation sweeps
  flag = HYPRE_StructPFMGSetNumPreRelax(udata->precond, udata->pfmg_nrelax);
  if (flag != 0) { return -1; }

  flag = HYPRE_StructPFMGSetNumPostRelax(udata->precond, udata->pfmg_nrelax);
  if (flag != 0) { return -1; }

  // Set up the solver
  flag = HYPRE_StructPFMGSetup(udata->precond, udata->Amatrix, udata->bvec,
                               udata->xvec);
  if (flag != 0) { return -1; }

  // Return success
  return 0;
}

// Preconditioner solve routine for Pz = r
int PSolve(sunrealtype t, N_Vector u, N_Vector f, N_Vector r, N_Vector z,
           sunrealtype gamma, sunrealtype delta, int lr, void* user_data)
{
  int flag;

  // Access user_data structure
  UserData* udata = (UserData*)user_data;

  // Insert rhs N_Vector entries into HYPRE vector b and assemble
  flag = HYPRE_StructVectorSetBoxValues(udata->bvec, udata->ilower,
                                        udata->iupper, N_VGetArrayPointer(r));
  if (flag != 0) { return -1; }

  flag = HYPRE_StructVectorAssemble(udata->bvec);
  if (flag != 0) { return -1; }

  // Set the initial guess into HYPRE vector x and assemble
  flag = HYPRE_StructVectorSetConstantValues(udata->xvec, ZERO);
  if (flag != 0) { return -1; }

  flag = HYPRE_StructVectorAssemble(udata->xvec);
  if (flag != 0) { return -1; }

  // // Print matrix and RHS to disk
  // cout << "Printing HYPRE matrix, RHS, and solution to disk (gamma = " << gamma << ")" << endl;
  // flag = HYPRE_StructMatrixPrint("hypre_matrix", udata->Amatrix, 0);
  // if (flag != 0) { return -1; }
  // flag = HYPRE_StructVectorPrint("hypre_rhs", udata->bvec, 0);
  // if (flag != 0) { return -1; }

  // Update the preconditioner solver tolerance
  flag = HYPRE_StructPFMGSetTol(udata->precond, delta);
  if (flag != 0) { return -1; }

  // Solve the linear system
  flag = HYPRE_StructPFMGSolve(udata->precond, udata->Amatrix, udata->bvec,
                               udata->xvec);

  // // Print solution to disk
  // flag = HYPRE_StructVectorPrint("hypre_sol", udata->xvec, 0);
  // if (flag != 0) { return -1; }

  // If a convergence error occurred, clear the error and continue. For any
  // other error return with a recoverable error.
  if (flag == HYPRE_ERROR_CONV) { HYPRE_ClearError(HYPRE_ERROR_CONV); }
  else if (flag != 0) { return 1; }

  // Update precond statistics
  HYPRE_Int itmp;
  flag = HYPRE_StructPFMGGetNumIterations(udata->precond, &itmp);
  if (flag != 0) { return -1; }

  udata->pfmg_its += itmp;

  // Extract solution values
  flag = HYPRE_StructVectorGetBoxValues(udata->xvec, udata->ilower,
                                        udata->iupper, N_VGetArrayPointer(z));
  if (flag != 0) { return -1; }

  // Return success
  return 0;
}

// -----------------------------------------------------------------------------
// Preconditioner helper functions
// -----------------------------------------------------------------------------

// Create hypre objects
int SetupHypre(UserData& udata)
{
  int flag, result;

  // Check if the grid or stencil have been created
  if ((udata.grid != NULL || udata.stencil != NULL))
  {
    cerr << "SetupHypre error: grid or stencil already exists" << endl;
    return -1;
  }

  // Check for valid 2D Cartesian MPI communicator
  flag = MPI_Topo_test(udata.comm_c, &result);
  if ((flag != MPI_SUCCESS) || (result != MPI_CART))
  {
    cerr << "SetupHypre error: communicator is not Cartesian" << endl;
    return -1;
  }

  flag = MPI_Cartdim_get(udata.comm_c, &result);
  if ((flag != MPI_SUCCESS) || (result != 2))
  {
    cerr << "SetupHypre error: communicator is not 2D" << endl;
    return -1;
  }

  // -----
  // Grid
  // -----

  // Create 2D grid object
  flag = HYPRE_StructGridCreate(udata.comm_c, 2, &(udata.grid));
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructGridCreate = " << flag << endl;
    return -1;
  }

  // Set grid extents (lower left and upper right corners)
  udata.ilower[0] = udata.is;
  udata.ilower[1] = udata.js;

  udata.iupper[0] = udata.ie;
  udata.iupper[1] = udata.je;

  flag = HYPRE_StructGridSetExtents(udata.grid, udata.ilower, udata.iupper);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructGridSetExtents = " << flag << endl;
    return -1;
  }

  // Set periodicity
  udata.periodic[0] = udata.nx;
  udata.periodic[1] = udata.ny;

  flag = HYPRE_StructGridSetPeriodic(udata.grid, udata.periodic);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructGridSetPeriodic = " << flag << endl;
    return -1;
  }

  // Assemble the grid
  flag = HYPRE_StructGridAssemble(udata.grid);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructGridAssemble = " << flag << endl;
    return -1;
  }

  // --------
  // Stencil
  // --------

  // Create the 2D 5 point stencil object
  flag = HYPRE_StructStencilCreate(2, 5, &(udata.stencil));
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructStencilCreate = " << flag << endl;
    return -1;
  }

  // Set the stencil entries (center, left, right, bottom, top)
  HYPRE_Int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

  for (int entry = 0; entry < 5; entry++)
  {
    flag = HYPRE_StructStencilSetElement(udata.stencil, entry, offsets[entry]);
    if (flag != 0)
    {
      cerr << "Error in HYPRE_StructStencilSetElement = " << flag << endl;
      return -1;
    }
  }

  // -----------
  // Work array
  // -----------

  udata.nwork = 5 * udata.nodes_loc;
  udata.work  = NULL;
  udata.work  = new HYPRE_Real[udata.nwork];
  if (udata.work == NULL)
  {
    cerr << "Error: unable to allocate work array" << endl;
    return -1;
  }

  // ---------
  // x vector
  // ---------

  flag = HYPRE_StructVectorCreate(udata.comm_c, udata.grid, &(udata.xvec));
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructVectorCreate (x) = " << flag << endl;
    return -1;
  }

  flag = HYPRE_StructVectorInitialize(udata.xvec);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructVectorInitialize (x) = " << flag << endl;
    return -1;
  }

  // ---------
  // b vector
  // ---------

  flag = HYPRE_StructVectorCreate(udata.comm_c, udata.grid, &(udata.bvec));
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructVectorCreate (b) = " << flag << endl;
    return -1;
  }

  flag = HYPRE_StructVectorInitialize(udata.bvec);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructVectorInitialize (b) = " << flag << endl;
    return -1;
  }

  // ---------
  // J matrix
  // ---------

  flag = HYPRE_StructMatrixCreate(udata.comm_c, udata.grid, udata.stencil,
                                  &(udata.Jmatrix));
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructMatrixCreate (J) = " << flag << endl;
    return -1;
  }

  flag = HYPRE_StructMatrixInitialize(udata.Jmatrix);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructMatrixInitialize (A) = " << flag << endl;
    return -1;
  }

  // ---------
  // A matrix
  // ---------

  flag = HYPRE_StructMatrixCreate(udata.comm_c, udata.grid, udata.stencil,
                                  &(udata.Amatrix));
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructMatrixCreate (A) = " << flag << endl;
    return -1;
  }

  flag = HYPRE_StructMatrixInitialize(udata.Amatrix);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructMatrixInitialize (A) = " << flag << endl;
    return -1;
  }

  // --------------------
  // PFMG preconditioner
  // --------------------

  // Note a new PFMG preconditioner must be created and attached each time the
  // linear system is updated. As such it is constructed in the preconditioner
  // setup function (if enabled).
  udata.precond = NULL;

  return 0;
}

// Jac function to compute the ODE RHS function Jacobian, (df/dy)(t,y).
static int Jac(UserData& udata)
{
  // Shortcuts to hypre matrix and grid extents, work array, etc.
  HYPRE_StructMatrix Jmatrix = udata.Jmatrix;

  HYPRE_Int ilower[2];
  HYPRE_Int iupper[2];

  ilower[0] = udata.ilower[0];
  ilower[1] = udata.ilower[1];

  iupper[0] = udata.iupper[0];
  iupper[1] = udata.iupper[1];

  HYPRE_Int nwork  = udata.nwork;
  HYPRE_Real* work = udata.work;

  sunindextype nx_loc = udata.nx_loc;
  sunindextype ny_loc = udata.ny_loc;

  // Matrix stencil: center, left, right, bottom, top
  HYPRE_Int entries[5] = {0, 1, 2, 3, 4};
  HYPRE_Int entry[1];

  // Grid extents for setting boundary entries
  HYPRE_Int bc_ilower[2];
  HYPRE_Int bc_iupper[2];

  // Loop counters
  HYPRE_Int idx, ix, iy;

  // hypre return flag
  int flag;

  // ----------
  // Compute J
  // ----------

  // Only do work if the box is non-zero in size
  if ((ilower[0] <= iupper[0]) && (ilower[1] <= iupper[1]))
  {
    // --------------------------------
    // Set matrix values for all nodes
    // --------------------------------

    // Set the matrix interior entries (center, left, right, bottom, top)
    idx = 0;
    for (iy = 0; iy < ny_loc; iy++)
    {
      const sunrealtype Dy_s = Diffusion_Coeff_Y((udata.js + iy) * udata.dy,
                                                 &udata) /
                               (udata.dy * udata.dy);
      const sunrealtype Dy_n = Diffusion_Coeff_Y((udata.js + iy + 1) * udata.dy,
                                                 &udata) /
                               (udata.dy * udata.dy);
      for (ix = 0; ix < nx_loc; ix++)
      {
        const sunrealtype Dx_w = Diffusion_Coeff_X((udata.is + ix) * udata.dx,
                                                   &udata) /
                                 (udata.dx * udata.dx);
        const sunrealtype Dx_e =
          Diffusion_Coeff_X((udata.is + ix + 1) * udata.dx, &udata) /
          (udata.dx * udata.dx);
        work[idx]     = -((Dx_w + Dx_e) + (Dy_s + Dy_n));
        work[idx + 1] = Dx_w;
        work[idx + 2] = Dx_e;
        work[idx + 3] = Dy_s;
        work[idx + 4] = Dy_n;
        idx += 5;
      }
    }

    // Insert entries into the matrix
    flag = HYPRE_StructMatrixSetBoxValues(Jmatrix, ilower, iupper, 5, entries,
                                          work);
    if (flag != 0)
    {
      cerr << "Error in HYPRE_StructMatrixSetBoxValues (interior) = " << flag
           << endl;
      return -1;
    }
  }

  // The matrix is assembled matrix in hypre setup

  // Return success
  return 0;
}

// Fill A = I - gamma * J matrix
static int ScaleAddI(UserData& udata, sunrealtype gamma)
{
  int flag;

  // Variable shortcuts
  HYPRE_Int ilower[2];
  HYPRE_Int iupper[2];

  ilower[0] = udata.ilower[0];
  ilower[1] = udata.ilower[1];

  iupper[0] = udata.iupper[0];
  iupper[1] = udata.iupper[1];

  HYPRE_Int nwork  = udata.nwork;
  HYPRE_Real* work = udata.work;

  // Matrix stencil: center, left, right, bottom, top
  HYPRE_Int entries[5] = {0, 1, 2, 3, 4};

  // Copy all matrix values into work array from J
  flag = HYPRE_StructMatrixGetBoxValues(udata.Jmatrix, ilower, iupper, 5,
                                        entries, work);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructMatrixGetBoxValues = " << flag << endl;
    return (flag);
  }

  // Scale work array by -gamma
  for (HYPRE_Int i = 0; i < nwork; i++) { work[i] *= -gamma; }

  // Insert scaled values into A
  flag = HYPRE_StructMatrixSetBoxValues(udata.Amatrix, ilower, iupper, 5,
                                        entries, work);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructMatrixSetBoxValues = " << flag << endl;
    return (flag);
  }

  // Set first 1/5 of work array to 1.0
  for (HYPRE_Int i = 0; i < nwork / 5; i++) { work[i] = ONE; }

  // Add values to the diagonal of A
  HYPRE_Int entry[1] = {0};
  flag = HYPRE_StructMatrixAddToBoxValues(udata.Amatrix, ilower, iupper, 1,
                                          entry, work);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructMatrixAddToBoxValues = " << flag << endl;
    return (flag);
  }

  // Return success
  return 0;
}
