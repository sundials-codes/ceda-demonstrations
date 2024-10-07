/* -----------------------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ SMU
 * -----------------------------------------------------------------------------
 *
 * HYPRE-PFMG preconditiner for 2D diffusion benchmark problem
 * ---------------------------------------------------------------------------*/

#include "diffusion_2D.hpp"
#include "HYPRE_struct_ls.h"

// Utility routines
static int SetupHypre(UserData* udata);
static int Jac(UserData* udata);
static int ScaleAddI(UserData* udata, sunrealtype gamma);

// Preconditioner setup routine
static int PSetup(sunrealtype t, N_Vector u, N_Vector f, sunbooleantype jok,
                  sunbooleantype* jcurPtr, sunrealtype gamma, void* user_data)
{
  int flag;

  // Start timer
  double t1 = MPI_Wtime();

  // Access problem data
  UserData* udata = (UserData*)user_data;

  // --------------
  // Fill Jacobian
  // --------------

  flag = Jac(udata);
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
  flag = ScaleAddI(udata, gamma);
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

  // Use non-Galerkin corase grid operator
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

  // Stop timer
  double t2 = MPI_Wtime();

  // Update timer
  udata->psetuptime += t2 - t1;

  // Return success
  return 0;
}

// Preconditioner solve routine for Pz = r
static int PSolve(sunrealtype t, N_Vector u, N_Vector f, N_Vector r, N_Vector z,
                  sunrealtype gamma, sunrealtype delta, int lr, void* user_data)
{
  int flag;

  // Start timer
  double t1 = MPI_Wtime();

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

  // Solve the linear system
  flag = HYPRE_StructPFMGSolve(udata->precond, udata->Amatrix, udata->bvec,
                               udata->xvec);

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

  // Stop timer
  double t2 = MPI_Wtime();

  // Update timer
  udata->psolvetime += t2 - t1;

  // Return success
  return 0;
}

// -----------------------------------------------------------------------------
// Preconditioner helper functions
// -----------------------------------------------------------------------------

// Create hypre objects
static int SetupHypre(UserData* udata)
{
  int flag, result;

  // Check input
  if (udata == NULL) { return -1; }

  // Check if the grid or stencil have been created
  if ((udata->grid != NULL || udata->stencil != NULL))
  {
    cerr << "SetupHypre error: grid or stencil already exists" << endl;
    return -1;
  }

  // Check for valid 2D Cartesian MPI communicator
  flag = MPI_Topo_test(udata->comm_c, &result);
  if ((flag != MPI_SUCCESS) || (result != MPI_CART))
  {
    cerr << "SetupHypre error: communicator is not Cartesian" << endl;
    return -1;
  }

  flag = MPI_Cartdim_get(udata->comm_c, &result);
  if ((flag != MPI_SUCCESS) || (result != 2))
  {
    cerr << "SetupHypre error: communicator is not 2D" << endl;
    return -1;
  }

  // -----
  // Grid
  // -----

  // Create 2D grid object
  flag = HYPRE_StructGridCreate(udata->comm_c, 2, &(udata->grid));
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructGridCreate = " << flag << endl;
    FreeUserData(udata);
    return -1;
  }

  // Set grid extents (lower left and upper right corners)
  udata->ilower[0] = udata->is;
  udata->ilower[1] = udata->js;

  udata->iupper[0] = udata->ie;
  udata->iupper[1] = udata->je;

  flag = HYPRE_StructGridSetExtents(udata->grid, udata->ilower, udata->iupper);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructGridSetExtents = " << flag << endl;
    FreeUserData(udata);
    return -1;
  }

  // Assemble the grid
  flag = HYPRE_StructGridAssemble(udata->grid);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructGridAssemble = " << flag << endl;
    FreeUserData(udata);
    return -1;
  }

  // --------
  // Stencil
  // --------

  // Create the 2D 5 point stencil object
  flag = HYPRE_StructStencilCreate(2, 5, &(udata->stencil));
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructStencilCreate = " << flag << endl;
    FreeUserData(udata);
    return -1;
  }

  // Set the stencil entries (center, left, right, bottom, top)
  HYPRE_Int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

  for (int entry = 0; entry < 5; entry++)
  {
    flag = HYPRE_StructStencilSetElement(udata->stencil, entry, offsets[entry]);
    if (flag != 0)
    {
      cerr << "Error in HYPRE_StructStencilSetElement = " << flag << endl;
      FreeUserData(udata);
      return -1;
    }
  }

  // -----------
  // Work array
  // -----------

  udata->nwork = 5 * udata->nodes_loc;
  udata->work  = NULL;
  udata->work  = new HYPRE_Real[udata->nwork];
  if (udata->work == NULL)
  {
    cerr << "Error: unable to allocate work array" << endl;
    FreeUserData(udata);
    return -1;
  }

  // ---------
  // x vector
  // ---------

  flag = HYPRE_StructVectorCreate(udata->comm_c, udata->grid, &(udata->xvec));
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructVectorCreate (x) = " << flag << endl;
    FreeUserData(udata);
    return -1;
  }

  flag = HYPRE_StructVectorInitialize(udata->xvec);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructVectorInitialize (x) = " << flag << endl;
    FreeUserData(udata);
    return -1;
  }

  // ---------
  // b vector
  // ---------

  flag = HYPRE_StructVectorCreate(udata->comm_c, udata->grid, &(udata->bvec));
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructVectorCreate (b) = " << flag << endl;
    FreeUserData(udata);
    return -1;
  }

  flag = HYPRE_StructVectorInitialize(udata->bvec);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructVectorInitialize (b) = " << flag << endl;
    FreeUserData(udata);
    return -1;
  }

  if (udata->matvec)
  {
    // ---------
    // v vector
    // ---------

    flag = HYPRE_StructVectorCreate(udata->comm_c, udata->grid, &(udata->vvec));
    if (flag != 0)
    {
      cerr << "Error in HYPRE_StructVectorCreate (v) = " << flag << endl;
      FreeUserData(udata);
      return -1;
    }

    flag = HYPRE_StructVectorInitialize(udata->vvec);
    if (flag != 0)
    {
      cerr << "Error in HYPRE_StructVectorInitialize (v) = " << flag << endl;
      FreeUserData(udata);
      return -1;
    }

    // ----------
    // Jv vector
    // ----------

    flag = HYPRE_StructVectorCreate(udata->comm_c, udata->grid, &(udata->Jvvec));
    if (flag != 0)
    {
      cerr << "Error in HYPRE_StructVectorCreate (Jv) = " << flag << endl;
      FreeUserData(udata);
      return -1;
    }

    flag = HYPRE_StructVectorInitialize(udata->Jvvec);
    if (flag != 0)
    {
      cerr << "Error in HYPRE_StructVectorInitialize (Jv) = " << flag << endl;
      FreeUserData(udata);
      return -1;
    }
  }

  // ---------
  // J matrix
  // ---------

  flag = HYPRE_StructMatrixCreate(udata->comm_c, udata->grid, udata->stencil,
                                  &(udata->Jmatrix));
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructMatrixCreate (J) = " << flag << endl;
    FreeUserData(udata);
    return -1;
  }

  flag = HYPRE_StructMatrixInitialize(udata->Jmatrix);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructMatrixInitialize (A) = " << flag << endl;
    FreeUserData(udata);
    return -1;
  }

  // ---------
  // A matrix
  // ---------

  flag = HYPRE_StructMatrixCreate(udata->comm_c, udata->grid, udata->stencil,
                                  &(udata->Amatrix));
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructMatrixCreate (A) = " << flag << endl;
    FreeUserData(udata);
    return -1;
  }

  flag = HYPRE_StructMatrixInitialize(udata->Amatrix);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructMatrixInitialize (A) = " << flag << endl;
    FreeUserData(udata);
    return -1;
  }

  // --------------------
  // PFMG preconditioner
  // --------------------

  // Note a new PFMG preconditioner must be created and attached each time the
  // linear system is updated. As such it is constructed in the preconditioner
  // setup function (if enabled).
  udata->precond = NULL;

  return 0;
}

// Jac function to compute the ODE RHS function Jacobian, (df/dy)(t,y).
static int Jac(UserData* udata)
{
  // Shortcuts to hypre matrix and grid extents, work array, etc.
  HYPRE_StructMatrix Jmatrix = udata->Jmatrix;

  HYPRE_Int ilower[2];
  HYPRE_Int iupper[2];

  ilower[0] = udata->ilower[0];
  ilower[1] = udata->ilower[1];

  iupper[0] = udata->iupper[0];
  iupper[1] = udata->iupper[1];

  HYPRE_Int nwork  = udata->nwork;
  HYPRE_Real* work = udata->work;

  sunindextype nx_loc = udata->nx_loc;
  sunindextype ny_loc = udata->ny_loc;

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

  // Start timer
  double t1 = MPI_Wtime();

  // Only do work if the box is non-zero in size
  if ((ilower[0] <= iupper[0]) && (ilower[1] <= iupper[1]))
  {
    // Jacobian values
    sunrealtype cx = udata->kx / (udata->dx * udata->dx);
    sunrealtype cy = udata->ky / (udata->dy * udata->dy);
    sunrealtype cc = -TWO * (cx + cy);

    // --------------------------------
    // Set matrix values for all nodes
    // --------------------------------

    // Set the matrix interior entries (center, left, right, bottom, top)
    idx = 0;
    for (iy = 0; iy < ny_loc; iy++)
    {
      for (ix = 0; ix < nx_loc; ix++)
      {
        work[idx]     = cc;
        work[idx + 1] = cx;
        work[idx + 2] = cx;
        work[idx + 3] = cy;
        work[idx + 4] = cy;
        idx += 5;
      }
    }

    // Modify the matrix
    flag = HYPRE_StructMatrixSetBoxValues(Jmatrix, ilower, iupper, 5, entries,
                                          work);
    if (flag != 0)
    {
      cerr << "Error in HYPRE_StructMatrixSetBoxValues (interior) = " << flag
           << endl;
      return -1;
    }

    // ----------------------------------------
    // Correct matrix values at boundary nodes
    // ----------------------------------------

    // Set the matrix boundary entries (center, left, right, bottom, top)
    if (ilower[1] == 0 || iupper[1] == (udata->ny - 1) || ilower[0] == 0 ||
        iupper[0] == (udata->nx - 1))
    {
      idx = 0;
      for (iy = 0; iy < ny_loc; iy++)
      {
        for (ix = 0; ix < nx_loc; ix++)
        {
          work[idx]     = ONE;
          work[idx + 1] = ZERO;
          work[idx + 2] = ZERO;
          work[idx + 3] = ZERO;
          work[idx + 4] = ZERO;
          idx += 5;
        }
      }
    }

    // Set cells on western boundary
    if (ilower[0] == 0)
    {
      // Grid cell on south-west corner
      bc_ilower[0] = ilower[0];
      bc_ilower[1] = ilower[1];

      // Grid cell on north-west corner
      bc_iupper[0] = ilower[0];
      bc_iupper[1] = iupper[1];

      // Only do work if the box is non-zero in size
      if ((bc_ilower[0] <= bc_iupper[0]) && (bc_ilower[1] <= bc_iupper[1]))
      {
        // Modify the matrix
        flag = HYPRE_StructMatrixSetBoxValues(Jmatrix, bc_ilower, bc_iupper, 5,
                                              entries, work);
        if (flag != 0)
        {
          cerr << "Error in HYPRE_StructMatrixSetBoxValues (west bdry) = " << flag
               << endl;
          return -1;
        }
      }
    }

    // Set cells on eastern boundary
    if (iupper[0] == (udata->nx - 1))
    {
      // Grid cell on south-east corner
      bc_ilower[0] = iupper[0];
      bc_ilower[1] = ilower[1];

      // Grid cell on north-east corner
      bc_iupper[0] = iupper[0];
      bc_iupper[1] = iupper[1];

      // Only do work if the box is non-zero in size
      if ((bc_ilower[0] <= bc_iupper[0]) && (bc_ilower[1] <= bc_iupper[1]))
      {
        // Modify the matrix
        flag = HYPRE_StructMatrixSetBoxValues(Jmatrix, bc_ilower, bc_iupper, 5,
                                              entries, work);
        if (flag != 0)
        {
          cerr << "Error in HYPRE_StructMatrixSetBoxValues (east bdry) = " << flag
               << endl;
          return -1;
        }
      }
    }

    // Correct cells on southern boundary
    if (ilower[1] == 0)
    {
      // Grid cell on south-west corner
      bc_ilower[0] = ilower[0];
      bc_ilower[1] = ilower[1];

      // Grid cell on south-east corner
      bc_iupper[0] = iupper[0];
      bc_iupper[1] = ilower[1];

      // Only do work if the box is non-zero in size
      if ((bc_ilower[0] <= bc_iupper[0]) && (bc_ilower[1] <= bc_iupper[1]))
      {
        // Modify the matrix
        flag = HYPRE_StructMatrixSetBoxValues(Jmatrix, bc_ilower, bc_iupper, 5,
                                              entries, work);
        if (flag != 0)
        {
          cerr << "Error in HYPRE_StructMatrixSetBoxValues (south bdry) = "
               << flag << endl;
          return -1;
        }
      }
    }

    // Set cells on northern boundary
    if (iupper[1] == (udata->ny - 1))
    {
      // Grid cell on north-west corner
      bc_ilower[0] = ilower[0];
      bc_ilower[1] = iupper[1];

      // Grid cell on north-east corner
      bc_iupper[0] = iupper[0];
      bc_iupper[1] = iupper[1];

      // Only do work if the box is non-zero in size
      if ((bc_ilower[0] <= bc_iupper[0]) && (bc_ilower[1] <= bc_iupper[1]))
      {
        // Modify the matrix
        flag = HYPRE_StructMatrixSetBoxValues(Jmatrix, bc_ilower, bc_iupper, 5,
                                              entries, work);
        if (flag != 0)
        {
          cerr << "Error in HYPRE_StructMatrixSetBoxValues (north bdry) = "
               << flag << endl;
          return -1;
        }
      }
    }

    // -----------------------------------------------------------
    // Remove connections between the interior and boundary nodes
    // -----------------------------------------------------------

    // Zero out work array
    for (ix = 0; ix < nwork; ix++) { work[ix] = ZERO; }

    // Second column of nodes (depends on western boundary)
    if ((ilower[0] <= 1) && (iupper[0] >= 1))
    {
      // Remove western dependency
      entry[0] = 1;

      // Grid cell on south-west corner
      bc_ilower[0] = 1;
      bc_ilower[1] = ilower[1];

      // Grid cell on north-west corner
      bc_iupper[0] = 1;
      bc_iupper[1] = iupper[1];

      // Only do work if the box is non-zero in size
      if ((bc_ilower[0] <= bc_iupper[0]) && (bc_ilower[1] <= bc_iupper[1]))
      {
        // Modify the matrix
        flag = HYPRE_StructMatrixSetBoxValues(Jmatrix, bc_ilower, bc_iupper, 1,
                                              entry, work);
        if (flag != 0)
        {
          cerr << "Error in HYPRE_StructMatrixSetBoxValues (disconnect west "
                  "bdry) = "
               << flag << endl;
          return -1;
        }
      }
    }

    // Next to last column (depends on eastern boundary)
    if ((ilower[0] <= (udata->nx - 2)) && (iupper[0] >= (udata->nx - 2)))
    {
      // Remove eastern dependency
      entry[0] = 2;

      // Grid cell on south-east corner
      bc_ilower[0] = udata->nx - 2;
      bc_ilower[1] = ilower[1];

      // Grid cell on north-east corner
      bc_iupper[0] = udata->nx - 2;
      bc_iupper[1] = iupper[1];

      // Only do work if the box is non-zero in size
      if ((bc_ilower[0] <= bc_iupper[0]) && (bc_ilower[1] <= bc_iupper[1]))
      {
        // Modify the matrix
        flag = HYPRE_StructMatrixSetBoxValues(Jmatrix, bc_ilower, bc_iupper, 1,
                                              entry, work);
        if (flag != 0)
        {
          cerr << "Error in HYPRE_StructMatrixSetBoxValues (disconnect east "
                  "bdry) = "
               << flag << endl;
          return -1;
        }
      }
    }

    // Second row of nodes (depends on southern boundary)
    if ((ilower[1] <= 1) && (iupper[1] >= 1))
    {
      // Remove southern dependency
      entry[0] = 3;

      // Grid cell on south-west corner
      bc_ilower[0] = ilower[0];
      bc_ilower[1] = 1;

      // Grid cell on south-east corner
      bc_iupper[0] = iupper[0];
      bc_iupper[1] = 1;

      // Only do work if the box is non-zero in size
      if ((bc_ilower[0] <= bc_iupper[0]) && (bc_ilower[1] <= bc_iupper[1]))
      {
        // Modify the matrix
        flag = HYPRE_StructMatrixSetBoxValues(Jmatrix, bc_ilower, bc_iupper, 1,
                                              entry, work);
        if (flag != 0)
        {
          cerr << "Error in HYPRE_StructMatrixSetBoxValues (disconnect south "
                  "bdry) = "
               << flag << endl;
          return -1;
        }
      }
    }

    // Next to last row of nodes (depends on northern boundary)
    if ((ilower[1] <= (udata->ny - 2)) && (iupper[1] >= (udata->ny - 2)))
    {
      // Remove northern dependency
      entry[0] = 4;

      // Grid cell on north-west corner
      bc_ilower[0] = ilower[0];
      bc_ilower[1] = udata->ny - 2;

      // Grid cell on north-east corner
      bc_iupper[0] = iupper[0];
      bc_iupper[1] = udata->ny - 2;

      // Only do work if the box is non-zero in size
      if ((bc_ilower[0] <= bc_iupper[0]) && (bc_ilower[1] <= bc_iupper[1]))
      {
        // Modify the matrix
        flag = HYPRE_StructMatrixSetBoxValues(Jmatrix, bc_ilower, bc_iupper, 1,
                                              entry, work);
        if (flag != 0)
        {
          cerr << "Error in HYPRE_StructMatrixSetBoxValues (disconnect north "
                  "bdry) = "
               << flag << endl;
          return -1;
        }
      }
    }
  }

  // The matrix is assembled matrix in hypre setup

  // Stop timer
  double t2 = MPI_Wtime();

  // Update timer
  udata->matfilltime += t2 - t1;

  // Return success
  return 0;
}

// Fill A = I - gamma * J matrix
static int ScaleAddI(UserData* udata, sunrealtype gamma)
{
  int flag;

  // Variable shortcuts
  HYPRE_Int ilower[2];
  HYPRE_Int iupper[2];

  ilower[0] = udata->ilower[0];
  ilower[1] = udata->ilower[1];

  iupper[0] = udata->iupper[0];
  iupper[1] = udata->iupper[1];

  HYPRE_Int nwork  = udata->nwork;
  HYPRE_Real* work = udata->work;

  // Matrix stencil: center, left, right, bottom, top
  HYPRE_Int entries[5] = {0, 1, 2, 3, 4};

  // Copy all matrix values into work array from J
  flag = HYPRE_StructMatrixGetBoxValues(udata->Jmatrix, ilower, iupper, 5,
                                        entries, work);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructMatrixGetBoxValues = " << flag << endl;
    return (flag);
  }

  // Scale work array by c
  for (HYPRE_Int i = 0; i < nwork; i++) { work[i] *= -gamma; }

  // Insert scaled values into A
  flag = HYPRE_StructMatrixSetBoxValues(udata->Amatrix, ilower, iupper, 5,
                                        entries, work);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructMatrixSetBoxValues = " << flag << endl;
    return (flag);
  }

  // Set first 1/5 of work array to 1
  for (HYPRE_Int i = 0; i < nwork / 5; i++) { work[i] = ONE; }

  // Add values to the diagonal of A
  HYPRE_Int entry[1] = {0};
  flag = HYPRE_StructMatrixAddToBoxValues(udata->Amatrix, ilower, iupper, 1,
                                          entry, work);
  if (flag != 0)
  {
    cerr << "Error in HYPRE_StructMatrixAddToBoxValues = " << flag << endl;
    return (flag);
  }

  // Return success
  return 0;
}
