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
 * Shared header file for 2D diffusion benchmark problem
 * ---------------------------------------------------------------------------*/

#ifndef DIFFUSION_2D_HPP
#define DIFFUSION_2D_HPP

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <sundials/sundials_core.hpp>
#include <vector>

#include "mpi.h"

#if defined(USE_HIP)
#include "nvector/nvector_hip.h"
#include "nvector/nvector_mpiplusx.h"
#elif defined(USE_CUDA)
#include "nvector/nvector_cuda.h"
#include "nvector/nvector_mpiplusx.h"
#else
#include "nvector/nvector_parallel.h"
#endif
#if defined(USE_HYPRE)
#include "HYPRE_struct_ls.h"
#endif
#include "sunlinsol/sunlinsol_pcg.h"
#include "sunlinsol/sunlinsol_spgmr.h"

// Macros for problem constants
#define ZERO  SUN_RCONST(0.0)
#define ONE   SUN_RCONST(1.0)
#define TWO   SUN_RCONST(2.0)
#define EIGHT SUN_RCONST(8.0)

// Macro to access (x,y) location in 1D NVector array
#define IDX(x, y, n) ((n) * (y) + (x))

// Ceiling for integers ceil(a/b) = ceil((a + b - 1) / b)
#define ICEIL(a, b) (((a) + (b)-1) / (b))

using namespace std;

// -----------------------------------------------------------------------------
// UserData structure
// -----------------------------------------------------------------------------

struct UserData
{
  SUNProfiler prof = NULL;

  // Diffusion coefficients in the x and y directions
  sunrealtype kx = ONE;
  sunrealtype ky = ONE;
  bool inhomogeneous = false;

  // Final time
  sunrealtype tf = ONE;

  // Lower and Upper bounds in x and y directions
  sunrealtype xl = -M_PI;
  sunrealtype yl = -6.0;
  sunrealtype xu = M_PI;
  sunrealtype yu = 6.0;

  // Global number of nodes in the x and y directions
  sunindextype nx = 64;
  sunindextype ny = 64;

  // Global total number of nodes
  sunindextype nodes = nx * ny;

  // Mesh spacing in the x and y directions
  sunrealtype dx = (xu - xl) / (nx - 1);
  sunrealtype dy = (yu - xl) / (ny - 1);

  // Minimum number of local nodes in the x and y directions
  sunindextype qx = 0;
  sunindextype qy = 0;

  // Leftover nodes in the x and y directions
  sunindextype rx = 0;
  sunindextype ry = 0;

  // Local number of nodes in the x and y directions
  sunindextype nx_loc = 0;
  sunindextype ny_loc = 0;

  // Overall number of local nodes
  sunindextype nodes_loc = 0;

  // Global x and y indices of this subdomain
  sunindextype is = 0; // x starting index
  sunindextype ie = 0; // x ending index
  sunindextype js = 0; // y starting index
  sunindextype je = 0; // y ending index

  // MPI variables
  MPI_Comm comm_c = MPI_COMM_NULL; // Cartesian communicator in space

  int np  = 1; // total number of MPI processes in Comm world
  int npx = 0; // number of MPI processes in the x-direction
  int npy = 0; // number of MPI processes in the y-direction

  int myid_c = 0; // process ID in Cartesian communicator
  int idx    = 0; // process x-coordinate
  int idy    = 0; // process y-coordinate

  // Flags denoting if this process has a neighbor
  bool HaveNbrW = true;
  bool HaveNbrE = true;
  bool HaveNbrS = true;
  bool HaveNbrN = true;

  // Neighbor IDs for exchange
  int ipW = -1;
  int ipE = -1;
  int ipS = -1;
  int ipN = -1;

  // Receive buffers for neighbor exchange
  sunrealtype* Wrecv = NULL;
  sunrealtype* Erecv = NULL;
  sunrealtype* Srecv = NULL;
  sunrealtype* Nrecv = NULL;

  // Receive requests for neighbor exchange
  MPI_Request reqRW;
  MPI_Request reqRE;
  MPI_Request reqRS;
  MPI_Request reqRN;

  // Send buffers for neighbor exchange
  sunrealtype* Wsend = NULL;
  sunrealtype* Esend = NULL;
  sunrealtype* Ssend = NULL;
  sunrealtype* Nsend = NULL;

  // Send requests for neighbor exchange
  MPI_Request reqSW;
  MPI_Request reqSE;
  MPI_Request reqSS;
  MPI_Request reqSN;

  // Preconditioner data
#ifdef USE_HYPRE
  HYPRE_StructGrid grid = NULL;
  HYPRE_StructStencil stencil = NULL;
  HYPRE_StructMatrix Jmatrix = NULL;
  HYPRE_StructMatrix Amatrix = NULL;
  HYPRE_StructVector bvec = NULL;
  HYPRE_StructVector xvec = NULL;
  HYPRE_StructVector vvec = NULL;
  HYPRE_StructSolver precond = NULL;

  // hypre grid extents
  HYPRE_Int ilower[2];
  HYPRE_Int iupper[2];

  // hypre grid periodicity
  HYPRE_Int periodic[2];

  // hypre workspace
  HYPRE_Int nwork;
  HYPRE_Real* work;

  // hypre counters
  HYPRE_Int pfmg_its = 0;

  // hypre PFMG settings (hypre defaults)
  HYPRE_Int pfmg_relax = 2;  // type of relaxation:
                             //   0 - Jacobi
                             //   1 - Weighted Jacobi
                             //   2 - symmetric R/B Gauss-Seidel
                             //   3 - nonsymmetric R/B Gauss-Seidel
  HYPRE_Int pfmg_nrelax = 2; // number of pre and post relaxation sweeps

#else
  N_Vector diag = NULL;
#endif

  UserData(SUNProfiler prof_) : prof(prof_) {}

  ~UserData();

  // Helper functions
  int parse_args(vector<string>& args, bool outproc);
  void help();
  void print();
  int setup();
  int start_exchange(const N_Vector u);
  int end_exchange();

private:
  int allocate_buffers();
  int pack_buffers(const N_Vector u);
  int free_buffers();
};

// -----------------------------------------------------------------------------
// UserOutput structure
// -----------------------------------------------------------------------------

struct UserOutput
{
  // Output variables
  int output     = 1;    // 0 = no output, 1 = stats output, 2 = output to disk
  int nout       = 20;   // number of output times
  N_Vector error = NULL; // error vector
  ofstream uoutstream;   // output file stream
  ofstream eoutstream;   // error file stream

  // Helper functions
  int parse_args(vector<string>& args, bool outproc);
  void help();
  void print();

  // Output functions
  int open(UserData* udata);
  int write(sunrealtype t, N_Vector u, UserData* udata);
  int close(UserData* udata);
};

// -----------------------------------------------------------------------------
// Functions provided to the SUNDIALS integrator
// -----------------------------------------------------------------------------

// Common function for computing the Laplacian
int laplacian(sunrealtype t, N_Vector u, N_Vector f, UserData* udata);

// ODE right hand side function
int diffusion(sunrealtype t, N_Vector u, N_Vector f, void* user_data);

// Preconditioner setup and solve functions
int PSetup(sunrealtype t, N_Vector u, N_Vector f, sunbooleantype jok,
           sunbooleantype* jcurPtr, sunrealtype gamma, void* user_data);

int PSolve(sunrealtype t, N_Vector u, N_Vector f, N_Vector r, N_Vector z,
           sunrealtype gamma, sunrealtype delta, int lr, void* user_data);

// -----------------------------------------------------------------------------
// Utility functions
// -----------------------------------------------------------------------------

// Diffusion coefficients
sunrealtype Diffusion_Coeff_X(sunrealtype x, UserData* udata);
sunrealtype Diffusion_Coeff_Y(sunrealtype y, UserData* udata);

// Compute the initial condition
int Initial(sunrealtype t, N_Vector u, UserData* udata);

// Check function return values
int check_flag(const void* flagvalue, const string funcname, int opt);

#if defined(USE_HYPRE)
int SetupHypre(UserData& udata);
#endif

#endif
