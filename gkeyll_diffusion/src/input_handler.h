/* -----------------------------------------------------------------------------
 * Programmer(s): Mustafa Aggul @ UMBC
 * -----------------------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2025, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------------------
 * The input parsing header file for the gk_diffusion project.
 * ---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "arkode/arkode_lsrkstep.h" // access to LSRKStep
#include "nvector/nvector_serial.h" // access to the serial N_Vector

#define PI   SUN_RCONST(3.141592653589793238462643383279502884197169)
#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)

// -----------------------------------------------------------------------------
// User data structure
// -----------------------------------------------------------------------------

typedef struct
{
  // Diffusion amplitude
  sunrealtype k;

  // Final time
  sunrealtype tf;

  // Integrator settings
  sunrealtype rtol;           // relative tolerance
  sunrealtype atol;           // absolute tolerance
  sunrealtype hfixed;         // fixed step size
  int maxsteps;               // max number of steps between outputs
  int wrms_norm_type;         // wrms norm type (1:componentwise 2:cellwise)

  // LSRKStep options
  ARKODE_LSRKMethodType method; // LSRK method choice
  long int eigfrequency;        // dominant eigenvalue update frequency
  int stage_max_limit;          // maximum number of stages per step
  int num_SSP_stages;           // number of stages in the SSP method
  sunrealtype eigsafety;        // dominant eigenvalue safety factor

  // Output variables
  int nout;       // number of output times

  // DEE options
  sunbooleantype user_dom_eig; // whether a user-provided dominant eigenvalue function is used
  int dee_id;            // DEE ID (0: Power, 1: Arnoldi)
  int dee_num_init_wups; // number of initial warmups before the first estimate
  int dee_num_succ_wups; // number of succeeding warmups before each estimate
  int dee_max_iters;     // max number of iterations
  int dee_krylov_dim;    // Krylov dimension for DEE
  double dee_reltol;     // tolerance

} UserData;

// -----------------------------------------------------------------------------
// UserData and input functions
// -----------------------------------------------------------------------------

// Set the default values in the UserData structure
int InitUserData(UserData* udata);

// Free memory allocated within UserData
int FreeUserData(UserData* udata);

// Read the command line inputs and set UserData values
int ReadInputs(int argc, char** argv, UserData* udata);

// -----------------------------------------------------------------------------
// Output and utility functions
// -----------------------------------------------------------------------------

// Print the command line options
void InputHelp();

// Print some UserData information
int PrintUserData(UserData* udata);

// Print integration timing
int OutputTiming(UserData* udata);