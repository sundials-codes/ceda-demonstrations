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
 * The input parsing source file for the gk_diffusion project.
 * ---------------------------------------------------------------------------*/

#ifndef INPUT_HANDLER_H
#define INPUT_HANDLER_H

#include "input_handler.h"
#include <ctype.h>

// -----------------------------------------------------------------------------
// UserData and input functions
// -----------------------------------------------------------------------------

// Initialize memory allocated within UserData
int InitUserData(UserData* udata)
{
  // Diffusion coefficient
  udata->k = SUN_RCONST(0.1);

  // Final time
  udata->tf = ONE;

  // Integrator settings
  udata->rtol           = SUN_RCONST(1.e-5);  // relative tolerance
  udata->atol           = SUN_RCONST(1.e-12); // absolute tolerance
  udata->hfixed         = ZERO;               // using adaptive step sizes
  udata->maxsteps       = 100000;             // max steps between outputs
  udata->wrms_norm_type = 2;                  // cellwise wrms norm

  // LSRKStep options
  udata->method          = ARKODE_LSRK_RKL_2; // RKL
  udata->eigfrequency    = 0;   // 0 refers to constant dominant eigenvalue
  udata->stage_max_limit = 200; // allow up to 200 stages/step
  udata->num_SSP_stages  = 4;   // number of stages in the SSP method
  udata->eigsafety       = SUN_RCONST(1.01); // 1% safety factor

  // Output variables
  udata->nout = 10; // Number of output times

  // DEE options
  udata->user_dom_eig = SUNFALSE; // No user-provided dominant eigenvalue function
  udata->dee_id            = 0;   // DEE ID (0 for PI and 1 for Arnoldi)
  udata->dee_num_init_wups = 100;
  udata->dee_num_succ_wups = 0;
  udata->dee_max_iters     = 1000;
  udata->dee_krylov_dim    = 3;
  udata->dee_reltol        = 0.01;

  // Return success
  return 0;
}

// Function to check if a string is a valid integer
sunbooleantype isInteger(const char* str)
{
  if (str == NULL || *str == '\0')
  { // Handle empty or NULL strings
    return SUNFALSE;
  }

  // Handle optional leading sign
  int i = 0;
  if (str[0] == '-' || str[0] == '+') { i = 1; }

  // Check if there are any digits after the sign (if present)
  if (str[i] == '\0')
  {
    return SUNFALSE; // Only a sign, no digits
  }

  // Iterate through the rest of the string
  for (; str[i] != '\0'; i++)
  {
    if (!isdigit(str[i]))
    {
      return SUNFALSE; // Found a non-digit character
    }
  }
  return SUNTRUE; // All characters are digits (or a valid sign followed by digits)
}

// Read command line inputs
int ReadInputs(int argc, char** argv, UserData* udata)
{
  // Check for input args
  int arg_idx = 1;

  while (arg_idx < argc)
  {
    char* arg = argv[arg_idx++];

    // Gkeyll runtime arguments.
    if (strcmp(arg, "-g") == 0) {}
    else if (strcmp(arg, "-M") == 0) {}
    else if (strcmp(arg, "-s") == 0) {}
    else if (strcmp(arg, "-r") == 0) {}
    else if (strcmp(arg, "-o") == 0) {}
    else if (strcmp(arg, "-c") == 0) {}
    // Diffusion parameters
    else if (strcmp(arg, "--k") == 0) { udata->k = atof(argv[arg_idx++]); }
    // Temporal domain settings
    else if (strcmp(arg, "--tf") == 0) { udata->tf = atof(argv[arg_idx++]); }
    // Integrator settings
    else if (strcmp(arg, "--rtol") == 0)
    {
      udata->rtol = atof(argv[arg_idx++]);
    }
    else if (strcmp(arg, "--atol") == 0)
    {
      udata->atol = atof(argv[arg_idx++]);
    }
    else if (strcmp(arg, "--fixedstep") == 0)
    {
      udata->hfixed = atof(argv[arg_idx++]);
    }
    else if (strcmp(arg, "--method") == 0)
    {
      udata->method = (ARKODE_LSRKMethodType)atoi(argv[arg_idx++]);
    }
    else if (strcmp(arg, "--eigfrequency") == 0)
    {
      udata->eigfrequency = atol(argv[arg_idx++]);
    }
    else if (strcmp(arg, "--stage_max_limit") == 0)
    {
      udata->stage_max_limit = atoi(argv[arg_idx++]);
    }
    else if (strcmp(arg, "--num_SSP_stages") == 0)
    {
      udata->num_SSP_stages = atoi(argv[arg_idx++]);
    }
    else if (strcmp(arg, "--eigsafety") == 0)
    {
      udata->eigsafety = atof(argv[arg_idx++]);
    }
    // Output settings
    else if (strcmp(arg, "--nout") == 0)
    {
      udata->nout = atoi(argv[arg_idx++]);
    }
    else if (strcmp(arg, "--wrms_norm_type") == 0)
    {
      udata->wrms_norm_type = atoi(argv[arg_idx++]);
    }
    else if (strcmp(arg, "--maxsteps") == 0)
    {
      udata->maxsteps = atoi(argv[arg_idx++]);
    }
    // DEE options
    else if (strcmp(arg, "--user_dom_eig") == 0)
    {
      udata->user_dom_eig = SUNTRUE;
    }
    else if (strcmp(arg, "--dee_id") == 0)
    {
      udata->dee_id = atoi(argv[arg_idx++]);
    }
    else if (strcmp(arg, "--dee_num_init_wups") == 0)
    {
      udata->dee_num_init_wups = atoi(argv[arg_idx++]);
    }
    else if (strcmp(arg, "--dee_num_succ_wups") == 0)
    {
      udata->dee_num_succ_wups = atoi(argv[arg_idx++]);
    }
    else if (strcmp(arg, "--dee_max_iters") == 0)
    {
      udata->dee_max_iters = atoi(argv[arg_idx++]);
    }
    else if (strcmp(arg, "--dee_krylov_dim") == 0)
    {
      udata->dee_krylov_dim = atoi(argv[arg_idx++]);
    }
    else if (strcmp(arg, "--dee_reltol") == 0)
    {
      udata->dee_reltol = atof(argv[arg_idx++]);
    }
    // Help
    else if (strcmp(arg, "--help") == 0)
    {
      InputHelp();
      return -1;
    }
    // Unknown input
    else
    {
      int isnum = isInteger(arg);
      if (isnum == 0)
      {
        fprintf(stderr, "ERROR: Invalid input %s\n", arg);
        InputHelp();
        return -1;
      }
    }
  }

  return 0;
}

// Print command line options
void InputHelp(void)
{
  printf("\n");
  printf("Command line options:\n");
  printf("   -g                         : Run on GPUs if GPUs are present and "
         "code built for GPUs\n");
  printf(
    "   -M                         : Run with MPI if code built with MPI\n");
  printf("   -s <N>                     : Only run N steps of simulation\n");
  printf(
    "   -r <N>                     : Restart the simulation from frame N\n");
  printf("   -o                         : Optional arguments (as string, "
         "requires parsing)\n");
  printf("   -c <N>                     : Domain decomposition in x\n");
  printf("  --k <amplitude>             : diffusion amplitude\n");
  printf("  --tf <time>                 : final time\n");
  printf("  --rtol <rtol>               : relative tolerance\n");
  printf("  --atol <atol>               : absolute tolerance\n");
  printf("  --fixedstep <step>          : used fixed step size\n");
  printf("  --wrms_norm_type <type>     : WRMS norm type (componentwise: 1, "
         "cellwise norm:2)\n");
  printf("  --method <mth>              : LSRK method choice (0:RCK, 1:RKL, "
         "2:SSP2, 3:SSP3, 4:SSP4)\n");
  printf(
    "  --eigfrequency <nst>        : dominant eigenvalue update frequency\n");
  printf("  --stage_max_limit <smax>    : maximum number of stages per step\n");
  printf(
    "  --num_SSP_stages <nstages>  : number of stages in the SSP method\n");
  printf("  --eigsafety <safety>        : dominant eigenvalue safety factor\n");
  printf("  --nout <nout>               : number of outputs\n");
  printf("  --maxsteps <steps>          : max steps between outputs\n");
  printf("  --user_dom_eig              : use user-provided dominant "
         "eigenvalue function\n");
  printf("  --dee_id <id>               : DomEig Estimator (DEE) id (PI: 0, "
         "Arnoldi: 1)\n");
  printf("  --dee_num_init_wups <num>   : number of DEE initial warmups\n");
  printf("  --dee_num_succ_wups <num>   : number of DEE succeeding warmups\n");
  printf("  --dee_max_iters <num>       : max iterations in DEE\n");
  printf("  --dee_krylov_dim <dim>      : Krylov dimension for DEE\n");
  printf("  --dee_reltol <tol>          : DEE tolerance\n");
  printf("  --help                      : print this message and exit\n");
}

// Print user data
int PrintUserData(UserData* udata, int rank)
{
  if (rank == 0)
  {
    printf("\n");
    printf("gkeyll diffusion test problem:\n");
    printf(" ------------------------------------ \n");
    printf(" k                 = %g\n", udata->k);
    printf(" tf                = %g\n", udata->tf);
    printf(" nout              = %d\n", udata->nout);
    printf(" ------------------------------------ \n");
    printf(" rtol              = %.2e\n", udata->rtol);
    printf(" atol              = %.2e\n", udata->atol);
    printf(" hfixed            = %.2e\n", udata->hfixed);
    printf(" method            = %d\n", (int)udata->method);
    printf(" eigfrequency      = %ld\n", udata->eigfrequency);
    printf(" stage_max_limit   = %d\n", udata->stage_max_limit);
    printf(" num SSP stages    = %d\n", udata->num_SSP_stages);
    printf(" eigsafety         = %g\n", udata->eigsafety);
    printf(" ------------------------------------ \n");
    printf(" wrms norm type    = %d\n", udata->wrms_norm_type);
    printf(" max steps         = %d\n", udata->maxsteps);
    printf(" ------------------------------------ \n");
    printf(" dom_eig provided  = %d\n", udata->user_dom_eig);
    printf(" dee ID            = %d\n", udata->dee_id);
    printf(" dee num_init_wups = %d\n", udata->dee_num_init_wups);
    printf(" dee num_succ_wups = %d\n", udata->dee_num_succ_wups);
    printf(" dee_max_iters     = %d\n", udata->dee_max_iters);
    printf(" dee_krylov_dim    = %d\n", udata->dee_krylov_dim);
    printf(" dee_reltol        = %g\n", udata->dee_reltol);
    printf(" ------------------------------------ \n\n");
  }

  return 0;
}

#endif // INPUT_HANDLER_H
