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
 * ARKODE main for 2D diffusion benchmark problem
 * ---------------------------------------------------------------------------*/

#include "arkode/arkode_arkstep.h"
#include "arkode/arkode_lsrkstep.h"
#include "diffusion_2D.hpp"
#include "sunadaptcontroller/sunadaptcontroller_imexgus.h"
#include "sunadaptcontroller/sunadaptcontroller_soderlind.h"
#include <sundomeigest/sundomeigest_power.h>

struct UserOptions
{
  // Integrator settings
  std::string integrator = "dirk";              // time integration method
  sunrealtype rtol       = SUN_RCONST(1.0e-5);  // relative tolerance
  sunrealtype atol       = SUN_RCONST(1.0e-10); // absolute tolerance
  sunrealtype hfixed     = ZERO;                // fixed step size
  int order              = 2;                   // ARKode method order
  int controller         = 0;                   // step size adaptivity method
  int maxsteps           = 0;                   // max steps between outputs
  int onestep            = 0;     // one step mode, number of steps
  bool error             = false; // compute reference solution to compare error

  // Solver and preconditioner settings
  bool linear          = true;  // linearly implicit RHS
  std::string ls       = "cg";  // linear solver to use
  bool preconditioning = true;  // preconditioner on/off
  bool lsinfo          = false; // output residual history
  int liniters         = 20;    // number of linear iterations
  int msbp             = 0;     // preconditioner setup frequency
  sunrealtype epslin   = ZERO;  // linear solver tolerance factor
  bool internaleig     = false; // internal eigenvalue estimation on/off

  // Helper functions
  int parse_args(vector<string>& args, bool outproc);
  void help();
  void print();
};

// -----------------------------------------------------------------------------
// LSRKStep-specific dominant eigenvalue function prototype
// -----------------------------------------------------------------------------

static int dom_eig(sunrealtype t, N_Vector y, N_Vector fn, sunrealtype* lambdaR,
                   sunrealtype* lambdaI, void* user_data, N_Vector temp1,
                   N_Vector temp2, N_Vector temp3);

// -----------------------------------------------------------------------------
// Main Program
// -----------------------------------------------------------------------------

int main(int argc, char* argv[])
{
  // Reusable error-checking flag
  int flag;

  // Initialize MPI
  flag = MPI_Init(&argc, &argv);
  if (check_flag(&flag, "MPI_Init", 1)) { return 1; }

  // Create SUNDIALS context
  MPI_Comm comm    = MPI_COMM_WORLD;
  SUNContext ctx   = nullptr;
  SUNProfiler prof = nullptr;

  flag = SUNContext_Create(comm, &ctx);
  if (check_flag(&flag, "SUNContextCreate", 1)) { return 1; }

  flag = SUNContext_GetProfiler(ctx, &prof);
  if (check_flag(&flag, "SUNContext_GetProfiler", 1)) { return 1; }

  // Add scope so objects are destroyed before MPI_Finalize
  {
    SUNDIALS_CXX_MARK_FUNCTION(prof);

    // MPI process ID
    int myid;
    flag = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (check_flag(&flag, "MPI_Comm_rank", 1)) { return 1; }

    bool outproc = (myid == 0);

    // --------------------------
    // Parse command line inputs
    // --------------------------

    UserData udata(prof);
    UserOptions uopts;
    UserOutput uout;

    vector<string> args(argv + 1, argv + argc);

    flag = udata.parse_args(args, outproc);
    if (check_flag(&flag, "UserData::parse_args", 1)) { return 1; }

    flag = uopts.parse_args(args, outproc);
    if (check_flag(&flag, "UserOptions::parse_args", 1)) { return 1; }

    flag = uout.parse_args(args, outproc);
    if (check_flag(&flag, "UserOutput::parse_args", 1)) { return 1; }

    // Check for unparsed inputs
    if (args.size() > 0)
    {
      // output any un-processed input arguments
      if (find(args.begin(), args.end(), "--help") == args.end())
      {
        cerr << "ERROR: Unknown inputs: ";
        for (auto i = args.begin(); i != args.end(); ++i) { cerr << *i << ' '; }
        cerr << endl;
      }
      // the user specified "--help" on the command-line, so exit gracefully.
      else
      {
        flag = MPI_Finalize();
        return 0;
      }
      return 1;
    }

    // Return with error on unsupported integration method type
    if ((uopts.integrator != "dirk") && (uopts.integrator != "erk") &&
        (uopts.integrator != "rkc") && (uopts.integrator != "rkl"))
    {
      cerr << "ERROR: illegal integrator" << endl;
      return 1;
    }

    // Set boolean control parameters based on user inputs
    bool impl = (uopts.integrator == "dirk");
    bool expl = (uopts.integrator == "erk");
    bool sts  = ((uopts.integrator == "rkc") || (uopts.integrator == "rkl"));

#ifdef USE_HYPRE
#if HYPRE_RELEASE_NUMBER >= 22000
    if (impl && uopts.preconditioning)
    {
      flag = HYPRE_Init();
      if (check_flag(&flag, "HYPRE_Init", 1)) { return 1; }
    }
#endif
#endif

    // -----------------------------
    // Setup parallel decomposition
    // -----------------------------

    flag = udata.setup();
    if (check_flag(&flag, "UserData::setup", 1)) { return 1; }

    if (outproc)
    {
      udata.print();
      uopts.print();
      uout.print();
    }

    // ---------------
    // Create vectors
    // ---------------

    // Create vector for solution
    N_Vector u = N_VNew_Parallel(udata.comm_c, udata.nodes_loc, udata.nodes, ctx);
    if (check_flag((void*)u, "N_VNew_Parallel", 0)) { return 1; }

    // Set initial condition
    flag = Initial(ZERO, u, &udata);
    if (check_flag(&flag, "Initial", 1)) { return 1; }

    // if computing reference solution and error, create vectors
    N_Vector uref = nullptr;
    N_Vector uerr = nullptr;
    if (uopts.error)
    {
      uref = N_VClone(u);
      if (check_flag((void*)uref, "N_VClone", 0)) { return 1; }
      uerr = N_VClone(u);
      if (check_flag((void*)uerr, "N_VClone", 0)) { return 1; }
      N_VScale(1.0, u, uref);
      N_VConst(0.0, uerr);
    }

    // Set up implicit solver, if applicable
    SUNLinearSolver LS = nullptr;
    SUNMatrix A        = nullptr;
    if (impl)
    {
      // ---------------------
      // Create linear solver
      // ---------------------

      // Create linear solver

      int prectype = (uopts.preconditioning) ? SUN_PREC_RIGHT : SUN_PREC_NONE;

      if (uopts.ls == "cg")
      {
        LS = SUNLinSol_PCG(u, prectype, uopts.liniters, ctx);
        if (check_flag((void*)LS, "SUNLinSol_PCG", 0)) { return 1; }
      }
      else if (uopts.ls == "gmres")
      {
        LS = SUNLinSol_SPGMR(u, prectype, uopts.liniters, ctx);
        if (check_flag((void*)LS, "SUNLinSol_SPGMR", 0)) { return 1; }
      }

      // Allocate preconditioner workspace
      if (uopts.preconditioning)
      {
#ifdef USE_HYPRE
        flag = SetupHypre(udata);
        if (check_flag(&flag, "SetupHypre", 1)) { return 1; }
#else
        udata.diag = N_VClone(u);
        if (check_flag((void*)(udata.diag), "N_VClone", 0)) { return 1; }
#endif
      }
    }

    // ----------------------
    // Setup ARKStep/LSRKStep
    // ----------------------

    // Create integrator
    void* arkode_mem = nullptr;
    void* arkref_mem = nullptr;
    if (impl)
    {
      arkode_mem = ARKStepCreate(nullptr, diffusion, ZERO, u, ctx);
      if (check_flag((void*)arkode_mem, "ARKStepCreate", 0)) { return 1; }
    }
    else if (expl)
    {
      arkode_mem = ARKStepCreate(diffusion, nullptr, ZERO, u, ctx);
      if (check_flag((void*)arkode_mem, "ARKStepCreate", 0)) { return 1; }
    }
    else
    {
      arkode_mem = LSRKStepCreateSTS(diffusion, ZERO, u, ctx);
      if (check_flag((void*)arkode_mem, "LSRKStepCreateSTS", 0)) { return 1; }
    }

    // Specify tolerances
    flag = ARKodeSStolerances(arkode_mem, uopts.rtol, uopts.atol);
    if (check_flag(&flag, "ARKodeSStolerances", 1)) { return 1; }

    // Attach user data
    flag = ARKodeSetUserData(arkode_mem, (void*)&udata);
    if (check_flag(&flag, "ARKodeSetUserData", 1)) { return 1; }

    // Integration method order
    if (impl)
    {
      flag = ARKodeSetOrder(arkode_mem, uopts.order);
      if (check_flag(&flag, "ARKodeSetOrder", 1)) { return 1; }
    }
    if (expl) // order: -2,-3,-4 indicate use of SSPRK methods
    {
      ARKodeButcherTable B = nullptr;
      if (uopts.order == -2)
      {
        B          = ARKodeButcherTable_Alloc(2, SUNTRUE);
        B->q       = 2;
        B->p       = 1;
        B->A[1][0] = SUN_RCONST(1.0);
        B->b[0]    = SUN_RCONST(0.5);
        B->b[1]    = SUN_RCONST(0.5);
        B->d[0]    = SUN_RCONST(0.694021459207626);
        B->d[1]    = SUN_RCONST(1.0) - SUN_RCONST(0.694021459207626);
        B->c[1]    = SUN_RCONST(1.0);
      }
      else if (uopts.order == -3)
      {
        B          = ARKodeButcherTable_Alloc(3, SUNTRUE);
        B->q       = 2;
        B->p       = 1;
        B->A[1][0] = SUN_RCONST(0.5);
        B->A[2][0] = SUN_RCONST(0.5);
        B->A[2][1] = SUN_RCONST(0.5);
        B->b[0]    = SUN_RCONST(1.0) / SUN_RCONST(3.0);
        B->b[1]    = SUN_RCONST(1.0) / SUN_RCONST(3.0);
        B->b[2]    = SUN_RCONST(1.0) / SUN_RCONST(3.0);
        B->d[0]    = SUN_RCONST(4.0) / SUN_RCONST(9.0);
        B->d[1]    = SUN_RCONST(1.0) / SUN_RCONST(3.0);
        B->d[2]    = SUN_RCONST(2.0) / SUN_RCONST(9.0);
        B->c[1]    = SUN_RCONST(0.5);
        B->c[2]    = SUN_RCONST(1.0);
      }
      else if (uopts.order == -4)
      {
        B                       = ARKodeButcherTable_Alloc(4, SUNTRUE);
        const sunrealtype third = SUN_RCONST(1.0) / SUN_RCONST(3.0);
        B->q                    = 2;
        B->p                    = 1;
        B->A[1][0]              = third;
        B->A[2][0]              = third;
        B->A[2][1]              = third;
        B->A[3][0]              = third;
        B->A[3][1]              = third;
        B->A[3][2]              = third;
        B->b[0]                 = SUN_RCONST(0.25);
        B->b[1]                 = SUN_RCONST(0.25);
        B->b[2]                 = SUN_RCONST(0.25);
        B->b[3]                 = SUN_RCONST(0.25);
        B->d[0]                 = SUN_RCONST(5.0) / SUN_RCONST(16.0);
        B->d[1]                 = SUN_RCONST(1.0) / SUN_RCONST(4.0);
        B->d[2]                 = SUN_RCONST(1.0) / SUN_RCONST(4.0);
        B->d[3]                 = SUN_RCONST(3.0) / SUN_RCONST(16.0);
        B->c[1]                 = third;
        B->c[2]                 = SUN_RCONST(2.0) * third;
        B->c[3]                 = SUN_RCONST(1.0);
      }
      if (B != nullptr)
      {
        flag = ARKStepSetTables(arkode_mem, B->q, B->p, nullptr, B);
        if (check_flag(&flag, "ARKodeSetOrder", 1)) { return 1; }
      }
      else
      {
        flag = ARKodeSetOrder(arkode_mem, uopts.order);
        if (check_flag(&flag, "ARKodeSetOrder", 1)) { return 1; }
      }
    }

    if (impl)
    {
      // Attach linear solver
      flag = ARKodeSetLinearSolver(arkode_mem, LS, A);
      if (check_flag(&flag, "ARKodeSetLinearSolver", 1)) { return 1; }

      if (uopts.preconditioning)
      {
        // Attach preconditioner
        flag = ARKodeSetPreconditioner(arkode_mem, PSetup, PSolve);
        if (check_flag(&flag, "ARKodeSetPreconditioner", 1)) { return 1; }

        // Set linear solver setup frequency (update preconditioner)
        flag = ARKodeSetLSetupFrequency(arkode_mem, uopts.msbp);
        if (check_flag(&flag, "ARKodeSetLSetupFrequency", 1)) { return 1; }
      }

      // Set linear solver tolerance factor
      flag = ARKodeSetEpsLin(arkode_mem, uopts.epslin);
      if (check_flag(&flag, "ARKodeSetEpsLin", 1)) { return 1; }

      // Specify linearly implicit non-time-dependent RHS
      if (uopts.linear)
      {
        flag = ARKodeSetLinear(arkode_mem, 0);
        if (check_flag(&flag, "ARKodeSetLinear", 1)) { return 1; }
      }
    }

    // Configure expl STS solver
    SUNDomEigEstimator DEE = nullptr;
    if (sts)
    {
      // Select LSRK method
      ARKODE_LSRKMethodType type =
        (uopts.integrator == "rkc") ? ARKODE_LSRK_RKC_2 : ARKODE_LSRK_RKL_2;
      flag = LSRKStepSetSTSMethod(arkode_mem, type);
      if (check_flag(&flag, "LSRKStepSetSTSMethod", 1)) { return 1; }

      // Dominant eigenvalue estimation
      if (uopts.internaleig)
      {
        DEE = SUNDomEigEst_Power(u, 100, 0.01, ctx);
        if (check_flag(DEE, "SUNDomEigEst_Power", 0)) { return 1; }

        flag = LSRKStepSetDomEigEstimator(arkode_mem, DEE);
        if (check_flag(&flag, "LSRKStepSetDomEigEstimator", 1)) { return 1; }
      }
      else
      {
        flag = LSRKStepSetDomEigFn(arkode_mem, dom_eig);
        if (check_flag(&flag, "LSRKStepSetDomEigFn", 1)) { return 1; }
      }
    }

    // Set fixed step size or adaptivity method
    SUNAdaptController C = nullptr;
    if (uopts.hfixed > ZERO)
    {
      flag = ARKodeSetFixedStep(arkode_mem, uopts.hfixed);
      if (check_flag(&flag, "ARKodeSetFixedStep", 1)) { return 1; }
    }
    else
    {
      switch (uopts.controller)
      {
      case (ARK_ADAPT_PID): C = SUNAdaptController_PID(ctx); break;
      case (ARK_ADAPT_PI): C = SUNAdaptController_PI(ctx); break;
      case (ARK_ADAPT_I): C = SUNAdaptController_I(ctx); break;
      case (ARK_ADAPT_EXP_GUS): C = SUNAdaptController_ExpGus(ctx); break;
      case (ARK_ADAPT_IMP_GUS): C = SUNAdaptController_ImpGus(ctx); break;
      case (ARK_ADAPT_IMEX_GUS): C = SUNAdaptController_ImExGus(ctx); break;
      }
      flag = ARKodeSetAdaptController(arkode_mem, C);
      if (check_flag(&flag, "ARKodeSetAdaptController", 1)) { return 1; }
    }

    // Set max steps between outputs
    flag = ARKodeSetMaxNumSteps(arkode_mem, uopts.maxsteps);
    if (check_flag(&flag, "ARKodeSetMaxNumSteps", 1)) { return 1; }

    // Set stopping time
    flag = ARKodeSetStopTime(arkode_mem, udata.tf);
    if (check_flag(&flag, "ARKodeSetStopTime", 1)) { return 1; }

    // Create and configure reference integrator
    if (uopts.error)
    {
      arkref_mem = LSRKStepCreateSTS(diffusion, ZERO, u, ctx);
      if (check_flag((void*)arkref_mem, "LSRKStepCreateSTS", 0)) { return 1; }
      flag = ARKodeSStolerances(arkref_mem, 1.e-12, uopts.atol);
      if (check_flag(&flag, "ARKodeSStolerances", 1)) { return 1; }
      flag = ARKodeSetUserData(arkref_mem, (void*)&udata);
      if (check_flag(&flag, "ARKodeSetUserData", 1)) { return 1; }
      flag = LSRKStepSetDomEigFn(arkref_mem, dom_eig);
      if (check_flag(&flag, "LSRKStepSetDomEigFn", 1)) { return 1; }
      flag = ARKodeSetMaxNumSteps(arkref_mem, 100000 * uopts.maxsteps);
      if (check_flag(&flag, "ARKodeSetMaxNumSteps", 1)) { return 1; }
    }

    // -----------------------
    // Loop over output times
    // -----------------------

    // Optionally run in one step mode for a fixed number of time steps (helpful
    // for debugging)
    int stepmode = ARK_NORMAL;

    if (uopts.onestep)
    {
      uout.nout = uopts.onestep;
      stepmode  = ARK_ONE_STEP;
    }

    sunrealtype t      = ZERO;
    sunrealtype t2     = ZERO;
    sunrealtype dTout  = udata.tf / uout.nout;
    sunrealtype tout   = dTout;
    sunrealtype errtot = ZERO;
    double tstart      = 0.0;
    double simtime     = 0.0;

    // Initial output
    flag = uout.open(&udata);
    if (check_flag(&flag, "UserOutput::open", 1)) { return 1; }

    flag = uout.write(t, u, &udata);
    if (check_flag(&flag, "UserOutput::write", 1)) { return 1; }

    for (int iout = 0; iout < uout.nout; iout++)
    {
      SUNDIALS_MARK_BEGIN(prof, "Evolve");

      // Evolve in time, recording elapsed runtime
      tstart = MPI_Wtime();
      if (uopts.error)
      {
        flag = ARKodeSetStopTime(arkode_mem, tout);
        if (check_flag(&flag, "ARKodeSetStopTime", 1)) { break; }
      }
      flag = ARKodeEvolve(arkode_mem, tout, u, &t, stepmode);
      if (check_flag(&flag, "ARKodeEvolve", 1)) { return 1; }
      simtime += MPI_Wtime() - tstart;
      SUNDIALS_MARK_END(prof, "Evolve");

      // Evolve reference solution
      if (uopts.error)
      {
        flag = ARKodeSetStopTime(arkref_mem, t);
        if (check_flag(&flag, "ARKodeSetStopTime", 1)) { break; }
        flag = ARKodeEvolve(arkref_mem, tout, uref, &t2, ARK_NORMAL);
        if (check_flag(&flag, "ARKodeEvolve", 1)) { break; }
      }

      // Output solution
      flag = uout.write(t, u, &udata);
      if (check_flag(&flag, "UserOutput::write", 1)) { return 1; }

      // Accumulate temporal error
      if (uopts.error)
      {
        N_VLinearSum(1.0, uref, -1.0, u, uerr);
        errtot = std::max(errtot, N_VMaxNorm(uerr) / N_VMaxNorm(uref));
      }

      // Update output time
      tout += dTout;
      tout = (tout > udata.tf) ? udata.tf : tout;
    }

    // Close output
    flag = uout.close(&udata);
    if (check_flag(&flag, "UserOutput::close", 1)) { return 1; }

    // --------------
    // Final outputs
    // --------------

    // Print final integrator stats
    if (outproc)
    {
      if (uopts.error) cout << "Maximum relative error = " << errtot << endl;
      cout << "Total simulation time = " << simtime << endl << endl;
      cout << "Final integrator statistics:" << endl;
      flag = ARKodePrintAllStats(arkode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
      if (check_flag(&flag, "ARKodePrintAllStats", 1)) { return 1; }

#if HYPRE_RELEASE_NUMBER >= 22000
      if (impl && uopts.preconditioning)
      {
        cout << "Total PFMG iterations        = " << udata.pfmg_its << endl;
      }
#endif
    }

    // ---------
    // Clean up
    // ---------

    // Free MPI Cartesian communicator
    MPI_Comm_free(&(udata.comm_c));
    (void)SUNAdaptController_Destroy(C); // Free timestep adaptivity controller

    ARKodeFree(&arkode_mem);
    if (impl)
    {
      SUNLinSolFree(LS);

      // Finalize hypre if v2.20.0 or newer
#if HYPRE_RELEASE_NUMBER >= 22000
      if (uopts.preconditioning)
      {
        flag = HYPRE_Finalize();
        if (check_flag(&flag, "HYPRE_Finalize", 1)) { return 1; }
      }
#endif
    }

    // Free vectors
    N_VDestroy(u);
  }
  // Close scope so objects are destroyed before MPI_Finalize

  SUNContext_Free(&ctx);

  // Finalize MPI
  flag = MPI_Finalize();
  return 0;
}

// -----------------------------------------------------------------------------
// Dominant eigenvalue estimation function
// -----------------------------------------------------------------------------

static int dom_eig(sunrealtype t, N_Vector y, N_Vector fn, sunrealtype* lambdaR,
                   sunrealtype* lambdaI, void* user_data, N_Vector temp1,
                   N_Vector temp2, N_Vector temp3)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  // Fill in spectral radius value
  *lambdaR = -SUN_RCONST(8.0) * std::max(udata->kx / udata->dx / udata->dx,
                                         udata->ky / udata->dy / udata->dy);
  *lambdaI = SUN_RCONST(0.0);

  // return with success
  return 0;
}

// -----------------------------------------------------------------------------
// UserOptions Helper functions
// -----------------------------------------------------------------------------

int UserOptions::parse_args(vector<string>& args, bool outproc)
{
  vector<string>::iterator it;

  it = find(args.begin(), args.end(), "--help");
  if (it != args.end())
  {
    if (outproc) { help(); }
    return 0;
  }

  it = find(args.begin(), args.end(), "--integrator");
  if (it != args.end())
  {
    integrator = *(it + 1);
    args.erase(it, it + 2);
  }

  it = find(args.begin(), args.end(), "--rtol");
  if (it != args.end())
  {
    rtol = stod(*(it + 1));
    args.erase(it, it + 2);
  }

  it = find(args.begin(), args.end(), "--atol");
  if (it != args.end())
  {
    atol = stod(*(it + 1));
    args.erase(it, it + 2);
  }

  it = find(args.begin(), args.end(), "--fixedstep");
  if (it != args.end())
  {
    hfixed = stod(*(it + 1));
    args.erase(it, it + 2);
  }

  it = find(args.begin(), args.end(), "--order");
  if (it != args.end())
  {
    order = stoi(*(it + 1));
    args.erase(it, it + 2);
  }

  it = find(args.begin(), args.end(), "--controller");
  if (it != args.end())
  {
    controller = stoi(*(it + 1));
    args.erase(it, it + 2);
  }

  it = find(args.begin(), args.end(), "--maxsteps");
  if (it != args.end())
  {
    maxsteps = stoi(*(it + 1));
    args.erase(it, it + 2);
  }

  it = find(args.begin(), args.end(), "--onestep");
  if (it != args.end())
  {
    onestep = stoi(*(it + 1));
    args.erase(it, it + 2);
  }

  it = find(args.begin(), args.end(), "--error");
  if (it != args.end())
  {
    error = true;
    args.erase(it);
  }

  it = find(args.begin(), args.end(), "--nonlinear");
  if (it != args.end())
  {
    linear = false;
    args.erase(it);
  }

  it = find(args.begin(), args.end(), "--ls");
  if (it != args.end())
  {
    ls = *(it + 1);
    args.erase(it, it + 2);
  }

  it = find(args.begin(), args.end(), "--lsinfo");
  if (it != args.end())
  {
    lsinfo = true;
    args.erase(it);
  }

  it = find(args.begin(), args.end(), "--liniters");
  if (it != args.end())
  {
    liniters = stoi(*(it + 1));
    args.erase(it, it + 2);
  }

  it = find(args.begin(), args.end(), "--msbp");
  if (it != args.end())
  {
    msbp = stoi(*(it + 1));
    args.erase(it, it + 2);
  }

  it = find(args.begin(), args.end(), "--epslin");
  if (it != args.end())
  {
    epslin = stod(*(it + 1));
    args.erase(it, it + 2);
  }

  it = find(args.begin(), args.end(), "--noprec");
  if (it != args.end())
  {
    preconditioning = false;
    args.erase(it);
  }

  it = find(args.begin(), args.end(), "--internaleig");
  if (it != args.end())
  {
    internaleig = true;
    args.erase(it);
  }

  return 0;
}

// Print command line options
void UserOptions::help()
{
  cout << endl;
  cout << "Integrator command line options:" << endl;
  cout << "  --integrator <dirk|erk|rkc|rkl> : time integration method" << endl;
  cout << "  --rtol <rtol>        : relative tolerance" << endl;
  cout << "  --atol <atol>        : absolute tolerance" << endl;
  cout << "  --onestep            : evolve in one-step mode" << endl;
  cout << "  --controller <ctr>   : time step adaptivity controller" << endl;
  cout << "                         PID = " << ARK_ADAPT_PID << endl;
  cout << "                         PI = " << ARK_ADAPT_PI << endl;
  cout << "                         I = " << ARK_ADAPT_I << endl;
  cout << "                         ExpGus = " << ARK_ADAPT_EXP_GUS << endl;
  cout << "                         ImpGus = " << ARK_ADAPT_IMP_GUS << endl;
  cout << "                         ImExGus = " << ARK_ADAPT_IMEX_GUS << endl;
  cout << "  --maxsteps <nstep>   : maximum allowed number of time steps" << endl;
  cout << "  --fixedstep <step>   : fixed step size to use" << endl;
  cout << "  --error              : compute reference solution to compare error"
       << endl;
  cout << endl;
  cout << "ERK solver command line options:" << endl;
  cout << "  --order <ord>        : method order" << endl;
  cout << endl;
  cout << "DIRK solver command line options:" << endl;
  cout << "  --order <ord>        : method order" << endl;
  cout << "  --nonlinear          : disable linearly implicit flag" << endl;
  cout << "  --ls <cg|gmres>      : linear solver" << endl;
  cout << "  --lsinfo             : output residual history" << endl;
  cout << "  --liniters <iters>   : max number of iterations" << endl;
  cout << "  --epslin <factor>    : linear tolerance factor" << endl;
  cout << "  --noprec             : disable preconditioner" << endl;
  cout << "  --msbp <steps>       : max steps between prec setups" << endl;
  cout << "RKC/RKL solver command line options:" << endl;
  cout << "  --internaleig        : estimate dominant eigenvalue internally" << endl;
  cout << endl;
  cout << endl;
}

// Print user options
void UserOptions::print()
{
  bool impl = (integrator == "dirk");
  bool expl = (integrator == "erk");
  bool sts  = ((integrator == "rkc") || (integrator == "rkl"));

  cout << endl;
  cout << " Integrator: " << integrator << endl;
  cout << " --------------------------------- " << endl;
  cout << " rtol        = " << rtol << endl;
  cout << " atol        = " << atol << endl;
  cout << " controller  = " << controller << endl;
  cout << " hfixed      = " << hfixed << endl;
  cout << " max steps   = " << maxsteps << endl;
  if (sts)
  {
    if (internaleig)  cout << " internal eigenvalue estimation" << endl;
    else              cout << " user-provided dominant eigenvalue" << endl;
  }
  if (impl || expl)
  {
    cout << " order       = " << order << endl;
    if (impl)
    {
      if (linear) cout << " linearly implicit RHS" << endl;
      else cout << " nonlinearly implicit RHS" << endl;
    }
    cout << " --------------------------------- " << endl;
    if (impl)
    {
      cout << endl;
      cout << " Linear solver options:" << endl;
      cout << " --------------------------------- " << endl;
      cout << " LS       = " << ls << endl;
      cout << " precond  = " << preconditioning << endl;
#ifdef USE_HYPRE
      cout << " HYPRE preconditioner" << endl;
#else
      cout << " Jacobi preconditioner" << endl;
#endif
      cout << " LS info  = " << lsinfo << endl;
      cout << " LS iters = " << liniters << endl;
      cout << " msbp     = " << msbp << endl;
      cout << " epslin   = " << epslin << endl;
      cout << " --------------------------------- " << endl;
    }
  }
}

//---- end of file ----
