/* -----------------------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ UMBC
 * Based on the 2D advection-diffusion-reaction example in PIROCK.
 * -----------------------------------------------------------------------------
 * This example simulates the 2D diffusion-advection-reaction equation,
 *
 *   u_t = Df*(u_xx + u_yy) + cux*u_x + cuy*u_y + A + u^2 * v - (B + 1)*u
 *   v_t = Df*(v_xx + v_yy) + cvx*v_x + cvy*v_y + B*u - u^2 * v
 *
 * where u and v represent the concentrations of chemical species,
 * The default parameters set up a nonstiff advection-diffusion-reaction problem,
 * where Df = 0.01, cux=-0.5, cuy=1.0, cvx=0.4, cvy=0.7, A = 1.3, and B = 1.0.
 *
 * The problem is evolved for t in [0, 1] and (x,y) in [0, 1]^2, with initial
 * conditions given by
 *
 *   u(x,y,0) = 22 * y * (1 - y)^(1.5)
 *   v(x,y,0) = 27 * x * (1 - x)^(1.5)
 *
 * and periodic boundary conditions i.e.,
 *
 *   u(x+1, y, t) = u(x , y, t) = u(x, y+1, t) = 0.
 *
 * The system is advanced in time using one of the following approaches based on
 * the --integrator <int> flag value. The following options are available:
 *
 *   0. An explicit Runge-Kutta method with ERKStep.
 *
 *   1. An IMEX Runge-Kutta method with ARKStep.  Diffusion is treated implicitly,
 *      while advection is evolved explicitly in time.  Reactions can be treated
 *      explicitly or implicitly, based on the --implicit-reaction flag.
 *
 *      Note: the advection operator can be disabled using the --no-advection flag.
 *
 *   2. An extended super-time-stepping method that combines MRIStep and LSRKStep.
 *      Here, diffusion is treated explicitly using a STS method, and advection
 *      is treated explicitly using the ExtSTS method.  Reactions can be treated
 *      explicitly or implicitly, based on the --implicit-reaction flag.
 *
 *      Note: the advection operator can be disabled using the --no-advection flag.
 *
 *   3. A second-order Strang operator splitting method that combines LSRKStep and
 *      ARKStep, where diffusion is treated explicitly using LSRKStep, and advection
 *      is treated explicitly using ARKStep.  Reactions can be treated explicitly or
 *      implicitly, based on the --implicit-reaction flag. This option must be used
 *      with fixed time step sizes.  ARKStep will always use the ARS(2,2,2) table
 *      (whether it is run in ERK, DIRK, or ARK mode).
 *
 *      Note: the advection operator can be disabled using the --no-advection flag.
 *
 * Several command line options are available to change the problem parameters
 * and integrator settings. Use the flag --help for more information.
 * ---------------------------------------------------------------------------*/

#include "advection_diffusion_reaction_2d.hpp"

int main(int argc, char* argv[])
{
  // SUNDIALS context object for this simulation
  sundials::Context ctx;

  // -----------------
  // Setup the problem
  // -----------------

  UserData udata;
  UserOptions uopts;

  vector<string> args(argv + 1, argv + argc);

  int flag = ReadInputs(args, udata, uopts, ctx);
  if (flag < 0)
  {
    cerr << "ERROR: ReadInputs returned " << flag << endl;
    return 1;
  }
  if (flag > 0) { return 0; }

  flag = PrintSetup(udata, uopts);
  if (check_flag(flag, "PrintSetup")) { return 1; }

  // Create state vector and set initial condition
  N_Vector y = N_VNew_Serial(udata.neq, ctx);
  if (check_ptr(y, "N_VNew_Serial")) { return 1; }

  flag = SetIC(y, udata);
  if (check_flag(flag, "SetIC")) { return 1; }

  // Create reference and error vectors
  N_Vector yref = nullptr;
  N_Vector yerr = nullptr;
  if (uopts.calc_error)
  {
    yref = N_VNew_Serial(udata.neq, ctx);
    if (check_ptr(yref, "N_VNew_Serial")) { return 1; }
    yerr = N_VNew_Serial(udata.neq, ctx);
    if (check_ptr(yerr, "N_VNew_Serial")) { return 1; }
    N_VScale(1.0, y, yref);
    N_VScale(0.0, y, yerr);
  }

  // --------------------
  // Setup the integrator
  // --------------------

  // ARKODE memory structures
  void* arkode_mem = nullptr;
  void* arkref_mem = nullptr;

  // Matrix and linear solver for ImEx, Strang, and ExtSTS integrators
  SUNMatrix A           = nullptr;
  SUNLinearSolver LS    = nullptr;
  SUNLinearSolver LSref = nullptr;

  // STS integrator for ExtSTS method
  MRIStepInnerStepper sts_mem = nullptr;

  // LSRKStep and ARKStep integrators for Strang splitting method
  SUNStepper steppers[2];
  void* lsrkstep_mem = nullptr;
  void* arkstep_mem = nullptr;

  // Create integrator
  switch (uopts.integrator)
  {
  case (0): flag = SetupERK(ctx, udata, uopts, y, &arkode_mem); break;
  case (1): flag = SetupARK(ctx, udata, uopts, y, &LS, &arkode_mem); break;
  case (2): flag = SetupExtSTS(ctx, udata, uopts, y, &A, &LS, &sts_mem, &arkode_mem); break;
  case (3): flag = SetupStrang(ctx, udata, uopts, y, &A, &LS, steppers, &lsrkstep_mem,
                               &arkstep_mem, &arkode_mem); break;
  default: flag = -1;
  }
  if (check_flag(flag, "Integrator setup")) { return 1; }

  // Create reference solver (4th-order ARK with tighter relative tolerance)
  if (uopts.calc_error)
  {
    flag = SetupReference(ctx, udata, uopts, yref, &LSref, &arkref_mem);
    if (check_flag(flag, "Reference solver setup")) { return 1; }
    flag = ARKodeSStolerances(arkref_mem, uopts.rtol/100, uopts.atol);
    if (check_flag(flag, "ARKodeSStolerances")) { return 1; }
    flag = ARKodeSetOrder(arkref_mem, 4);
    if (check_flag(flag, "ARKodeSetOrder")) { return 1; }
  }

  // ----------------------
  // Evolve problem in time
  // ----------------------

  // Initial time
  sunrealtype t  = ZERO;

  // Initial output
  flag = OpenOutput(udata, uopts);
  if (check_flag(flag, "OpenOutput")) { return 1; }

  // Timers
  sunrealtype ref_time = 0.0;
  sunrealtype solve_time = 0.0;

  // Either perform normal or one-step evolution, based on calc_error flag
  if (uopts.calc_error)      // one step mode
  {
    sunrealtype t2 = ZERO;
    sunrealtype total_error = ZERO;
    uopts.nout = 0;

    // Set stop time for solution end
    flag = ARKodeSetStopTime(arkode_mem, udata.tf);
    if (check_flag(flag, "ARKodeSetStopTime")) { return 1; }

    // Loop over internal time steps
    while (udata.tf - t > std::sqrt(std::numeric_limits<sunrealtype>::epsilon()))
    {
      // Archive current solution in reference vector
      t2 = t;
      N_VScale(1.0, y, yref);

      // Call "test" solver in one-step mode
      auto solver_start = chrono::high_resolution_clock::now();
      flag = ARKodeEvolve(arkode_mem, udata.tf, y, &t, ARK_ONE_STEP);
      if (check_flag(flag, "ARKodeEvolve")) { return 1; }
      auto solver_end = chrono::high_resolution_clock::now();
      solve_time += chrono::duration<sunrealtype>(solver_end - solver_start).count();

      // Have the reference solver take an identical step (with same initial condition), and accumulate error
      flag = ARKodeReset(arkref_mem, t2, yref);
      if (check_flag(flag, "ARKodeReset")) { return 1; }
      flag = ARKodeSetStopTime(arkref_mem, t);
      if (check_flag(flag, "ARKodeSetStopTime")) { return 1; }
      auto reference_start = chrono::high_resolution_clock::now();
      flag = ARKodeEvolve(arkref_mem, t, yref, &t2, ARK_NORMAL);
      if (check_flag(flag, "ARKodeEvolve (ref)")) { return 1; }
      auto reference_end = chrono::high_resolution_clock::now();
      ref_time += chrono::duration<sunrealtype>(reference_end - reference_start).count();
      N_VLinearSum(1.0, y, -1.0, yref, yerr);
      sunrealtype curr_error = N_VDotProd(yerr, yerr);
      total_error += curr_error;
      uopts.nout++;
      long int nst_ref;
      flag = ARKodeGetNumSteps(arkref_mem, &nst_ref);
      if (check_flag(flag, "ARKodeGetNumSteps")) { return -1; }
      std::cout << " t = " << t << ", ||err_curr|| = " << std::sqrt(curr_error / udata.nx / 3)
                << ", ||err_tot|| = " << std::sqrt(total_error / uopts.nout / udata.nx / 3)
                << ", reference steps = " << nst_ref << std::endl;
    }
  }
  else                       // normal mode
  {
    sunrealtype dTout = udata.tf / uopts.nout;
    sunrealtype tout  = dTout;
    for (int iout = 0; iout < uopts.nout; iout++)
    {
      if (uopts.output == 3)
      {
        // Stop at output time (do not interpolate output)
        flag = ARKodeSetStopTime(arkode_mem, tout);
        if (check_flag(flag, "ARKodeSetStopTime")) { return 1; }
      }

      //   Advance in time
      auto solver_start = chrono::high_resolution_clock::now();
      flag = ARKodeEvolve(arkode_mem, tout, y, &t, ARK_NORMAL);
      if (check_flag(flag, "ARKodeEvolve")) { return 1; }
      auto solver_end = chrono::high_resolution_clock::now();
      solve_time += chrono::duration<sunrealtype>(solver_end - solver_start).count();

      // Update output time
      tout += dTout;
      tout = (tout > udata.tf) ? udata.tf : tout;
    }
  }

  // Output solution
  flag = WriteOutput(udata.tf, y, udata, uopts);
  if (check_flag(flag, "WriteOutput")) { return 1; }

  // Close output
  flag = CloseOutput(uopts);
  if (check_flag(flag, "CloseOutput")) { return 1; }

  // Write reference solution to disk (if applicable)
  flag = WriteSolution(t, yref, udata, uopts);
  if (check_flag(flag, "WriteSolution")) { return 1; }

  // ------------
  // Output stats
  // ------------

  if (uopts.output)
  {
    cout << "Final integrator statistics:" << endl;
    cout << "  Total solve time   = " << setprecision(2) << solve_time << endl;
    cout << "  Reference time     = " << setprecision(2) << ref_time << endl;
    switch (uopts.integrator)
    {
    case (0): flag = OutputStatsERK(arkode_mem, udata); break;
    case (1): flag = OutputStatsARK(arkode_mem, udata); break;
    case (2): flag = OutputStatsExtSTS(arkode_mem, sts_mem, udata); break;
    case (3): flag = OutputStatsStrang(arkode_mem, arkstep_mem, lsrkstep_mem, udata); break;
    default: flag = -1;
    }
    if (check_flag(flag, "OutputStats")) { return 1; }
  }

  // --------
  // Clean up
  // --------

  switch (uopts.integrator)
  {
  case (0): ARKodeFree(&arkode_mem); break;
  case (1): ARKodeFree(&arkode_mem); break;
  case (2):
  {
    void* inner_content = nullptr;
    MRIStepInnerStepper_GetContent(sts_mem, &inner_content);
    STSInnerStepperContent* content = (STSInnerStepperContent*)inner_content;
    ARKodeFree(&(content->sts_arkode_mem));
    free(content);
    MRIStepInnerStepper_Free(&sts_mem);
    ARKodeFree(&arkode_mem);
    break;
  }
  case (3):
  {
    ARKodeFree(&lsrkstep_mem);
    ARKodeFree(&arkstep_mem);
    SUNStepper_Destroy(&steppers[0]);
    SUNStepper_Destroy(&steppers[1]);
    ARKodeFree(&arkode_mem);
    break;
  }
  }
  if (uopts.calc_error)
  {
    ARKodeFree(&arkref_mem);
    N_VDestroy(yref);
    N_VDestroy(yerr);
    SUNLinSolFree(LSref);
  }

  N_VDestroy(y);
  SUNMatDestroy(A);
  SUNLinSolFree(LS);

  return 0;
}

// -----------------------------------------------------------------------------
// Setup the integrator
// -----------------------------------------------------------------------------

int SetupERK(SUNContext ctx, UserData& udata, UserOptions& uopts, N_Vector y,
             void** arkode_mem)
{
  // Problem configuration
  ARKRhsFn f_RHS; // explicit RHS function

  if (udata.advection)
  {
    // Explicit -- advection-diffusion-reaction
    f_RHS = f_adv_diff_react;
  }
  else
  {
    // Explicit -- diffusion-reaction
    f_RHS = f_diff_react;
  }

  // Create ERKStep memory
  *arkode_mem = ERKStepCreate(f_RHS, ZERO, y, ctx);
  if (check_ptr(arkode_mem, "ERKStepCreate")) { return 1; }

  // Specify tolerances
  int flag = ARKodeSStolerances(*arkode_mem, uopts.rtol, uopts.atol);
  if (check_flag(flag, "ARKodeSStolerances")) { return 1; }

  // Attach user data
  flag = ARKodeSetUserData(*arkode_mem, &udata);
  if (check_flag(flag, "ARKodeSetUserData")) { return 1; }

  // Select method order
  flag = ARKodeSetOrder(*arkode_mem, uopts.order);
  if (check_flag(flag, "ARKodeSetOrder")) { return 1; }

  // Set fixed step size
  if (uopts.fixed_h > ZERO)
  {
    flag = ARKodeSetFixedStep(*arkode_mem, uopts.fixed_h);
    if (check_flag(flag, "ARKodeSetFixedStep")) { return 1; }
  }

  // Set max steps between outputs
  flag = ARKodeSetMaxNumSteps(*arkode_mem, uopts.maxsteps);
  if (check_flag(flag, "ARKodeSetMaxNumSteps")) { return 1; }

  // Set stopping time
  flag = ARKodeSetStopTime(*arkode_mem, udata.tf);
  if (check_flag(flag, "ARKodeSetStopTime")) { return 1; }

  return 0;
}

int SetupARK(SUNContext ctx, UserData& udata, UserOptions& uopts, N_Vector y,
             SUNLinearSolver* LS, void** arkode_mem)
{
  // Problem configuration
  ARKRhsFn fe_RHS; // explicit RHS function
  ARKRhsFn fi_RHS; // implicit RHS function

  // advection + diffusion + implicit reaction
  if (udata.impl_reaction && udata.advection)
  {
    fe_RHS = f_advection;
    fi_RHS = f_diff_react;
  }
  // advection + diffusion + explicit reaction
  else if (!udata.impl_reaction && udata.advection)
  {
    fe_RHS = f_adv_react;
    fi_RHS = f_diffusion;
  }
  // diffusion + implicit reaction
  else if (udata.impl_reaction && !udata.advection)
  {
    fe_RHS = nullptr;
    fi_RHS = f_diff_react;
  }
  // diffusion + explicit reaction
  else if (!udata.impl_reaction && !udata.advection)
  {
    fe_RHS = f_reaction;
    fi_RHS = f_diffusion;
  }
  else
  {
    cerr << "ERROR: Invalid problem configuration" << endl;
    return -1;
  }

  // Create ARKStep memory
  *arkode_mem = ARKStepCreate(fe_RHS, fi_RHS, ZERO, y, ctx);
  if (check_ptr(arkode_mem, "ARKStepCreate")) { return 1; }

  // Specify tolerances
  int flag = ARKodeSStolerances(*arkode_mem, uopts.rtol, uopts.atol);
  if (check_flag(flag, "ARKodeSStolerances")) { return 1; }

  // Attach user data
  flag = ARKodeSetUserData(*arkode_mem, &udata);
  if (check_flag(flag, "ARKodeSetUserData")) { return 1; }

  // Create and attach linear solver (and reaction preconditioner, if applicable)
  if (udata.impl_reaction)
  {
    *LS = SUNLinSol_SPGMR(y, SUN_PREC_RIGHT, uopts.maxl, ctx);
    if (check_ptr(*LS, "SUNLinSol_SPGMR")) { return 1; }
    flag = ARKodeSetLinearSolver(*arkode_mem, *LS, nullptr);
    if (check_flag(flag, "ARKodeSetLinearSolver")) { return 1; }
    flag = ARKBBDPrecInit(*arkode_mem, udata.neq, 2, 2, 2, 2,
                           ZERO, floc_reaction, nullptr);
    if (check_flag(flag, "ARKBBDPrecInit")) { return 1; }
  }
  else
  {
    *LS = SUNLinSol_SPGMR(y, SUN_PREC_NONE, uopts.maxl, ctx);
    if (check_ptr(*LS, "SUNLinSol_SPGMR")) { return 1; }

    flag = ARKodeSetLinearSolver(*arkode_mem, *LS, nullptr);
    if (check_flag(flag, "ARKodeSetLinearSolver")) { return 1; }
  }

  // Tighten implicit solver tolerances and allow more Newton iterations
  flag = ARKodeSetMaxNonlinIters(*arkode_mem, uopts.maxnewt);
  if (check_flag(flag, "ARKodeSetMaxNonlinIters")) { return 1; }
  flag = ARKodeSetNonlinConvCoef(*arkode_mem, uopts.nlscoef);
  if (check_flag(flag, "ARKodeSetNonlinConvCoef")) { return 1; }
  flag = ARKodeSetEpsLin(*arkode_mem, uopts.epslin);
  if (check_flag(flag, "ARKodeSetEpsLin")) { return 1; }

  // Use "deduce implicit RHS" option
  flag = ARKodeSetDeduceImplicitRhs(*arkode_mem, SUNTRUE);
  if (check_flag(flag, "ARKodeSetDeduceImplicitRhs")) { return 1; }

  // Set the predictor method
  flag = ARKodeSetPredictorMethod(*arkode_mem, uopts.predictor);
  if (check_flag(flag, "ARKodeSetPredictorMethod")) { return 1; }

  if (uopts.linear)
  {
    // Specify linearly implicit non-time-dependent RHS
    flag = ARKodeSetLinear(*arkode_mem, SUNFALSE);
    if (check_flag(flag, "ARKodeSetLinear")) { return 1; }
  }

  if (uopts.table_id > 0)
  {
    // Set the RK tables
    ARKodeButcherTable Be = nullptr;
    ARKodeButcherTable Bi = nullptr;
    if (uopts.table_id == 1) // ARS(2,2,2)
    {
      const sunrealtype gamma = (SUN_RCONST(2.0) - SUNRsqrt(SUN_RCONST(2.0))) /
                                SUN_RCONST(2.0);
      const sunrealtype delta = SUN_RCONST(1.0) -
                                SUN_RCONST(1.0) / (SUN_RCONST(2.0) * gamma);
      if (fe_RHS != nullptr)
      {
        Be                      = ARKodeButcherTable_Alloc(3, SUNTRUE);
        Be->c[1] = gamma;
        Be->c[2] = SUN_RCONST(1.0);
        Be->A[1][0] = gamma;
        Be->A[2][0] = delta;
        Be->A[2][1] = SUN_RCONST(1.0)-delta;
        Be->b[0] = delta;
        Be->b[1] = SUN_RCONST(1.0)-delta;
        Be->d[1] = SUN_RCONST(3.0)/SUN_RCONST(5.0);
        Be->d[2] = SUN_RCONST(2.0)/SUN_RCONST(5.0);
        Be->q = 2;
        Be->p = 1;
      }
      if (fi_RHS != nullptr)
      {
        Bi = ARKodeButcherTable_Alloc(3, SUNTRUE);
        Bi->c[1] = gamma;
        Bi->c[2] = SUN_RCONST(1.0);;
        Bi->A[1][1] = gamma;
        Bi->A[2][1] = SUN_RCONST(1.0) - gamma;
        Bi->A[2][2] = gamma;
        Bi->b[1] = SUN_RCONST(1.0)-gamma;
        Bi->b[2] = gamma;
        Bi->d[1] = SUN_RCONST(3.0)/SUN_RCONST(5.0);
        Bi->d[2] = SUN_RCONST(2.0)/SUN_RCONST(5.0);
        Bi->q = 2;
        Bi->p = 1;
      }
    }
    else if (uopts.table_id == 2) // Giraldo ARK2
    {
      if (fe_RHS != nullptr)
      {
        Be          = ARKodeButcherTable_Alloc(3, SUNTRUE);
        Be->c[1]    = SUN_RCONST(2.0) - SUNRsqrt(SUN_RCONST(2.0));
        Be->c[2]    = SUN_RCONST(1.0);
        Be->A[1][0] = SUN_RCONST(2.0) - SUNRsqrt(SUN_RCONST(2.0));
        Be->A[2][0] = (SUN_RCONST(3.0) - SUNRsqrt(SUN_RCONST(8.0))) /
                      SUN_RCONST(6.0);
        Be->A[2][1] = (SUN_RCONST(3.0) + SUNRsqrt(SUN_RCONST(8.0))) /
                      SUN_RCONST(6.0);
        Be->b[0] = SUN_RCONST(1.0) / SUNRsqrt(SUN_RCONST(8.0));
        Be->b[1] = SUN_RCONST(1.0) / SUNRsqrt(SUN_RCONST(8.0));
        Be->b[2] = SUN_RCONST(1.0) - SUN_RCONST(1.0) / SUNRsqrt(SUN_RCONST(2.0));
        Be->d[0] = (SUN_RCONST(4.0) - SUNRsqrt(SUN_RCONST(2.0))) / SUN_RCONST(8.0);
        Be->d[1] = (SUN_RCONST(4.0) - SUNRsqrt(SUN_RCONST(2.0))) / SUN_RCONST(8.0);
        Be->d[2] = SUN_RCONST(1.0) / SUNRsqrt(SUN_RCONST(8.0));
        Be->q    = 2;
        Be->p    = 1;
      }
      if (fi_RHS != nullptr)
      {
        Bi       = ARKodeButcherTable_Alloc(3, SUNTRUE);
        Bi->c[1] = SUN_RCONST(2.0) - SUNRsqrt(SUN_RCONST(2.0));
        Bi->c[2] = SUN_RCONST(1.0);
        Bi->A[1][0] = SUN_RCONST(1.0) - SUN_RCONST(1.0) / SUNRsqrt(SUN_RCONST(2.0));
        Bi->A[1][1] = SUN_RCONST(1.0) - SUN_RCONST(1.0) / SUNRsqrt(SUN_RCONST(2.0));
        Bi->A[2][0] = SUN_RCONST(1.0) / SUNRsqrt(SUN_RCONST(8.0));
        Bi->A[2][1] = SUN_RCONST(1.0) / SUNRsqrt(SUN_RCONST(8.0));
        Bi->A[2][2] = SUN_RCONST(1.0) - SUN_RCONST(1.0) / SUNRsqrt(SUN_RCONST(2.0));
        Bi->b[0] = SUN_RCONST(1.0) / SUNRsqrt(SUN_RCONST(8.0));
        Bi->b[1] = SUN_RCONST(1.0) / SUNRsqrt(SUN_RCONST(8.0));
        Bi->b[2] = SUN_RCONST(1.0) - SUN_RCONST(1.0) / SUNRsqrt(SUN_RCONST(2.0));
        Bi->d[0] = (SUN_RCONST(4.0) - SUNRsqrt(SUN_RCONST(2.0))) / SUN_RCONST(8.0);
        Bi->d[1] = (SUN_RCONST(4.0) - SUNRsqrt(SUN_RCONST(2.0))) / SUN_RCONST(8.0);
        Bi->d[2] = SUN_RCONST(1.0) / SUNRsqrt(SUN_RCONST(8.0));
        Bi->q    = 2;
        Bi->p    = 1;
      }
    }
    else if (uopts.table_id == 3) // Ralston
    {
      Be          = ARKodeButcherTable_Alloc(3, SUNTRUE);
      Be->c[1]    = SUN_RCONST(2.0) / SUN_RCONST(3.0);
      Be->c[2]    = SUN_RCONST(1.0);
      Be->A[1][0] = SUN_RCONST(2.0) / SUN_RCONST(3.0);
      Be->A[2][0] = SUN_RCONST(1.0) / SUN_RCONST(4.0);
      Be->A[2][1] = SUN_RCONST(3.0) / SUN_RCONST(4.0);
      Be->b[0]    = SUN_RCONST(1.0) / SUN_RCONST(4.0);
      Be->b[1]    = SUN_RCONST(3.0) / SUN_RCONST(4.0);
      Be->d[0]    = SUN_RCONST(5.0) / SUN_RCONST(37.0);
      Be->d[1]    = SUN_RCONST(2.0) / SUN_RCONST(3.0);
      Be->d[2]    = SUN_RCONST(22.0) / SUN_RCONST(111.0);
      Be->q       = 2;
      Be->p       = 1;
    }
    else if (uopts.table_id == 4) // Heun-Euler
    {
      Be          = ARKodeButcherTable_Alloc(3, SUNTRUE);
      Be->c[1]    = SUN_RCONST(1.0);
      Be->c[2]    = SUN_RCONST(1.0);
      Be->A[1][0] = SUN_RCONST(1.0);
      Be->A[2][0] = SUN_RCONST(0.5);
      Be->A[2][1] = SUN_RCONST(0.5);
      Be->b[0]    = SUN_RCONST(0.5);
      Be->b[1]    = SUN_RCONST(0.5);
      Be->d[0]    = SUN_RCONST(1.0);
      Be->q       = 2;
      Be->p       = 1;
    }
    else if (uopts.table_id == 5) // SSP SDIRK2
    {
      Bi                     = ARKodeButcherTable_Alloc(2, SUNTRUE);
      const sunrealtype beta = SUN_RCONST(1.0) -
                               SUN_RCONST(1.0) / SUNRsqrt(SUN_RCONST(2.0));
      Bi->c[0]    = beta;
      Bi->c[1]    = SUN_RCONST(1.0) - beta;
      Bi->A[0][0] = beta;
      Bi->A[1][0] = SUN_RCONST(1.0) - SUN_RCONST(2.0) * beta;
      Bi->A[1][1] = beta;
      Bi->b[0]    = SUN_RCONST(0.5);
      Bi->b[1]    = SUN_RCONST(0.5);
      Bi->d[0]    = SUN_RCONST(5.0) / SUN_RCONST(12.0);
      Bi->d[1]    = SUN_RCONST(7.0) / SUN_RCONST(12.0);
      Bi->q       = 2;
      Bi->p       = 1;
    }
    flag = ARKStepSetTables(*arkode_mem, 2, 1, Bi, Be);
    if (check_flag(flag, "ARKStepSetTables")) { return 1; }
    if (Be) { ARKodeButcherTable_Free(Be); }
    if (Bi) { ARKodeButcherTable_Free(Bi); }
  }
  else
  {
    // Select default method of a given order
    flag = ARKodeSetOrder(*arkode_mem, uopts.order);
    if (check_flag(flag, "ARKodeSetOrder")) { return 1; }
  }

  // Set fixed step size, or error bias if using adaptive time stepping
  if (uopts.fixed_h > ZERO)
  {
    flag = ARKodeSetFixedStep(*arkode_mem, uopts.fixed_h);
    if (check_flag(flag, "ARKodeSetFixedStep")) { return 1; }
  }
  else
  {
    flag = ARKodeSetErrorBias(*arkode_mem, uopts.error_bias);
    if (check_flag(flag, "ARKodeSetErrorBias")) { return 1; }
  }

  // Set max steps between outputs
  flag = ARKodeSetMaxNumSteps(*arkode_mem, uopts.maxsteps);
  if (check_flag(flag, "ARKodeSetMaxNumSteps")) { return 1; }

  // Set stopping time
  flag = ARKodeSetStopTime(*arkode_mem, udata.tf);
  if (check_flag(flag, "ARKodeSetStopTime")) { return 1; }

  return 0;
}

int SetupReference(SUNContext ctx, UserData& udata, UserOptions& uopts,
                   N_Vector y, SUNLinearSolver* LS, void** arkode_mem)
{
  // Problem configuration
  ARKRhsFn fe_RHS;   // explicit RHS function
  ARKRhsFn fi_RHS;   // implicit RHS function

  // advection + diffusion + implicit reaction
  if (udata.impl_reaction && udata.advection)
  {
    fe_RHS = f_advection;
    fi_RHS = f_diff_react;
  }
  // advection + diffusion + explicit reaction
  else if (!udata.impl_reaction && udata.advection)
  {
    fe_RHS = f_adv_react;
    fi_RHS = f_diffusion;
  }
  // diffusion + implicit reaction
  else if (udata.impl_reaction && !udata.advection)
  {
    fe_RHS = nullptr;
    fi_RHS = f_diff_react;
  }
  // diffusion + explicit reaction
  else if (!udata.impl_reaction && !udata.advection)
  {
    fe_RHS = f_reaction;
    fi_RHS = f_diffusion;
  }
  else
  {
    cerr << "ERROR: Invalid problem configuration" << endl;
    return -1;
  }

  // Create ARKStep memory
  *arkode_mem = ARKStepCreate(fe_RHS, fi_RHS, ZERO, y, ctx);
  if (check_ptr(arkode_mem, "ARKStepCreate")) { return 1; }

  // Specify tolerances
  int flag = ARKodeSStolerances(*arkode_mem, uopts.rtol, uopts.atol);
  if (check_flag(flag, "ARKodeSStolerances")) { return 1; }

  // Attach user data
  flag = ARKodeSetUserData(*arkode_mem, &udata);
  if (check_flag(flag, "ARKodeSetUserData")) { return 1; }

  // Create linear solver
  *LS = SUNLinSol_SPGMR(y, SUN_PREC_NONE, uopts.maxl, ctx);
  if (check_ptr(*LS, "SUNLinSol_SPGMR")) { return 1; }

  // Attach linear solver
  flag = ARKodeSetLinearSolver(*arkode_mem, *LS, nullptr);
  if (check_flag(flag, "ARKodeSetLinearSolver")) { return 1; }

  // Tighten implicit solver tolerances and allow more Newton iterations
  flag = ARKodeSetMaxNonlinIters(*arkode_mem, uopts.maxnewt);
  if (check_flag(flag, "ARKodeSetMaxNonlinIters")) { return 1; }
  flag = ARKodeSetNonlinConvCoef(*arkode_mem, uopts.nlscoef);
  if (check_flag(flag, "ARKodeSetNonlinConvCoef")) { return 1; }
  flag = ARKodeSetEpsLin(*arkode_mem, uopts.epslin);
  if (check_flag(flag, "ARKodeSetEpsLin")) { return 1; }

  // Use "deduce implicit RHS" option
  flag = ARKodeSetDeduceImplicitRhs(*arkode_mem, SUNTRUE);
  if (check_flag(flag, "ARKodeSetDeduceImplicitRhs")) { return 1; }

  // Set the predictor method
  flag = ARKodeSetPredictorMethod(*arkode_mem, uopts.predictor);
  if (check_flag(flag, "ARKodeSetPredictorMethod")) { return 1; }

  // Set linear solver setup frequency
  flag = ARKodeSetLSetupFrequency(*arkode_mem, uopts.ls_setup_freq);
  if (check_flag(flag, "ARKodeSetLSetupFrequency")) { return 1; }

  // Select default method of a given order
  flag = ARKodeSetOrder(*arkode_mem, uopts.order);
  if (check_flag(flag, "ARKodeSetOrder")) { return 1; }

  // Set fixed step size
  if (uopts.fixed_h > ZERO)
  {
    flag = ARKodeSetFixedStep(*arkode_mem, uopts.fixed_h);
    if (check_flag(flag, "ARKodeSetFixedStep")) { return 1; }
  }

  // Set max steps between outputs
  flag = ARKodeSetMaxNumSteps(*arkode_mem, uopts.maxsteps);
  if (check_flag(flag, "ARKodeSetMaxNumSteps")) { return 1; }

  // Set stopping time
  flag = ARKodeSetStopTime(*arkode_mem, udata.tf);
  if (check_flag(flag, "ARKodeSetStopTime")) { return 1; }

  return 0;
}

int SetupExtSTS(SUNContext ctx, UserData& udata, UserOptions& uopts, N_Vector y,
                SUNMatrix* A, SUNLinearSolver* LS, MRIStepInnerStepper* sts_mem,
                void** arkode_mem)
{
  // Problem configuration
  ARKRhsFn fe_RHS;   // explicit RHS function
  ARKRhsFn fi_RHS;   // implicit RHS function
  ARKLsJacFn Ji_RHS; // implicit RHS Jacobian function

  fi_RHS = (udata.impl_reaction) ? f_reaction : nullptr;
  Ji_RHS = (udata.impl_reaction) ? J_reaction : nullptr;
  if (udata.advection)
  {
    fe_RHS = (udata.impl_reaction) ? f_advection : f_adv_react;
  }
  else
  {
    fe_RHS = (udata.impl_reaction) ? nullptr : f_reaction;
  }

  // -------------------------------
  // Setup the custom STS integrator
  // -------------------------------

  // Create LSRKStep memory
  void* sts_arkode_mem = LSRKStepCreateSTS(f_diffusion_forcing, ZERO, y, ctx);
  if (check_ptr(arkode_mem, "LSRKStepCreateSTS")) { return 1; }

  // Attach user data
  int flag = ARKodeSetUserData(sts_arkode_mem, &udata);
  if (check_flag(flag, "ARKodeSetUserData")) { return 1; }

  // Select STS method
  ARKODE_LSRKMethodType ststype = (uopts.sts_method == 0) ? ARKODE_LSRK_RKC_2
                                                          : ARKODE_LSRK_RKL_2;
  flag = LSRKStepSetSTSMethod(sts_arkode_mem, ststype);
  if (check_flag(flag, "LSRKStepSetSTSMethod")) { return 1; }

  // Set dominant eigenvalue function and frequency
  flag = LSRKStepSetDomEigFn(sts_arkode_mem, diffusion_domeig);
  if (check_flag(flag, "LSRKStepSetDomEigFn")) { return 1; }
  flag = LSRKStepSetDomEigFrequency(sts_arkode_mem, uopts.ls_setup_freq);
  if (check_flag(flag, "LSRKStepSetDomEigFrequency")) { return 1; }

  // Increase the maximum number of internal STS stages allowed
  flag = LSRKStepSetMaxNumStages(sts_arkode_mem, 10000);
  if (check_flag(flag, "LSRKStepSetMaxNumStages")) { return 1; }

  // Disable temporal interpolation for inner STS method
  flag = ARKodeSetInterpolantType(sts_arkode_mem, ARK_INTERP_NONE);
  if (check_flag(flag, "ARKodeSetInterpolantType")) { return 1; }

  // Create the inner stepper wrapper
  flag = MRIStepInnerStepper_Create(ctx, sts_mem);
  if (check_flag(flag, "MRIStepInnerStepper_Create")) { return 1; }

  // Attach memory and operations
  STSInnerStepperContent* inner_content = new STSInnerStepperContent;
  inner_content->sts_arkode_mem         = sts_arkode_mem;
  inner_content->user_data              = &udata;

  flag = MRIStepInnerStepper_SetContent(*sts_mem, inner_content);
  if (check_flag(flag, "MRIStepInnerStepper_SetContent")) { return 1; }

  flag = MRIStepInnerStepper_SetEvolveFn(*sts_mem, STSInnerStepper_Evolve);
  if (check_flag(flag, "MRIStepInnerStepper_SetEvolveFn")) { return 1; }

  flag = MRIStepInnerStepper_SetFullRhsFn(*sts_mem, STSInnerStepper_FullRhs);
  if (check_flag(flag, "MRIStepInnerStepper_SetFullRhsFn")) { return 1; }

  flag = MRIStepInnerStepper_SetResetFn(*sts_mem, STSInnerStepper_Reset);
  if (check_flag(flag, "MRIStepInnerStepper_SetResetFn")) { return 1; }

  // Attach inner stepper memory to user data
  udata.sts_mem = *sts_mem;

  // -------------------------
  // Setup the MRI integrator
  // -------------------------

  // Create "slow" integrator for advection+reaction and attach "fast" integrator for diffusion
  *arkode_mem = MRIStepCreate(fe_RHS, fi_RHS, ZERO, y, *sts_mem, ctx);
  if (check_ptr(*arkode_mem, "MRIStepCreate")) { return 1; }

  // Set fixed step size, or temporal adaptivity error bias
  if (uopts.fixed_h > ZERO)
  {
    flag = ARKodeSetFixedStep(*arkode_mem, uopts.fixed_h);
    if (check_flag(flag, "ARKodeSetFixedStep")) { return 1; }
  }
  else
  {
    flag = ARKodeSetErrorBias(*arkode_mem, uopts.error_bias);
    if (check_flag(flag, "ARKodeSetErrorBias")) { return 1; }
  }

  // Specify tolerances
  flag = ARKodeSStolerances(*arkode_mem, uopts.rtol, uopts.atol);
  if (check_flag(flag, "ARKodeSStolerances")) { return 1; }

  // Attach user data
  flag = ARKodeSetUserData(*arkode_mem, &udata);
  if (check_flag(flag, "ARKodeSetUserData")) { return 1; }

  // If implicit, setup solvers
  if (udata.impl_reaction)
  {
    // Create banded matrix
    *A = SUNBandMatrix(udata.neq, 2, 2, ctx);
    if (check_ptr(*A, "SUNBandMatrix")) { return 1; }

    // Create linear solver
    *LS = SUNLinSol_Band(y, *A, ctx);
    if (check_ptr(*LS, "SUNLinSol_Band")) { return 1; }

    // Attach linear solver
    flag = ARKodeSetLinearSolver(*arkode_mem, *LS, *A);
    if (check_flag(flag, "ARKodeSetLinearSolver")) { return 1; }

    // Attach Jacobian function
    flag = ARKodeSetJacFn(*arkode_mem, Ji_RHS);
    if (check_flag(flag, "ARKodeSetJacFn")) { return 1; }

    // Set linear solver setup frequency
    flag = ARKodeSetLSetupFrequency(*arkode_mem, uopts.ls_setup_freq);
    if (check_flag(flag, "ARKodeSetLSetupFrequency")) { return 1; }

    // Tighten implicit solver tolerances and allow more Newton iterations
    flag = ARKodeSetMaxNonlinIters(*arkode_mem, uopts.maxnewt);
    if (check_flag(flag, "ARKodeSetMaxNonlinIters")) { return 1; }
    flag = ARKodeSetNonlinConvCoef(*arkode_mem, uopts.nlscoef);
    if (check_flag(flag, "ARKodeSetNonlinConvCoef")) { return 1; }

    // // Use "deduce implicit RHS" option
    // flag = ARKodeSetDeduceImplicitRhs(*arkode_mem, SUNTRUE);
    // if (check_flag(flag, "ARKodeSetDeduceImplicitRhs")) { return 1; }

    // Set the predictor method
    flag = ARKodeSetPredictorMethod(*arkode_mem, uopts.predictor);
    if (check_flag(flag, "ARKodeSetPredictorMethod")) { return 1; }
  }

  // Select ExtSTS method via MRIStepCoupling structure
  MRIStepCoupling C;
  if (uopts.extsts_method == 0) // ARS(2,2,2)
  {
    if (udata.impl_reaction && udata.advection)
    {
      C = MRIStepCoupling_Alloc(1, 5, MRISTEP_IMEX);
      const sunrealtype one   = SUN_RCONST(1.0);
      const sunrealtype gamma = one - one / SUNRsqrt(SUN_RCONST(2.0));
      const sunrealtype delta = one - one / (SUN_RCONST(2.0) * gamma);
      const sunrealtype three = SUN_RCONST(3.0);
      C->q = 2;
      C->p = 1;
      C->c[1] = gamma;
      C->c[2] = gamma;
      C->c[3] = one;
      C->c[4] = one;
      C->W[0][1][0] = gamma;
      C->W[0][3][0] = delta - gamma;
      C->W[0][3][2] = one - delta;
      C->W[0][5][0] = -delta;
      C->W[0][5][2] = delta - SUN_RCONST(0.4);
      C->W[0][5][4] = SUN_RCONST(0.4);
      C->G[0][1][0] =  gamma;
      C->G[0][2][0] = -gamma;
      C->G[0][2][2] =  gamma;
      C->G[0][3][2] =  one - gamma;
      C->G[0][4][2] = -gamma;
      C->G[0][4][4] =  gamma;
      C->G[0][5][2] = -SUN_RCONST(0.4);
      C->G[0][5][4] =  SUN_RCONST(0.4);
    }
    else if (udata.impl_reaction)
    {
      C = MRIStepCoupling_Alloc(1, 5, MRISTEP_IMPLICIT);
      const sunrealtype one = SUN_RCONST(1.0);
      const sunrealtype gamma = one - one / SUNRsqrt(SUN_RCONST(2.0));
      const sunrealtype delta = one - one / (SUN_RCONST(2.0)*gamma);
      const sunrealtype three = SUN_RCONST(3.0);
      C->q = 2;
      C->p = 1;
      C->c[1] = gamma;
      C->c[2] = gamma;
      C->c[3] = one;
      C->c[4] = one;
      C->G[0][1][0] =  gamma;
      C->G[0][2][0] = -gamma;
      C->G[0][2][2] =  gamma;
      C->G[0][3][2] =  one - gamma;
      C->G[0][4][2] = -gamma;
      C->G[0][4][4] =  gamma;
      C->G[0][5][2] = -SUN_RCONST(0.4);
      C->G[0][5][4] =  SUN_RCONST(0.4);
    }
    else
    {
      C = MRIStepCoupling_Alloc(1, 5, MRISTEP_EXPLICIT);
      const sunrealtype one = SUN_RCONST(1.0);
      const sunrealtype gamma = one - one / SUNRsqrt(SUN_RCONST(2.0));
      const sunrealtype delta = one - one / (SUN_RCONST(2.0)*gamma);
      const sunrealtype three = SUN_RCONST(3.0);
      C->q = 2;
      C->p = 1;
      C->c[1] = gamma;
      C->c[2] = gamma;
      C->c[3] = one;
      C->c[4] = one;
      C->W[0][1][0] = gamma;
      C->W[0][3][0] = delta - gamma;
      C->W[0][3][2] = one - delta;
      C->W[0][5][0] = -delta;
      C->W[0][5][2] = delta - SUN_RCONST(0.4);
      C->W[0][5][4] = SUN_RCONST(0.4);
    }
  }
  else if (uopts.extsts_method == 1) // Giraldo
  {
    if (udata.impl_reaction && udata.advection)
    {
      C = MRIStepCoupling_Alloc(1, 6, MRISTEP_IMEX);
      const sunrealtype one   = SUN_RCONST(1.0);
      const sunrealtype two   = SUN_RCONST(2.0);
      const sunrealtype three = SUN_RCONST(3.0);
      const sunrealtype four  = SUN_RCONST(4.0);
      const sunrealtype six   = SUN_RCONST(6.0);
      const sunrealtype eight = SUN_RCONST(8.0);
      const sunrealtype sqrt2 = SUNRsqrt(two);
      C->q                    = 2;
      C->p                    = 1;
      C->c[1]                 = two - sqrt2;
      C->c[2]                 = two - sqrt2;
      C->c[3]                 = one;
      C->c[4]                 = one;
      C->c[5]                 = one;
      C->W[0][1][0]           = two - sqrt2;
      C->W[0][3][0]           = (three - two * sqrt2) / six - (two - sqrt2);
      C->W[0][3][2]           = (three + two * sqrt2) / six;
      C->W[0][5][0] = one / (two * sqrt2) - (three - two * sqrt2) / six;
      C->W[0][5][2] = one / (two * sqrt2) - (three + two * sqrt2) / six;
      C->W[0][5][4] = one - one / SUNRsqrt(SUN_RCONST(2.0));
      C->W[0][6][0] = (four - sqrt2) / eight - (three - two * sqrt2) / six;
      C->W[0][6][2] = (four - sqrt2) / eight - (three + two * sqrt2) / six;
      C->W[0][6][4] = one / (two * sqrt2);
      C->G[0][1][0] = two - sqrt2;
      C->G[0][2][0] = one - one / sqrt2 - (two - sqrt2);
      C->G[0][2][2] = one - one / sqrt2;
      C->G[0][3][0] = one / sqrt2 - one;
      C->G[0][3][2] = one / sqrt2;
      C->G[0][4][0] = one / (two * sqrt2);
      C->G[0][4][2] = one / (two * sqrt2) - one;
      C->G[0][4][4] = one - one / sqrt2;
      C->G[0][6][0] = (four - sqrt2) / eight - one / (two * sqrt2);
      C->G[0][6][2] = (four - sqrt2) / eight - one / (two * sqrt2);
      C->G[0][6][4] = one / (two * sqrt2) - (one - one / sqrt2);
    }
    else if (udata.impl_reaction)
    {
      C = MRIStepCoupling_Alloc(1, 6, MRISTEP_IMPLICIT);
      const sunrealtype one   = SUN_RCONST(1.0);
      const sunrealtype two   = SUN_RCONST(2.0);
      const sunrealtype three = SUN_RCONST(3.0);
      const sunrealtype four  = SUN_RCONST(4.0);
      const sunrealtype six   = SUN_RCONST(6.0);
      const sunrealtype eight = SUN_RCONST(8.0);
      const sunrealtype sqrt2 = SUNRsqrt(two);
      C->q                    = 2;
      C->p                    = 1;
      C->c[1]                 = two - sqrt2;
      C->c[2]                 = two - sqrt2;
      C->c[3]                 = one;
      C->c[4]                 = one;
      C->c[5]                 = one;
      C->G[0][1][0]           = two - sqrt2;
      C->G[0][2][0]           = one - one / sqrt2 - (two - sqrt2);
      C->G[0][2][2]           = one - one / sqrt2;
      C->G[0][3][0]           = one / sqrt2 - one;
      C->G[0][3][2]           = one / sqrt2;
      C->G[0][4][0]           = one / (two * sqrt2);
      C->G[0][4][2]           = one / (two * sqrt2) - one;
      C->G[0][4][4]           = one - one / sqrt2;
      C->G[0][6][0]           = (four - sqrt2) / eight - one / (two * sqrt2);
      C->G[0][6][2]           = (four - sqrt2) / eight - one / (two * sqrt2);
      C->G[0][6][4]           = one / (two * sqrt2) - (one - one / sqrt2);
    }
    else
    {
      C = MRIStepCoupling_Alloc(1, 6, MRISTEP_EXPLICIT);
      const sunrealtype one = SUN_RCONST(1.0);
      const sunrealtype two = SUN_RCONST(2.0);
      const sunrealtype three = SUN_RCONST(3.0);
      const sunrealtype four = SUN_RCONST(4.0);
      const sunrealtype six = SUN_RCONST(6.0);
      const sunrealtype eight = SUN_RCONST(8.0);
      const sunrealtype sqrt2 = SUNRsqrt(two);
      C->q = 2;
      C->p = 1;
      C->c[1] = two - sqrt2;
      C->c[2] = two - sqrt2;
      C->c[3] = one;
      C->c[4] = one;
      C->c[5] = one;
      C->W[0][1][0] = two - sqrt2;
      C->W[0][3][0] = (three - two * sqrt2)/six - (two - sqrt2);
      C->W[0][3][2] = (three + two * sqrt2)/six;
      C->W[0][5][0] = one/(two * sqrt2) - (three - two * sqrt2)/six;
      C->W[0][5][2] = one/(two * sqrt2) - (three + two * sqrt2)/six;
      C->W[0][5][4] = one - one / SUNRsqrt(SUN_RCONST(2.0));
      C->W[0][6][0] = (four - sqrt2) / eight - (three - two * sqrt2)/six;
      C->W[0][6][2] = (four - sqrt2) / eight - (three + two * sqrt2)/six;
      C->W[0][6][4] = one / (two * sqrt2);
    }
  }
  else if (uopts.extsts_method == 2)  // Ralston (explicit only)
  {
    C                       = MRIStepCoupling_Alloc(1, 3, MRISTEP_EXPLICIT);
    const sunrealtype one   = SUN_RCONST(1.0);
    const sunrealtype two   = SUN_RCONST(2.0);
    const sunrealtype three = SUN_RCONST(3.0);
    const sunrealtype four  = SUN_RCONST(4.0);
    C->q                    = 2;
    C->p                    = 1;
    C->c[1]                 = two / three;
    C->c[2]                 = one;
    C->W[0][1][0]           = two / three;
    C->W[0][2][0]           = one / four - two / three;
    C->W[0][2][1]           = three / four;
    C->W[0][3][0] = SUN_RCONST(5.0) / SUN_RCONST(37.0) - two / three;
    C->W[0][3][1] = two / three - three / four;
    C->W[0][3][2] = SUN_RCONST(22.0) / SUN_RCONST(111.0);
  }
  else if (uopts.extsts_method == 3) // Heun-Euler (explicit only)
  {
    C                       = MRIStepCoupling_Alloc(1, 3, MRISTEP_EXPLICIT);
    const sunrealtype one   = SUN_RCONST(1.0);
    const sunrealtype two   = SUN_RCONST(2.0);
    const sunrealtype three = SUN_RCONST(3.0);
    C->q                    = 2;
    C->p                    = 1;
    C->c[1]                 = one;
    C->c[2]                 = one;
    C->W[0][1][0]           = one;
    C->W[0][2][0]           = -one / two;
    C->W[0][2][1]           = one / two;
  }
  else if (uopts.extsts_method == 4)  // SSP SDIRK 2
  {
    C = MRIStepCoupling_Alloc(1, 6, MRISTEP_IMPLICIT);
    const sunrealtype one = SUN_RCONST(1.0);
    const sunrealtype two = SUN_RCONST(2.0);
    const sunrealtype five = SUN_RCONST(5.0);
    const sunrealtype seven = SUN_RCONST(7.0);
    const sunrealtype twelve = SUN_RCONST(12.0);
    const sunrealtype gamma = one - one / SUNRsqrt(two);
    C->q = 2;
    C->p = 1;
    C->c[1] = gamma;
    C->c[2] = gamma;
    C->c[3] = one - gamma;
    C->c[4] = one - gamma;
    C->c[5] = one;
    C->G[0][1][0] = gamma;
    C->G[0][2][0] = -gamma;
    C->G[0][2][2] = gamma;
    C->G[0][3][2] = one - two * gamma;
    C->G[0][4][2] = -gamma;
    C->G[0][4][4] = gamma;
    C->G[0][5][2] = two * gamma - one / two;
    C->G[0][5][4] = one / two - gamma;
    C->G[0][6][2] = two*gamma - seven / twelve;
    C->G[0][6][4] = seven / twelve - gamma;
  }
  else if (uopts.extsts_method < 0)  // use abs(method) to get MRI table
  {
    ARKODE_MRITableID mri_table = static_cast<ARKODE_MRITableID>(-uopts.extsts_method);
    C = MRIStepCoupling_LoadTable(mri_table);
    if (C == nullptr)
    {
      cerr << "ERROR: Unable to load MRI table " << mri_table << endl;
      return -1;
    }
  }
  else // illegal configuration
  {
    cerr << "ERROR: Invalid extsts method " << uopts.extsts_method << endl;
    return -1;
  }
  flag = MRIStepSetCoupling(*arkode_mem, C);
  if (check_flag(flag, "MRIStepSetCoupling")) { return 1; }
  MRIStepCoupling_Free(C);

  // Set max steps between outputs
  flag = ARKodeSetMaxNumSteps(*arkode_mem, uopts.maxsteps);
  if (check_flag(flag, "ARKodeSetMaxNumSteps")) { return 1; }

  // Tighten safety factor for time step selection
  flag = ARKodeSetSafetyFactor(*arkode_mem, 0.8);
  if (check_flag(flag, "ARKodeSetSafetyFactor")) { return 1; }

  // Set stopping time
  flag = ARKodeSetStopTime(*arkode_mem, udata.tf);
  if (check_flag(flag, "ARKodeSetStopTime")) { return 1; }

  return 0;
}

int SetupStrang(SUNContext ctx, UserData& udata, UserOptions& uopts, N_Vector y,
                SUNMatrix* A, SUNLinearSolver* LS,  SUNStepper steppers[2],
                void** lsrkstep_mem, void** arkstep_mem, void** arkode_mem)
{
  // Problem configuration
  ARKRhsFn fe_RHS;     // explicit RHS function
  ARKRhsFn fi_RHS;     // implicit RHS function
  ARKLsJacFn Ji_RHS;   // implicit RHS Jacobian function

  fi_RHS = (udata.impl_reaction) ? f_reaction : nullptr;
  Ji_RHS = (udata.impl_reaction) ? J_reaction : nullptr;
  if (udata.advection)
  {
    fe_RHS = (udata.impl_reaction) ? f_advection : f_adv_react;
  }
  else
  {
    fe_RHS = (udata.impl_reaction) ? nullptr : f_reaction;
  }

  // -----------------------------
  // Setup the LSRKStep integrator
  // -----------------------------

  // Create LSRKStep memory, and attach to steppers[0]
  *lsrkstep_mem = LSRKStepCreateSTS(f_diffusion, ZERO, y, ctx);
  if (check_ptr(*lsrkstep_mem, "LSRKStepCreateSTS")) { return 1; }

  // Attach user data
  int flag = ARKodeSetUserData(*lsrkstep_mem, &udata);
  if (check_flag(flag, "ARKodeSetUserData")) { return 1; }

  // Select STS method
  ARKODE_LSRKMethodType ststype = (uopts.sts_method == 0) ? ARKODE_LSRK_RKC_2 : ARKODE_LSRK_RKL_2;
  flag = LSRKStepSetSTSMethod(*lsrkstep_mem, ststype);
  if (check_flag(flag, "LSRKStepSetSTSMethod")) { return 1; }

  // Set dominant eigenvalue function and frequency
  flag = LSRKStepSetDomEigFn(*lsrkstep_mem, diffusion_domeig);
  if (check_flag(flag, "LSRKStepSetDomEigFn")) { return 1; }
  flag = LSRKStepSetDomEigFrequency(*lsrkstep_mem, uopts.ls_setup_freq);
  if (check_flag(flag, "LSRKStepSetDomEigFrequency")) { return 1; }

  // Increase the maximum number of internal STS stages allowed
  flag = LSRKStepSetMaxNumStages(*lsrkstep_mem, 10000);
  if (check_flag(flag, "LSRKStepSetMaxNumStages")) { return 1; }

  // Set fixed step size
  flag = ARKodeSetFixedStep(*lsrkstep_mem, uopts.fixed_h);
  if (check_flag(flag, "ARKodeSetFixedStep")) { return 1; }

  // Set max steps between outputs
  flag = ARKodeSetMaxNumSteps(*lsrkstep_mem, uopts.maxsteps);
  if (check_flag(flag, "ARKodeSetMaxNumSteps")) { return 1; }

  // Disable temporal interpolation for STS method
  flag = ARKodeSetInterpolantType(*lsrkstep_mem, ARK_INTERP_NONE);
  if (check_flag(flag, "ARKodeSetInterpolantType")) { return 1; }

  // Wrap as a SUNStepper
  flag = ARKodeCreateSUNStepper(*lsrkstep_mem, &steppers[0]);
  if (check_flag(flag, "ARKodeCreateSUNStepper")) { return 1; }


  // ----------------------------
  // Setup the ARKStep integrator
  // ----------------------------

  // Create ARKStep memory, and attach to steppers[1]
  *arkstep_mem = ARKStepCreate(fe_RHS, fi_RHS, ZERO, y, ctx);
  if (check_ptr(*arkstep_mem, "ARKStepCreate")) { return 1; }

  // Attach user data
  flag = ARKodeSetUserData(*arkstep_mem, &udata);
  if (check_flag(flag, "ARKodeSetUserData")) { return 1; }

  // Set fixed step size
  flag = ARKodeSetFixedStep(*arkstep_mem, uopts.fixed_h);
  if (check_flag(flag, "ARKodeSetFixedStep")) { return 1; }

  // Set max steps between outputs
  flag = ARKodeSetMaxNumSteps(*arkstep_mem, uopts.maxsteps);
  if (check_flag(flag, "ARKodeSetMaxNumSteps")) { return 1; }

  // If implicit or ImEx, setup solvers
  if (udata.impl_reaction)
  {
    // Specify tolerances (relevant for nonlinear implicit solver)
    flag = ARKodeSStolerances(*arkstep_mem, uopts.rtol, uopts.atol);
    if (check_flag(flag, "ARKodeSStolerances")) { return 1; }

    // Create banded matrix
    *A = SUNBandMatrix(udata.neq, 2, 2, ctx);
    if (check_ptr(*A, "SUNBandMatrix")) { return 1; }

    // Create linear solver
    *LS = SUNLinSol_Band(y, *A, ctx);
    if (check_ptr(*LS, "SUNLinSol_Band")) { return 1; }

    // Attach linear solver
    flag = ARKodeSetLinearSolver(*arkstep_mem, *LS, *A);
    if (check_flag(flag, "ARKodeSetLinearSolver")) { return 1; }

    // Attach Jacobian function
    flag = ARKodeSetJacFn(*arkstep_mem, Ji_RHS);
    if (check_flag(flag, "ARKodeSetJacFn")) { return 1; }

    // Set linear solver setup frequency
    flag = ARKodeSetLSetupFrequency(*arkstep_mem, uopts.ls_setup_freq);
    if (check_flag(flag, "ARKodeSetLSetupFrequency")) { return 1; }

    // Tighten implicit solver tolerances and allow more Newton iterations
    flag = ARKodeSetMaxNonlinIters(*arkstep_mem, uopts.maxnewt);
    if (check_flag(flag, "ARKodeSetMaxNonlinIters")) { return 1; }
    flag = ARKodeSetNonlinConvCoef(*arkstep_mem, uopts.nlscoef);
    if (check_flag(flag, "ARKodeSetNonlinConvCoef")) { return 1; }

    // Use "deduce implicit RHS" option
    flag = ARKodeSetDeduceImplicitRhs(*arkstep_mem, SUNTRUE);
    if (check_flag(flag, "ARKodeSetDeduceImplicitRhs")) { return 1; }

    // Set the predictor method
    flag = ARKodeSetPredictorMethod(*arkstep_mem, uopts.predictor);
    if (check_flag(flag, "ARKodeSetPredictorMethod")) { return 1; }

  }

  // Set the RK tables (no embeddings needed)
  ARKodeButcherTable Be = nullptr;
  ARKodeButcherTable Bi = nullptr;
  if (fi_RHS != nullptr)
  {
    Bi = ARKodeButcherTable_Alloc(3, SUNFALSE);
    const sunrealtype gamma = (SUN_RCONST(2.0)-SUNRsqrt(SUN_RCONST(2.0)))/SUN_RCONST(2.0);
    const sunrealtype delta = SUN_RCONST(1.0)-SUN_RCONST(1.0)/(SUN_RCONST(2.0)*gamma);
    Bi->c[1] = gamma;
    Bi->c[2] = SUN_RCONST(1.0);
    Bi->A[1][1] = gamma;
    Bi->A[2][1] = SUN_RCONST(1.0)-gamma;
    Bi->A[2][2] = gamma;
    Bi->b[1] = SUN_RCONST(1.0)-gamma;
    Bi->b[2] = gamma;
    Bi->q = 2;
  }
  if (fe_RHS != nullptr)
  {
    Be = ARKodeButcherTable_Alloc(3, SUNFALSE);
    const sunrealtype gamma = (SUN_RCONST(2.0)-SUNRsqrt(SUN_RCONST(2.0)))/SUN_RCONST(2.0);
    const sunrealtype delta = SUN_RCONST(1.0)-SUN_RCONST(1.0)/(SUN_RCONST(2.0)*gamma);
    Be->c[1] = gamma;
    Be->c[2] = SUN_RCONST(1.0);;
    Be->A[1][0] = gamma;
    Be->A[2][0] = delta;
    Be->A[2][1] = SUN_RCONST(1.0)-delta;
    Be->b[0] = delta;
    Be->b[1] = SUN_RCONST(1.0)-delta;
    Be->q = 2;
  }

  flag = ARKStepSetTables(*arkstep_mem, 2, 0, Bi, Be);
  if (check_flag(flag, "ARKStepSetTables")) { return 1; }
  if (Be) { ARKodeButcherTable_Free(Be); }
  if (Bi) { ARKodeButcherTable_Free(Bi); }

  // Wrap as a SUNStepper
  flag = ARKodeCreateSUNStepper(*arkstep_mem, &steppers[1]);
  if (check_flag(flag, "ARKodeCreateSUNStepper")) { return 1; }


  // ----------------------------
  // Create the Strang integrator
  // ----------------------------

  // Create SplittingStep integrator
  *arkode_mem = SplittingStepCreate(steppers, 2, ZERO, y, ctx);
  if (check_ptr(*arkode_mem, "SplittingStepCreate")) { return 1; }

  // Set fixed step size
  if (uopts.fixed_h > ZERO)
  {
    flag = ARKodeSetFixedStep(*arkode_mem, uopts.fixed_h);
    if (check_flag(flag, "ARKodeSetFixedStep")) { return 1; }
  }
  else
  {
    std::cerr << "ERROR: Fixed step size must be specified for Strang splitting." << std::endl;
    return 1;
  }

  // Attach user data
  flag = ARKodeSetUserData(*arkode_mem, &udata);
  if (check_flag(flag, "ARKodeSetUserData")) { return 1; }

  // Set Strang coefficients
  SplittingStepCoefficients coefficients =
        SplittingStepCoefficients_LoadCoefficientsByName("ARKODE_SPLITTING_STRANG_2_2_2");
  if (check_ptr(coefficients, "SplittingStepCoefficients_LoadCoefficientsByName"))
  { return 1;}
  flag = SplittingStepSetCoefficients(*arkode_mem, coefficients);
  if (check_flag(flag, "SplittingStepSetCoefficients")) { return 1; }
  SplittingStepCoefficients_Destroy(&coefficients);

  // Set max steps between outputs
  flag = ARKodeSetMaxNumSteps(*arkode_mem, uopts.maxsteps);
  if (check_flag(flag, "ARKodeSetMaxNumSteps")) { return 1; }

  // Set stopping time
  flag = ARKodeSetStopTime(*arkode_mem, udata.tf);
  if (check_flag(flag, "ARKodeSetStopTime")) { return 1; }

  return 0;
}

// -----------------------------------------------------------------------------
// Custom inner stepper functions
// -----------------------------------------------------------------------------

// Advance one step of the STS IVP
int STSInnerStepper_Evolve(MRIStepInnerStepper sts_mem, sunrealtype t0,
                           sunrealtype tout, N_Vector y)
{
  void* inner_content = nullptr;
  int flag            = MRIStepInnerStepper_GetContent(sts_mem, &inner_content);
  if (check_flag(flag, "MRIStepInnerStepper_GetContent")) { return -1; }

  STSInnerStepperContent* content = (STSInnerStepperContent*)inner_content;

  // Reset STS integrator to current state
  flag = ARKodeReset(content->sts_arkode_mem, t0, y);
  if (check_flag(flag, "ARKodeReset")) { return 1; }

  // Set step size to get to tout in a single step
  flag = ARKodeSetFixedStep(content->sts_arkode_mem, tout - t0);
  if (check_flag(flag, "ARKodeSetFixedStep")) { return 1; }

  // Set stop time
  flag = ARKodeSetStopTime(content->sts_arkode_mem, tout);
  if (check_flag(flag, "ARKodeSetStopTime")) { return 1; }

  // Evolve a single time step
  sunrealtype tret;
  flag = ARKodeEvolve(content->sts_arkode_mem, tout, y, &tret, ARK_ONE_STEP);
  if (check_flag(flag, "ARKodeEvolve")) { return flag; }

  return 0;
}

// Compute the RHS of the diffusion IVP
int STSInnerStepper_FullRhs(MRIStepInnerStepper sts_mem, sunrealtype t,
                            N_Vector y, N_Vector f, int mode)
{
  void* inner_content = nullptr;
  int flag            = MRIStepInnerStepper_GetContent(sts_mem, &inner_content);
  if (check_flag(flag, "MRIStepInnerStepper_GetContent")) { return -1; }

  STSInnerStepperContent* content = (STSInnerStepperContent*)inner_content;

  flag = f_diffusion(t, y, f, content->user_data);
  if (flag) { return -1; }

  return 0;
}

// Reset the fast integrator to the given time and state
int STSInnerStepper_Reset(MRIStepInnerStepper sts_mem, sunrealtype tR, N_Vector yR)
{
  void* inner_content = nullptr;
  int flag            = MRIStepInnerStepper_GetContent(sts_mem, &inner_content);
  if (check_flag(flag, "MRIStepInnerStepper_GetContent")) { return -1; }

  STSInnerStepperContent* content = (STSInnerStepperContent*)inner_content;

  // Reset STS integrator to current state
  flag = ARKodeReset(content->sts_arkode_mem, tR, yR);
  if (check_flag(flag, "ARKodeReset")) { return 1; }

  return 0;
}

// -----------------------------------------------------------------------------
// Functions called by the integrator
// -----------------------------------------------------------------------------

// Advection RHS function
int f_advection(sunrealtype t, N_Vector y, N_Vector f, void* user_data)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  // Access data arrays
  sunrealtype* ydata = N_VGetArrayPointer(y);
  if (check_ptr(ydata, "N_VGetArrayPointer")) { return -1; }
  sunrealtype* fdata = N_VGetArrayPointer(f);
  if (check_ptr(fdata, "N_VGetArrayPointer")) { return -1; }

  // Compute advection RHS
  const sunrealtype cux = ONE * udata->cux / (TWO * udata->dx);
  const sunrealtype cuy = ONE * udata->cuy / (TWO * udata->dy);
  const sunrealtype cvx = ONE * udata->cvx / (TWO * udata->dx);
  const sunrealtype cvy = ONE * udata->cvy / (TWO * udata->dy);
  const sunindextype nx = udata->nx;
  const sunindextype ny = udata->ny;
  N_VConst(ZERO, f);
  for (sunindextype j = 0; j < ny; j++)//periodic boundary conditions
  {
    for (sunindextype i = 0; i < nx; i++)
    {
      const sunrealtype ulx = (i > 0)      ? ydata[UIDX(i - 1, j, nx)] : ydata[UIDX(nx - 1, j, nx)];
      const sunrealtype urx = (i < nx - 1) ? ydata[UIDX(i + 1, j, nx)] : ydata[UIDX(0, j, nx)];
      const sunrealtype uby = (j > 0)      ? ydata[UIDX(i, j - 1, nx)] : ydata[UIDX(i, ny - 1, nx)];
      const sunrealtype uty = (j < ny - 1) ? ydata[UIDX(i, j + 1, nx)] : ydata[UIDX(i, 0, nx)];

      const sunrealtype vlx = (i > 0)      ? ydata[VIDX(i - 1, j, nx)] : ydata[VIDX(nx - 1, j, nx)];
      const sunrealtype vrx = (i < nx - 1) ? ydata[VIDX(i + 1, j, nx)] : ydata[VIDX(0, j, nx)];
      const sunrealtype vby = (j > 0)      ? ydata[VIDX(i, j - 1, nx)] : ydata[VIDX(i, ny - 1, nx)];
      const sunrealtype vty = (j < ny - 1) ? ydata[VIDX(i, j + 1, nx)] : ydata[VIDX(i, 0, nx)];

      fdata[UIDX(i, j, nx)] = cux * (urx - ulx) + cuy * (uty - uby);
      fdata[VIDX(i, j, nx)] = cvx * (vrx - vlx) + cvy * (vty - vby);
    }
  }

  return 0;
}

// Diffusion RHS function
int f_diffusion(sunrealtype t, N_Vector y, N_Vector f, void* user_data)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  // Access data arrays
  sunrealtype* ydata = N_VGetArrayPointer(y);
  if (check_ptr(ydata, "N_VGetArrayPointer")) { return -1; }

  sunrealtype* fdata = N_VGetArrayPointer(f);
  if (check_ptr(fdata, "N_VGetArrayPointer")) { return -1; }

  // Compute diffusion RHS
  const sunrealtype d = udata->d;
  const sunrealtype dxinv2 = ONE / (udata->dx * udata->dx);
  const sunrealtype dyinv2 = ONE / (udata->dy * udata->dy);
  const sunindextype nx = udata->nx;
  const sunindextype ny = udata->ny;
  N_VConst(ZERO, f);
  for (sunindextype j = 0; j < ny; j++) //periodic boundary conditions
  {
    for (sunindextype i = 0; i < nx; i++)
    {
      const sunrealtype uc  = ydata[UIDX(i, j, nx)];
      const sunrealtype ulx = (i > 0)      ? ydata[UIDX(i - 1, j, nx)] : ydata[UIDX(nx - 1, j, nx)];
      const sunrealtype urx = (i < nx - 1) ? ydata[UIDX(i + 1, j, nx)] : ydata[UIDX(0, j, nx)];
      const sunrealtype uby = (j > 0)      ? ydata[UIDX(i, j - 1, nx)] : ydata[UIDX(i, ny - 1, nx)];
      const sunrealtype uty = (j < ny - 1) ? ydata[UIDX(i, j + 1, nx)] : ydata[UIDX(i, 0, nx)];

      const sunrealtype vc  = ydata[VIDX(i, j, nx)];
      const sunrealtype vlx = (i > 0)      ? ydata[VIDX(i - 1, j, nx)] : ydata[VIDX(nx - 1, j, nx)];
      const sunrealtype vrx = (i < nx - 1) ? ydata[VIDX(i + 1, j, nx)] : ydata[VIDX(0, j, nx)];
      const sunrealtype vby = (j > 0)      ? ydata[VIDX(i, j - 1, nx)] : ydata[VIDX(i, ny - 1, nx)];
      const sunrealtype vty = (j < ny - 1) ? ydata[VIDX(i, j + 1, nx)] : ydata[VIDX(i, 0, nx)];

      fdata[UIDX(i, j, nx)] = d * dxinv2 * (ulx + urx - TWO * uc)
                            + d * dyinv2 * (uby + uty - TWO * uc);
      fdata[VIDX(i, j, nx)] = d * dxinv2 * (vlx + vrx - TWO * vc)
                            + d * dyinv2 * (vby + vty - TWO * vc);
    }
  }

  return 0;
}

// Reaction RHS function
int f_reaction(sunrealtype t, N_Vector y, N_Vector f, void* user_data)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  // Access data arrays
  sunrealtype* ydata = N_VGetArrayPointer(y);
  if (check_ptr(ydata, "N_VGetArrayPointer")) { return -1; }
  sunrealtype* fdata = N_VGetArrayPointer(f);
  if (check_ptr(fdata, "N_VGetArrayPointer")) { return -1; }

  // Compute reaction RHS
  N_VConst(ZERO, f);
  for (sunindextype j = 0; j < udata->ny; j++) //periodic boundary conditions
  {
    for (sunindextype i = 0; i < udata->nx; i++)
    {
      const sunrealtype u = ydata[UIDX(i, j, udata->nx)];
      const sunrealtype v = ydata[VIDX(i, j, udata->nx)];

      fdata[UIDX(i, j, udata->nx)] = udata->A + u * u * v - (udata->B + ONE) * u;
      fdata[VIDX(i, j, udata->nx)] = udata->B * u - u * u * v;
    }
  }

  return 0;
}

// Reaction Jacobian function
int J_reaction(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  // Access data array
  sunrealtype* ydata = N_VGetArrayPointer(y);
  if (check_ptr(ydata, "N_VGetArrayPointer")) { return 1; }
  const sunindextype nx = udata->nx;
  SUNMatZero(J);
  for (sunindextype j = 0; j < udata->ny; j++) //periodic boundary conditions
  {
    for (sunindextype i = 0; i < udata->nx; i++)
    {
      const sunrealtype u = ydata[UIDX(i, j, nx)];
      const sunrealtype v = ydata[VIDX(i, j, nx)];

      // all vars wrt u
      SM_ELEMENT_B(J, UIDX(i, j, nx), UIDX(i, j, nx)) = TWO * u * v - (udata->B + ONE);
      SM_ELEMENT_B(J, VIDX(i, j, nx), UIDX(i, j, nx)) = udata->B    - TWO * u * v;

      // all vars wrt v
      SM_ELEMENT_B(J, UIDX(i, j, nx), VIDX(i, j, nx)) =  u * u;
      SM_ELEMENT_B(J, VIDX(i, j, nx), VIDX(i, j, nx)) = -u * u;
    }
  }

  return 0;
}

// "Local" Reaction RHS function for BBD preconditioner
int floc_reaction(sunindextype Nloc, sunrealtype t, N_Vector y, N_Vector f,
                  void* user_data)
{
  return f_reaction(t, y, f, user_data);
}

// Advection-diffusion RHS function
int f_adv_diff(sunrealtype t, N_Vector y, N_Vector f, void* user_data)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  // Compute advection
  int flag = f_advection(t, y, f, user_data);
  if (flag) { return flag; }

  // Compute diffusion
  flag = f_diffusion(t, y, udata->temp_v, user_data);
  if (flag) { return flag; }

  // Combine advection and reaction
  N_VLinearSum(ONE, f, ONE, udata->temp_v, f);

  return 0;
}

// Diffusion-reaction RHS function
int f_diff_react(sunrealtype t, N_Vector y, N_Vector f, void* user_data)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  // Compute diffusion
  int flag = f_diffusion(t, y, f, user_data);
  if (flag) { return flag; }

  // Compute reactions
  flag = f_reaction(t, y, udata->temp_v, user_data);
  if (flag) { return flag; }

  // Combine advection and reaction
  N_VLinearSum(ONE, f, ONE, udata->temp_v, f);

  return 0;
}

// Advection-reaction RHS function
int f_adv_react(sunrealtype t, N_Vector y, N_Vector f, void* user_data)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  // Compute advection
  int flag = f_advection(t, y, f, user_data);
  if (flag) { return flag; }

  // Compute reaction
  flag = f_reaction(t, y, udata->temp_v, user_data);
  if (flag) { return flag; }

  // Combine advection and reaction
  N_VLinearSum(ONE, f, ONE, udata->temp_v, f);

  return 0;
}

// Advection-diffusion-reaction RHS function
int f_adv_diff_react(sunrealtype t, N_Vector y, N_Vector f, void* user_data)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  // Compute advection
  int flag = f_advection(t, y, f, user_data);
  if (flag) { return flag; }

  // Compute diffusion
  flag = f_diffusion(t, y, udata->temp_v, user_data);
  if (flag) { return flag; }

  // Combine advection and reaction
  N_VLinearSum(ONE, f, ONE, udata->temp_v, f);

  // Compute reactions
  flag = f_reaction(t, y, udata->temp_v, user_data);
  if (flag) { return flag; }

  // Combine advection and reaction
  N_VLinearSum(ONE, f, ONE, udata->temp_v, f);

  return 0;
}

// Diffusion RHS function with MRI forcing
int f_diffusion_forcing(sunrealtype t, N_Vector y, N_Vector f, void* user_data)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  // Compute diffusion RHS
  int flag = f_diffusion(t, y, f, user_data);
  if (flag) { return flag; }

  // Apply inner forcing for MRI + LSRKStep
  flag = MRIStepInnerStepper_AddForcing(udata->sts_mem, t, f);
  if (check_flag(flag, "MRIStepInnerStepper_AddForcing")) { return -1; }

  return 0;
}

// Dominant eigenvalue function (for diffusion operator in LSRKStep)
int diffusion_domeig(sunrealtype t, N_Vector y, N_Vector fn,
                     sunrealtype* lambdaR, sunrealtype* lambdaI, void* user_data,
                     N_Vector temp1, N_Vector temp2, N_Vector temp3)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  // Fill in spectral radius value
  *lambdaR = -SUN_RCONST(4.0) * udata->d / udata->dx / udata->dx
             -SUN_RCONST(4.0) * udata->d / udata->dy / udata->dy;
  *lambdaI = SUN_RCONST(0.0);

  return 0;
}

// Compute the initial condition
int SetIC(N_Vector y, UserData& udata)
{
  sunrealtype* ydata = N_VGetArrayPointer(y);
  if (check_ptr(ydata, "N_VGetArrayPointer")) { return -1; }

  for (sunindextype j = 0; j < udata.ny; j++)
  {
    const sunrealtype y = udata.yl + j * udata.dy;
    for (sunindextype i = 0; i < udata.nx; i++)
    {
      const sunrealtype x = udata.xl + i * udata.dx;
      ydata[UIDX(i, j, udata.nx)] = 22.0 * y * SUNRpowerR((ONE - y), 1.5);
      ydata[VIDX(i, j, udata.nx)] = 27.0 * x * SUNRpowerR((ONE - x), 1.5);
    }
  }

  return 0;
}

//---- end of file ----
