/* -----------------------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ SMU
 * Based on the SUNDIALS example ark_advection_diffusion_reaction.cpp by
 * David J. Gardner @ LLNL
 * -----------------------------------------------------------------------------
 * This example simulates the 1D advection-diffusion-reaction equation,
 *
 *   u_t = -c u_x + d u_xx + A - (w + 1) * u + v * u^2
 *   v_t = -c v_x + d u_xx + w * u - v * u^2
 *   w_t = -c w_x + d w_xx + (B - w) / eps - w * u
 *
 * where u, v, and w represent the concentrations of chemical species, c = 0.01
 * is the advection speed, d = 0.1 is the diffusion rate, and the species with
 * constant concentration over time are A = 0.6 and B = 2.0.
 *
 * The problem is evolved for t in [0, 3] and x in [0, 1], with initial
 * conditions given by
 *
 *   u(0,x) =  A  + 0.1 * sin(pi * x)
 *   v(0,x) = B/A + 0.1 * sin(pi * x)
 *   w(0,x) =  B  + 0.1 * sin(pi * x)
 *
 * and stationary boundary conditions i.e.,
 *
 *   u_t(t,0) = u_t(t,1) = 0,
 *   v_t(t,0) = v_t(t,1) = 0,
 *   w_t(t,0) = w_t(t,1) = 0.
 *
 * The system is advanced in time using one of the following approaches based on
 * the --integrator <int> flag value. The following options are available:
 *
 *   0. An explicit Runge-Kutta method with ERKStep.
 *
 *   1. An IMEX Runge-Kutta method with ARKStep.  Both reaction and diffusion
 *      are treated implicitly, while advection is evolved explicitly in time.
 *
 *      Note: either of the advection and reaction operators can be disabled
 *      using the flags --no-advection and/or --no-reaction.
 *
 *   2. An extended super-time-stepping method that combines MRIStep and LSRKStep.
 *      Here, diffusion is treated explicitly using a STS method, advection
 *      is treated explicitly using the ExtSTS method, and reaction is treated
 *      implicitly using the ExtSTS method.
 *
 *      Note: either of the advection and reaction operators can be disabled
 *      using the flags --no-advection and/or --no-reaction.
 *
 * Several command line options are available to change the problem parameters
 * and integrator settings. Use the flag --help for more information.
 * ---------------------------------------------------------------------------*/

#include "advection_diffusion_reaction.hpp"

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

  // Matrix and linear solver for IMEX or ExtSTS integrators
  SUNMatrix A           = nullptr;
  SUNLinearSolver LS    = nullptr;
  SUNMatrix Aref        = nullptr;
  SUNLinearSolver LSref = nullptr;

  // STS integrator for ExtSTS method
  MRIStepInnerStepper sts_mem = nullptr;

  // Create integrator
  switch (uopts.integrator)
  {
  case (0):
    flag = SetupERK(ctx, udata, uopts, y, &arkode_mem);
    break;
  case (1):
    flag = SetupARK(ctx, udata, uopts, y, &A, &LS, &arkode_mem);
    break;
  case (2):
    flag = SetupExtSTS(ctx, udata, uopts, y, &A, &LS, &sts_mem, &arkode_mem);
    break;
  default: flag = -1;
  }
  if (check_flag(flag, "Integrator setup")) { return 1; }

  // Create reference solver (4th-order ARK with tighter relative tolerance)
  if (uopts.calc_error)
  {
    flag = SetupARK(ctx, udata, uopts, yref, &Aref, &LSref, &arkref_mem);
    if (check_flag(flag, "Reference solver setup")) { return 1; }
    flag = ARKodeSStolerances(arkref_mem, 1e-10, uopts.atol);
    if (check_flag(flag, "ARKodeSStolerances")) { return 1; }
    flag = ARKodeSetOrder(arkref_mem, 4);
    if (check_flag(flag, "ARKodeSetOrder")) { return 1; }
  }


  // ----------------------
  // Evolve problem in time
  // ----------------------

  // Initial time, time between outputs, output time
  sunrealtype t     = ZERO;
  sunrealtype t2    = ZERO;
  sunrealtype dTout = udata.tf / uopts.nout;
  sunrealtype tout  = dTout;

  // Accumulated error
  sunrealtype total_error = ZERO;

  // Initial output
  flag = OpenOutput(udata, uopts);
  if (check_flag(flag, "OpenOutput")) { return 1; }

  flag = (uopts.calc_error) ?
    WriteOutput(t, y, yerr, udata, uopts) :
    WriteOutput(t, y, udata, uopts);
  if (check_flag(flag, "WriteOutput")) { return 1; }

  // Loop over output times
  for (int iout = 0; iout < uopts.nout; iout++)
  {
    // Evolve
    if ((uopts.output == 3) || (uopts.calc_error))
    {
      // Stop at output time (do not interpolate output)
      flag = ARKodeSetStopTime(arkode_mem, tout);
      if (check_flag(flag, "ARKodeSetStopTime")) { return 1; }
    }

    //   Advance in time
    flag = ARKodeEvolve(arkode_mem, tout, y, &t, ARK_NORMAL);
    if (check_flag(flag, "ARKodeEvolve")) { return 1; }

    // Advance reference solution and compute error
    if (uopts.calc_error)
    {
      flag = ARKodeSetStopTime(arkref_mem, tout);
      if (check_flag(flag, "ARKodeSetStopTime")) { return 1; }
      flag = ARKodeEvolve(arkref_mem, tout, yref, &t2, ARK_NORMAL);
      if (check_flag(flag, "ARKodeEvolve (ref)")) { return 1; }
      N_VLinearSum(1.0, y, -1.0, yref, yerr);
      total_error += N_VDotProd(yerr, yerr);
    }

    // Output solution
    flag = (uopts.calc_error) ?
      WriteOutput(t, y, yerr, udata, uopts) :
      WriteOutput(t, y, udata, uopts);
    if (check_flag(flag, "WriteOutput")) { return 1; }

    // Update output time
    tout += dTout;
    tout = (tout > udata.tf) ? udata.tf : tout;
  }

  // Close output
  flag = CloseOutput(uopts);
  if (check_flag(flag, "CloseOutput")) { return 1; }

  // ------------
  // Output stats
  // ------------

  if (uopts.output)
  {
    cout << "Final integrator statistics:" << endl;
    if (uopts.calc_error)
    {
      cout << "  Solution error     = " << setprecision(2)
           << sqrt(total_error / uopts.nout / udata.nx) << endl;
    }
    switch (uopts.integrator)
    {
    case (0): flag = OutputStatsERK(arkode_mem, udata); break;
    case (1): flag = OutputStatsARK(arkode_mem, udata); break;
    case (2): flag = OutputStatsExtSTS(arkode_mem, sts_mem, udata); break;
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
  }
  if (uopts.calc_error)
  {
    ARKodeFree(&arkref_mem);
    N_VDestroy(yref);
    N_VDestroy(yerr);
    SUNMatDestroy(Aref);
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

  if (udata.reaction && udata.advection)
  {
    // Explicit -- advection-diffusion-reaction
    f_RHS = f_adv_diff_react;
  }
  else if (!udata.reaction && udata.advection)
  {
    // Explicit -- advection-diffusion
    f_RHS = f_adv_diff;
  }
  else if (udata.reaction && !udata.advection)
  {
    // Explicit -- diffusion-reaction
    f_RHS = f_diff_react;
  }
  else
  {
    cerr << "ERROR: Invalid problem configuration" << endl;
    return -1;
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
             SUNMatrix* A, SUNLinearSolver* LS, void** arkode_mem)
{
  // Problem configuration
  ARKRhsFn fe_RHS;   // explicit RHS function
  ARKRhsFn fi_RHS;   // implicit RHS function
  ARKLsJacFn Ji_RHS; // Jacobian of implicit RHS function

  // advection-diffusion-reaction
  if (udata.reaction && udata.advection)
  {
    fe_RHS = f_advection;
    fi_RHS = f_diff_react;
    Ji_RHS = J_diff_react;
  }
  // advection-diffusion
  else if (!udata.reaction && udata.advection)
  {
    fe_RHS = f_advection;
    fi_RHS = f_diffusion;
    Ji_RHS = J_diffusion;
  }
  // diffusion-reaction
  else if (udata.reaction && !udata.advection)
  {
    fe_RHS = nullptr;
    fi_RHS = f_diff_react;
    Ji_RHS = J_diff_react;
  }
  // diffusion
  else if (!udata.reaction && !udata.advection)
  {
    fe_RHS = nullptr;
    fi_RHS = f_diffusion;
    Ji_RHS = J_diffusion;
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

  // Create banded matrix
  *A = SUNBandMatrix(udata.neq, 3, 3, ctx);
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

  // Tighten implicit solver tolerances
  flag = ARKodeSetNonlinConvCoef(*arkode_mem, 1.e-1);
  if (check_flag(flag, "ARKodeSetNonlinConvCoef")) { return 1; }
  flag = ARKodeSetEpsLin(*arkode_mem, 1.e-1);
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

  if (uopts.linear)
  {
    // Specify linearly implicit non-time-dependent RHS
    flag = ARKodeSetLinear(*arkode_mem, SUNFALSE);
    if (check_flag(flag, "ARKodeSetLinear")) { return 1; }
  }

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
                SUNMatrix* A, SUNLinearSolver* LS,  MRIStepInnerStepper* sts_mem,
                void** arkode_mem)
{
  // Problem configuration
  ARKRhsFn fe_RHS;     // explicit RHS function
  ARKRhsFn fi_RHS;     // implicit RHS function
  ARKLsJacFn Ji_RHS;   // implicit RHS Jacobian function

  fe_RHS = (udata.advection) ? f_advection : nullptr;
  fi_RHS = (udata.reaction)  ? f_reaction  : nullptr;
  Ji_RHS = (udata.reaction)  ? J_reaction  : nullptr;

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
  ARKODE_LSRKMethodType ststype = (uopts.sts_method == 0) ? ARKODE_LSRK_RKC_2 : ARKODE_LSRK_RKL_2;
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
  inner_content->sts_arkode_mem = sts_arkode_mem;
  inner_content->user_data      = &udata;

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

  // Create slow integrator for diffusion and attach fast integrator
  *arkode_mem = MRIStepCreate(fe_RHS, fi_RHS, ZERO, y, *sts_mem, ctx);
  if (check_ptr(*arkode_mem, "MRIStepCreate")) { return 1; }

  // Set fixed step size
  if (uopts.fixed_h > ZERO)
  {
    flag = ARKodeSetFixedStep(*arkode_mem, uopts.fixed_h);
    if (check_flag(flag, "ARKodeSetFixedStep")) { return 1; }
  }

  // Specify tolerances
  flag = ARKodeSStolerances(*arkode_mem, uopts.rtol, uopts.atol);
  if (check_flag(flag, "ARKodeSStolerances")) { return 1; }

  // Attach user data
  flag = ARKodeSetUserData(*arkode_mem, &udata);
  if (check_flag(flag, "ARKodeSetUserData")) { return 1; }

  // If implicit, setup solvers
  if (udata.reaction)
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

    // Tighten implicit solver tolerances
    flag = ARKodeSetNonlinConvCoef(*arkode_mem, 1.e-1);
    if (check_flag(flag, "ARKodeSetNonlinConvCoef")) { return 1; }
    flag = ARKodeSetEpsLin(*arkode_mem, 1.e-1);
    if (check_flag(flag, "ARKodeSetEpsLin")) { return 1; }

    // Use "deduce implicit RHS" option
    flag = ARKodeSetDeduceImplicitRhs(*arkode_mem, SUNTRUE);
    if (check_flag(flag, "ARKodeSetDeduceImplicitRhs")) { return 1; }

    // Set the predictor method
    flag = ARKodeSetPredictorMethod(*arkode_mem, uopts.predictor);
    if (check_flag(flag, "ARKodeSetPredictorMethod")) { return 1; }

  }

  // Select ExtSTS method via MRIStepCoupling structure
  MRIStepCoupling C;
  if (udata.advection && udata.reaction)  // advection + diffusion + reaction
  {
    if (uopts.extsts_method == 0)  // ARS(2,2,2)
    {
      C = MRIStepCoupling_Alloc(1, 5, MRISTEP_IMEX);
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
      C->W[0][5][2] = delta - one / three;
      C->W[0][5][4] = one / three;
      C->G[0][1][0] =  gamma;
      C->G[0][2][0] = -gamma;
      C->G[0][2][2] =  gamma;
      C->G[0][3][2] =  one - gamma;
      C->G[0][4][2] = -gamma;
      C->G[0][4][4] =  gamma;
      C->G[0][5][2] = -one / three;
      C->G[0][5][4] =  one / three;
    }
    else                           // Giraldo ARK2
    {
      C = MRIStepCoupling_Alloc(1, 6, MRISTEP_IMEX);
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
  }
  else if (!udata.reaction)  // advection + diffusion -or- just diffusion (both are fully explicit)
  {
    if (uopts.extsts_method == 0)  // Ralston
    {
      C = MRIStepCoupling_Alloc(1, 3, MRISTEP_EXPLICIT);
      const sunrealtype one = SUN_RCONST(1.0);
      const sunrealtype two = SUN_RCONST(2.0);
      const sunrealtype three = SUN_RCONST(3.0);
      const sunrealtype four = SUN_RCONST(4.0);
      C->q = 2;
      C->p = 1;
      C->c[1] = two / three;
      C->c[2] = one;
      C->W[0][1][0] = two / three;
      C->W[0][2][0] = one / four - two / three;
      C->W[0][2][1] = three / four;
      C->W[0][3][0] = SUN_RCONST(5.0)/SUN_RCONST(37.0) - two / three;
      C->W[0][3][1] = two / three - three / four;
      C->W[0][3][2] = SUN_RCONST(22.0)/SUN_RCONST(111.0);
    }
    else                           // Heun-Euler
    {
      C = MRIStepCoupling_Alloc(1, 3, MRISTEP_EXPLICIT);
      const sunrealtype one = SUN_RCONST(1.0);
      const sunrealtype two = SUN_RCONST(2.0);
      const sunrealtype three = SUN_RCONST(3.0);
      C->q = 2;
      C->p = 1;
      C->c[1] = one;
      C->c[2] = one;
      C->W[0][1][0] = one;
      C->W[0][2][0] = -one / two;
      C->W[0][2][1] = one / two;
    }
  }
  else if (!udata.advection && udata.reaction)  // diffusion + reaction
  {
    if (uopts.extsts_method == 0)  // SSP SDIRK 2
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
    else                           // Giraldo DIRK2
    {
      C = MRIStepCoupling_Alloc(1, 6, MRISTEP_IMPLICIT);
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
  }
  else   // illegal configuration
  {
    cerr << "ERROR: Invalid problem configuration" << endl;
    return -1;
  }
  flag = MRIStepSetCoupling(*arkode_mem, C);
  if (check_flag(flag, "MRIStepSetCoupling")) { return 1; }
  MRIStepCoupling_Free(C);

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
  int flag = MRIStepInnerStepper_GetContent(sts_mem, &inner_content);
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
  int flag = MRIStepInnerStepper_GetContent(sts_mem, &inner_content);
  if (check_flag(flag, "MRIStepInnerStepper_GetContent")) { return -1; }

  STSInnerStepperContent* content = (STSInnerStepperContent*)inner_content;

  flag = f_diffusion(t, y, f, content->user_data);
  if (flag) { return -1; }

  return 0;
}

// Reset the fast integrator to the given time and state
int STSInnerStepper_Reset(MRIStepInnerStepper sts_mem, sunrealtype tR,
                            N_Vector yR)
{
  void* inner_content = nullptr;
  int flag = MRIStepInnerStepper_GetContent(sts_mem, &inner_content);
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
  sunrealtype ul, ur;
  sunrealtype vl, vr;
  sunrealtype wl, wr;

  sunrealtype c = -ONE * udata->c / (TWO * udata->dx);

  fdata[0] = fdata[1] = fdata[2] = ZERO;

  for (sunindextype i = 1; i < udata->nx - 1; i++)
  {
    ul = ydata[UIDX(i - 1)];
    ur = ydata[UIDX(i + 1)];

    vl = ydata[VIDX(i - 1)];
    vr = ydata[VIDX(i + 1)];

    wl = ydata[WIDX(i - 1)];
    wr = ydata[WIDX(i + 1)];

    fdata[UIDX(i)] = c * (ur - ul);
    fdata[VIDX(i)] = c * (vr - vl);
    fdata[WIDX(i)] = c * (wr - wl);
  }

  fdata[udata->neq - 3] = fdata[udata->neq - 2] = fdata[udata->neq - 1] = ZERO;

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
  sunrealtype ul, uc, ur;
  sunrealtype vl, vc, vr;
  sunrealtype wl, wc, wr;

  sunrealtype d = udata->d / (udata->dx * udata->dx);

  fdata[0] = fdata[1] = fdata[2] = ZERO;

  for (sunindextype i = 1; i < udata->nx - 1; i++)
  {
    ul = ydata[UIDX(i - 1)];
    uc = ydata[UIDX(i)];
    ur = ydata[UIDX(i + 1)];

    vl = ydata[VIDX(i - 1)];
    vc = ydata[VIDX(i)];
    vr = ydata[VIDX(i + 1)];

    wl = ydata[WIDX(i - 1)];
    wc = ydata[WIDX(i)];
    wr = ydata[WIDX(i + 1)];

    fdata[UIDX(i)] = d * (ul - TWO * uc + ur);
    fdata[VIDX(i)] = d * (vl - TWO * vc + vr);
    fdata[WIDX(i)] = d * (wl - TWO * wc + wr);
  }

  fdata[udata->neq - 3] = fdata[udata->neq - 2] = fdata[udata->neq - 1] = ZERO;

  return 0;
}

// Diffusion Jacobian function
int J_diffusion(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  sunrealtype d = udata->d / (udata->dx * udata->dx);

  for (sunindextype i = 1; i < udata->nx - 1; i++)
  {
    SM_ELEMENT_B(J, UIDX(i), UIDX(i - 1)) = d;
    SM_ELEMENT_B(J, UIDX(i), UIDX(i))     = -d * TWO;
    SM_ELEMENT_B(J, UIDX(i), UIDX(i + 1)) = d;

    SM_ELEMENT_B(J, VIDX(i), VIDX(i - 1)) = d;
    SM_ELEMENT_B(J, VIDX(i), VIDX(i))     = -d * TWO;
    SM_ELEMENT_B(J, VIDX(i), VIDX(i + 1)) = d;

    SM_ELEMENT_B(J, WIDX(i), WIDX(i - 1)) = d;
    SM_ELEMENT_B(J, WIDX(i), WIDX(i))     = -d * TWO;
    SM_ELEMENT_B(J, WIDX(i), WIDX(i + 1)) = d;
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
  sunrealtype u, v, w;

  fdata[0] = fdata[1] = fdata[2] = ZERO;

  for (sunindextype i = 1; i < udata->nx - 1; i++)
  {
    u = ydata[UIDX(i)];
    v = ydata[VIDX(i)];
    w = ydata[WIDX(i)];

    fdata[UIDX(i)] = udata->A - (w + ONE) * u + v * u * u;
    fdata[VIDX(i)] = w * u - v * u * u;
    fdata[WIDX(i)] = ((udata->B - w) / udata->eps) - w * u;
  }

  fdata[udata->neq - 3] = fdata[udata->neq - 2] = fdata[udata->neq - 1] = ZERO;

  return 0;
}

// Diffusion Jacobian function
int J_reaction(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  // Access data array
  sunrealtype* ydata = N_VGetArrayPointer(y);
  if (check_ptr(ydata, "N_VGetArrayPointer")) { return 1; }

  sunrealtype u, v, w;

  for (sunindextype i = 1; i < udata->nx - 1; i++)
  {
    u = ydata[UIDX(i)];
    v = ydata[VIDX(i)];
    w = ydata[WIDX(i)];

    // all vars wrt u
    SM_ELEMENT_B(J, UIDX(i), UIDX(i)) = -(w + ONE) + TWO * u * v;
    SM_ELEMENT_B(J, VIDX(i), UIDX(i)) = w - TWO * u * v;
    SM_ELEMENT_B(J, WIDX(i), UIDX(i)) = -w;

    // all vars wrt v
    SM_ELEMENT_B(J, UIDX(i), VIDX(i)) = u * u;
    SM_ELEMENT_B(J, VIDX(i), VIDX(i)) = -u * u;

    // all vars wrt w
    SM_ELEMENT_B(J, UIDX(i), WIDX(i)) = -u;
    SM_ELEMENT_B(J, VIDX(i), WIDX(i)) = u;
    SM_ELEMENT_B(J, WIDX(i), WIDX(i)) = (-ONE / udata->eps) - u;
  }

  return 0;
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

// Diffusion-reaction Jacobian function
int J_diff_react(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                 void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  // Compute diffusion Jacobian
  int flag = J_diffusion(t, y, fy, J, user_data, tmp1, tmp2, tmp3);
  if (flag) { return flag; }

  // Compute reaction Jacobian
  flag = SUNMatZero(udata->temp_J);
  if (flag) { return flag; }

  flag = J_reaction(t, y, fy, udata->temp_J, user_data, tmp1, tmp2, tmp3);
  if (flag) { return flag; }

  // Combine Jacobians
  flag = SUNMatScaleAdd(ONE, J, udata->temp_J);
  if (flag) { return -1; }

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
                     sunrealtype* lambdaR, sunrealtype* lambdaI,
                     void* user_data, N_Vector temp1, N_Vector temp2,
                     N_Vector temp3)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  // Fill in spectral radius value
  *lambdaR = -SUN_RCONST(4.0) * udata->d / udata->dx / udata->dx;
  *lambdaI = SUN_RCONST(0.0);

  return 0;
}

// Compute the initial condition
int SetIC(N_Vector y, UserData& udata)
{
  sunrealtype* ydata = N_VGetArrayPointer(y);
  if (check_ptr(ydata, "N_VGetArrayPointer")) { return -1; }

  sunrealtype x, p;

  for (sunindextype i = 0; i < udata.nx; i++)
  {
    x              = udata.xl + i * udata.dx;
    p              = SUN_RCONST(0.1) * sin(PI * x);
    ydata[UIDX(i)] = udata.A + p;
    ydata[VIDX(i)] = udata.B / udata.A + p;
    ydata[WIDX(i)] = udata.B + p;
  }

  return 0;
}

//---- end of file ----
