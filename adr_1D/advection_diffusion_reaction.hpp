/* -----------------------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ UMBC
 * Based on the SUNDIALS example ark_advection_diffusion_reaction.cpp by
 * David J. Gardner @ LLNL
 * -----------------------------------------------------------------------------
 * Header file for advection-diffusion-reaction equation example, see
 * advection_diffusion_reaction.cpp for more details.
 * ---------------------------------------------------------------------------*/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

// Include desired integrators, vectors, linear solvers, and nonlinear solvers
#include "arkode/arkode_erkstep.h"
#include "arkode/arkode_arkstep.h"
#include "arkode/arkode_lsrkstep.h"
#include "arkode/arkode_mristep.h"
#include "arkode/arkode_splittingstep.h"
#include "nvector/nvector_serial.h"
#include "sundials/sundials_core.hpp"
#include "sunlinsol/sunlinsol_spgmr.h"
#include "sunlinsol/sunlinsol_band.h"
#include "sunmatrix/sunmatrix_band.h"

// Macros for problem constants
#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)
#define TWO  SUN_RCONST(2.0)

#define NSPECIES 3

#define WIDTH (10 + numeric_limits<sunrealtype>::digits10)

// Macro to access each species at an x location
#define UIDX(i) (NSPECIES * (i))
#define VIDX(i) (NSPECIES * (i) + 1)
#define WIDX(i) (NSPECIES * (i) + 2)

using namespace std;

// -----------------------------------------------------------------------------
// Problem parameters
// -----------------------------------------------------------------------------

struct UserData
{
  // RHS options
  bool reaction = true;
  bool advection = true;

  // Advection and diffusion coefficients
  sunrealtype c = SUN_RCONST(1.0e-2);
  sunrealtype d = SUN_RCONST(1.0e-1);

  // Feed and reaction rates
  sunrealtype A = SUN_RCONST(0.6);
  sunrealtype B = SUN_RCONST(2.0);

  // Stiffness parameter
  sunrealtype eps = SUN_RCONST(1.0e-2);

  // Final simulation time
  sunrealtype tf = SUN_RCONST(3.0);

  // Domain boundaries
  sunrealtype xl = ZERO;
  sunrealtype xu = ONE;

  // Number of nodes
  sunindextype nx = 512;

  // Mesh spacing
  sunrealtype dx = (xu - xl) / (nx - 1);

  // Number of equations
  sunindextype neq = NSPECIES * nx;

  // Temporary workspace vector and matrix
  N_Vector temp_v  = nullptr;
  SUNMatrix temp_J = nullptr;

  // Inner stepper memory
  MRIStepInnerStepper sts_mem = nullptr;

  ~UserData();
};

UserData::~UserData()
{
  if (temp_v)
  {
    N_VDestroy(temp_v);
    temp_v = nullptr;
  }

  if (temp_J)
  {
    SUNMatDestroy(temp_J);
    temp_J = nullptr;
  }
}

// -----------------------------------------------------------------------------
// Problem options
// -----------------------------------------------------------------------------

struct UserOptions
{
  // Integration method (0 = ERK, 1 = ARK, 2 = ExtSTS, 3 = Splitting)
  int integrator = 1;

  // Table ID for ARK methods:
  //   0 = default
  //   1 = ARS(2,2,2)
  //   2 = Giraldo ARK2
  //   3 = Ralston
  //   4 = Heun-Euler
  //   5 = SSP SDIRK 2
  //   6 = Giraldo DIRK2
  int table_id = 0;

  // Method order
  int order = 2;

  // ExtSTS method options
  //   sts_method = 0 (RKC) or 1 (RKL)
  //   extsts_method:
  //   * advection+diffusion+reaction
  //         0 = ARS(2,2,2)
  //         1 = Giraldo ARK2
  //   * advection+diffusion
  //         0 = ARS(2,2,2) ERK
  //         1 = Giraldo ERK2
  //         2 = Ralston
  //         3 = Heun-Euler
  //   * diffusion+reaction
  //         0 = ARS(2,2,2) SDIRK
  //         1 = Giraldo DIRK2
  //         2 = SSP SDIRK 2
  int sts_method    = 0;
  int extsts_method = 0;

  // Relative and absolute tolerances
  sunrealtype rtol = SUN_RCONST(1.e-4);
  sunrealtype atol = SUN_RCONST(1.e-9);

  // Step size selection (ZERO = adaptive steps)
  sunrealtype fixed_h = ZERO;

  int maxsteps      = 10000; // max steps between outputs
  int predictor     = 0;     // predictor for nonlinear systems
  int ls_setup_freq = 0;     // linear solver setup frequency
  int maxl          = 0;     // maximum number of GMRES iterations

  bool linear = false; // signal that the problem is linearly implicit

  bool calc_error = false;
  bool write_solution = false;

  int output = 1;  // 0 = none, 1 = stats, 2 = disk, 3 = disk with tstop
  int nout   = 10; // number of output times
  ofstream uout;   // output file stream
};

// -----------------------------------------------------------------------------
// Custom inner stepper content and functions
// -----------------------------------------------------------------------------

struct STSInnerStepperContent
{
  void* sts_arkode_mem = nullptr; // LSRKStep memory structure
  void* user_data = nullptr; // user data pointer

  // saved integrator stats
  long int nst = 0; // time steps
  long int nfe = 0; // rhs evals
};

int STSInnerStepper_Evolve(MRIStepInnerStepper sts_mem, sunrealtype t0,
                           sunrealtype tout, N_Vector y);

int STSInnerStepper_FullRhs(MRIStepInnerStepper sts_mem, sunrealtype t,
                            N_Vector y, N_Vector f, int mode);

int STSInnerStepper_Reset(MRIStepInnerStepper sts_mem, sunrealtype tR,
                          N_Vector yR);

// -----------------------------------------------------------------------------
// Functions provided to the SUNDIALS integrators
// -----------------------------------------------------------------------------

// ODE right hand side (RHS) functions
int f_advection(sunrealtype t, N_Vector y, N_Vector f, void* user_data);
int f_diffusion(sunrealtype t, N_Vector y, N_Vector f, void* user_data);
int f_reaction(sunrealtype t, N_Vector y, N_Vector f, void* user_data);

int f_adv_diff(sunrealtype t, N_Vector y, N_Vector f, void* user_data);
int f_adv_react(sunrealtype t, N_Vector y, N_Vector f, void* user_data);
int f_diff_react(sunrealtype t, N_Vector y, N_Vector f, void* user_data);
int f_adv_diff_react(sunrealtype t, N_Vector y, N_Vector f, void* user_data);

int f_diffusion_forcing(sunrealtype t, N_Vector y, N_Vector f, void* user_data);

// Jacobian of RHS functions
int J_advection(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
int J_diffusion(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
int J_reaction(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
int J_adv_diff(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
int J_adv_react(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
int J_diff_react(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                 void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
int J_adv_diff_react(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                     void* user_data, N_Vector tmp1, N_Vector tmp2,
                     N_Vector tmp3);

// Dominant eigenvalue function (for diffusion operator in LSRKStep)
int diffusion_domeig(sunrealtype t, N_Vector y, N_Vector fn,
                     sunrealtype* lambdaR, sunrealtype* lambdaI,
                     void* user_data, N_Vector temp1, N_Vector temp2,
                     N_Vector temp3);

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

// Integrator setup functions
int SetupERK(SUNContext ctx, UserData& udata, UserOptions& uopts, N_Vector y,
             void** arkode_mem);

int SetupARK(SUNContext ctx, UserData& udata, UserOptions& uopts, N_Vector y,
             SUNLinearSolver* LS, void** arkode_mem);

int SetupExtSTS(SUNContext ctx, UserData& udata, UserOptions& uopts, N_Vector y,
                SUNMatrix* A, SUNLinearSolver* LS, MRIStepInnerStepper* sts_mem,
                void** arkode_mem);

int SetupStrang(SUNContext ctx, UserData& udata, UserOptions& uopts, N_Vector y,
                SUNMatrix* A, SUNLinearSolver* LS, SUNStepper steppers[2],
                void** lsrkstep_mem, void** arkstep_mem, void** arkode_mem);

int SetupReference(SUNContext ctx, UserData& udata, UserOptions& uopts, N_Vector y,
                   SUNMatrix* A, SUNLinearSolver* LS, void** arkode_mem);

// Compute the initial condition
int SetIC(N_Vector y, UserData& udata);

// -----------------------------------------------------------------------------
// Output and utility functions
// -----------------------------------------------------------------------------

// Check function return flag
static int check_flag(int flag, const string funcname)
{
  if (flag < 0)
  {
    cerr << "ERROR: " << funcname << " returned " << flag << endl;
    return 1;
  }
  return 0;
}

// Check if a function returned a NULL pointer
static int check_ptr(void* ptr, const string funcname)
{
  if (ptr) { return 0; }
  cerr << "ERROR: " << funcname << " returned NULL" << endl;
  return 1;
}

// Print ERK integrator statistics
static int OutputStatsERK(void* arkode_mem, UserData& udata)
{
  int flag;

  // Get integrator and solver stats
  long int nst, nst_a, netf, nfe;
  flag = ARKodeGetNumSteps(arkode_mem, &nst);
  if (check_flag(flag, "ARKodeGetNumSteps")) { return -1; }
  flag = ARKodeGetNumStepAttempts(arkode_mem, &nst_a);
  if (check_flag(flag, "ARKodeGetNumStepAttempts")) { return -1; }
  flag = ARKodeGetNumErrTestFails(arkode_mem, &netf);
  if (check_flag(flag, "ARKodeGetNumErrTestFails")) { return -1; }
  flag = ARKodeGetNumRhsEvals(arkode_mem, 0, &nfe);
  if (check_flag(flag, "ARKodeGetNumRhsEvals")) { return -1; }

  cout << "  Steps              = " << nst << endl;
  cout << "  Step attempts      = " << nst_a << endl;
  cout << "  Error test fails   = " << netf << endl;
  cout << "  RHS evals          = " << nfe << endl;

  return 0;
}

// Print ARK integrator statistics
static int OutputStatsARK(void* arkode_mem, UserData& udata)
{
  int flag;

  // Get integrator and solver stats
  long int nst, nst_a, netf, nfe, nfi, nfils;
  flag = ARKodeGetNumSteps(arkode_mem, &nst);
  if (check_flag(flag, "ARKodeGetNumSteps")) { return -1; }
  flag = ARKodeGetNumStepAttempts(arkode_mem, &nst_a);
  if (check_flag(flag, "ARKodeGetNumStepAttempts")) { return -1; }
  flag = ARKodeGetNumErrTestFails(arkode_mem, &netf);
  if (check_flag(flag, "ARKodeGetNumErrTestFails")) { return -1; }
  flag = ARKodeGetNumRhsEvals(arkode_mem, 0, &nfe);
  if (check_flag(flag, "ARKodeGetNumRhsEvals")) { return -1; }
  flag = ARKodeGetNumRhsEvals(arkode_mem, 1, &nfi);
  if (check_flag(flag, "ARKodeGetNumRhsEvals")) { return -1; }
  flag = ARKodeGetNumLinRhsEvals(arkode_mem, &nfils);
  if (check_flag(flag, "ARKodeGetNumLinRhsEvals")) { return -1; }

  cout << fixed << setprecision(6);
  cout << "  Steps              = " << nst << endl;
  cout << "  Step attempts      = " << nst_a << endl;
  cout << "  Error test fails   = " << netf << endl;
  cout << "  Explicit RHS evals = " << nfe << endl;
  cout << "  Implicit RHS evals = " << nfi + nfils << endl;

  long int nni, ncfn;
  flag = ARKodeGetNumNonlinSolvIters(arkode_mem, &nni);
  if (check_flag(flag, "ARKodeGetNumNonlinSolvIters")) { return -1; }
  flag = ARKodeGetNumNonlinSolvConvFails(arkode_mem, &ncfn);
  if (check_flag(flag, "ARKodeGetNumNonlinSolvConvFails")) { return -1; }

  long int nsetups, nje;
  flag = ARKodeGetNumLinSolvSetups(arkode_mem, &nsetups);
  if (check_flag(flag, "ARKodeGetNumLinSolvSetups")) { return -1; }
  flag = ARKodeGetNumJacEvals(arkode_mem, &nje);
  if (check_flag(flag, "ARKodeGetNumJacEvals")) { return -1; }

  cout << "  NLS iters          = " << nni << endl;
  cout << "  NLS fails          = " << ncfn << endl;
  cout << "  LS setups          = " << nsetups << endl;
  cout << "  J evals            = " << nje << endl;
  cout << endl;

  sunrealtype avgnli = (sunrealtype)nni / (sunrealtype)nst_a;
  sunrealtype avgls  = (sunrealtype)nsetups / (sunrealtype)nni;
  cout << "  Avg NLS iters per step attempt = " << avgnli << endl;
  cout << "  Avg LS setups per NLS iter     = " << avgls << endl;
  cout << endl;

  return 0;
}

// Print ExtSTS integrator statistics
static int OutputStatsExtSTS(void* arkode_mem, MRIStepInnerStepper sts_mem,
                             UserData& udata)
{
  int flag;

  // Print all ExtSTS integrator stats
  cout << fixed << setprecision(6);
  cout << endl << "ExtSTS Integrator:" << endl;
  flag = ARKodePrintAllStats(arkode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
  if (check_flag(flag, "ARKodePrintAllStats")) { return -1; }
  cout << endl;

  // Print inner sts integrator stats
  void* inner_content = nullptr;
  MRIStepInnerStepper_GetContent(sts_mem, &inner_content);
  STSInnerStepperContent* content = (STSInnerStepperContent*)inner_content;

  // Get STS integrator and solver stats
  cout << fixed << setprecision(6);
  cout << endl << "Inner STS Method:" << endl;
  flag = ARKodePrintAllStats(content->sts_arkode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
  if (check_flag(flag, "ARKodePrintAllStats")) { return -1; }

  return 0;
}

// Print Strang integrator statistics
static int OutputStatsStrang(void* arkode_mem, void* arkstep_mem, void* lsrkstep_mem,
                             UserData& udata)
{
  int flag;

  // Print all SplittingStep integrator stats
  cout << fixed << setprecision(6);
  cout << endl << "Strang Integrator:" << endl;
  flag = ARKodePrintAllStats(arkode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
  if (check_flag(flag, "ARKodePrintAllStats")) { return -1; }
  cout << endl;

  // Print all SplittingStep stats
  cout << endl << "ARKStep Stepper:" << endl;
  flag = ARKodePrintAllStats(arkstep_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
  if (check_flag(flag, "ARKodePrintAllStats")) { return -1; }
  cout << endl;

  // Print all LSRKStep stats
  cout << endl << "LSRKStep Stepper:" << endl;
  flag = ARKodePrintAllStats(lsrkstep_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
  if (check_flag(flag, "ARKodePrintAllStats")) { return -1; }
  cout << endl;

  return 0;
}

// Print command line options
static void InputHelp()
{
  cout << endl;
  cout << "Command line options:" << endl;
  cout << "  --no-advection           : disable advection\n";
  cout << "  --no-reaction            : disable reaction\n";
  cout << "  --c <real>               : advection coefficient\n";
  cout << "  --d <real>               : diffusion coefficient\n";
  cout << "  --A <real>               : species A concentration\n";
  cout << "  --B <real>               : species B concentration\n";
  cout << "  --eps <real>             : stiffness parameter\n";
  cout << "  --tf <real>              : final time\n";
  cout << "  --xl <real>              : domain lower boundary\n";
  cout << "  --xu <real>              : domain upper boundary\n";
  cout << "  --nx <int>               : number of mesh points\n";
  cout << "  --integrator <int>       : integrator option (0=ERK, 1=ARK, 2=ExtSTS, 3=Strang)\n";
  cout << "  --table_id <int>         : ARK table ID (0=default, 1=ARS, 2=GiraldoARK2, 3=Ralston, 4=Heun-Euler, 5=SSPSDIRK2, 6=GiraldoDIRK2)\n";
  cout << "  --order <int>            : method order\n";
  cout << "  --sts_method <int>       : STS method type (0=RKC, 1=RKL)\n";
  cout << "  --extsts_method <int>    : ExtSTS method type\n";
  cout << "                               advection+diffusion+reaction\n";
  cout << "                                   0 = ARS(2,2,2)\n";
  cout << "                                   1 = Giraldo ARK2\n";
  cout << "                               advection+diffusion\n";
  cout << "                                   0 = ARS(2,2,2) ERK\n";
  cout << "                                   1 = Giraldo ERK2\n";
  cout << "                                   2 = Ralston\n";
  cout << "                                   3 = Heun-Euler\n";
  cout << "                               diffusion+reaction\n";
  cout << "                                   0 = ARS(2,2,2) SDIRK\n";
  cout << "                                   1 = Giraldo DIRK2\n";
  cout << "                                   2 = SSP SDIRK 2\n";
  cout << "  --rtol <real>            : relative tolerance\n";
  cout << "  --atol <real>            : absolute tolerance\n";
  cout << "  --fixed_h <real>         : fixed step size\n";
  cout << "  --predictor <int>        : nonlinear solver predictor\n";
  cout << "  --lssetupfreq <int>      : LS setup frequency\n";
  cout << "  --maxl <int>             : max GMRES iterations\n";
  cout << "  --maxsteps <int>         : max steps between outputs\n";
  cout << "  --linear                 : linearly implicit\n";
  cout << "  --calc_error             : use reference solution to compute solution error\n";
  cout << "  --write_solution         : write the reference solution to disk\n";
  cout << "  --output <int>           : output level\n";
  cout << "  --nout <int>             : number of outputs\n";
  cout << "  --help                   : print options and exit\n";
}

inline void find_arg(vector<string>& args, const string key, sunrealtype& dest)
{
  auto it = find(args.begin(), args.end(), key);
  if (it != args.end())
  {
#if defined(SUNDIALS_SINGLE_PRECISION)
    dest = stof(*(it + 1));
#elif defined(SUNDIALS_DOUBLE_PRECISION)
    dest = stod(*(it + 1));
#elif defined(SUNDIALS_EXTENDED_PRECISION)
    dest = stold(*(it + 1));
#endif
    args.erase(it, it + 2);
  }
}

#if defined(SUNDIALS_INT64_T)
inline void find_arg(vector<string>& args, const string key, sunindextype& dest)
{
  auto it = find(args.begin(), args.end(), key);
  if (it != args.end())
  {
    dest = stoll(*(it + 1));
    args.erase(it, it + 2);
  }
}
#endif

inline void find_arg(vector<string>& args, const string key, int& dest)
{
  auto it = find(args.begin(), args.end(), key);
  if (it != args.end())
  {
    dest = stoi(*(it + 1));
    args.erase(it, it + 2);
  }
}

inline void find_arg(vector<string>& args, const string key, bool& dest,
                     bool store = true)
{
  auto it = find(args.begin(), args.end(), key);
  if (it != args.end())
  {
    dest = store;
    args.erase(it);
  }
}

static int ReadInputs(vector<string>& args, UserData& udata, UserOptions& uopts,
                      SUNContext ctx)
{
  if (find(args.begin(), args.end(), "--help") != args.end())
  {
    InputHelp();
    return 1;
  }

  // Problem parameters
  find_arg(args, "--no-advection", udata.advection, false);
  find_arg(args, "--no-reaction", udata.reaction, false);
  find_arg(args, "--c", udata.c);
  find_arg(args, "--d", udata.d);
  find_arg(args, "--A", udata.A);
  find_arg(args, "--B", udata.B);
  find_arg(args, "--eps", udata.eps);
  find_arg(args, "--tf", udata.tf);
  find_arg(args, "--xl", udata.xl);
  find_arg(args, "--xu", udata.xu);
  find_arg(args, "--nx", udata.nx);

  // Integrator options
  find_arg(args, "--integrator", uopts.integrator);
  find_arg(args, "--table_id", uopts.table_id);
  find_arg(args, "--order", uopts.order);
  find_arg(args, "--sts_method", uopts.sts_method);
  find_arg(args, "--extsts_method", uopts.extsts_method);
  find_arg(args, "--rtol", uopts.rtol);
  find_arg(args, "--atol", uopts.atol);
  find_arg(args, "--fixed_h", uopts.fixed_h);
  find_arg(args, "--predictor", uopts.predictor);
  find_arg(args, "--lssetupfreq", uopts.ls_setup_freq);
  find_arg(args, "--maxl", uopts.maxl);
  find_arg(args, "--maxsteps", uopts.maxsteps);
  find_arg(args, "--linear", uopts.linear);
  find_arg(args, "--calc_error", uopts.calc_error);
  find_arg(args, "--write_solution", uopts.write_solution);
  find_arg(args, "--output", uopts.output);
  find_arg(args, "--nout", uopts.nout);

  // Recompute mesh spacing and total number of nodes
  udata.dx  = (udata.xu - udata.xl) / (udata.nx - 1);
  udata.neq = NSPECIES * udata.nx;

  // Create workspace
  if ((uopts.integrator < 2) || uopts.calc_error)
  {
    udata.temp_v = N_VNew_Serial(udata.neq, ctx);
    if (check_ptr(udata.temp_v, "N_VNew_Serial")) { return -1; }
    N_VConst(ZERO, udata.temp_v);
  }
  if (uopts.calc_error)
  {
    udata.temp_J = SUNBandMatrix(udata.neq, 3, 3, ctx);
    if (check_ptr(udata.temp_J, "SUNBandMatrix")) { return -1; }
    SUNMatZero(udata.temp_J);
  }

  // Input checks
  if (!udata.reaction && !udata.advection)
  {
    cerr << "ERROR: Invalid problem configuration" << endl;
    return -1;
  }

  if (uopts.integrator < 0 || uopts.integrator > 3)
  {
    cerr << "ERROR: Invalid integrator option" << endl;
    return -1;
  }

  if (uopts.table_id < 0 || uopts.table_id > 6)
  {
    cerr << "ERROR: Invalid ARK table ID" << endl;
    return -1;
  }

  if (uopts.write_solution && !uopts.calc_error)
  {
    cerr << "ERROR: Cannot write_solution if calc_error is false"
         <<  " (since calc_error computes the reference solution)" << endl;
    return -1;
  }

  return 0;
}

// Print user data
static int PrintSetup(UserData& udata, UserOptions& uopts)
{
  cout << endl;
  cout << "Problem parameters and options:" << endl;
  cout << " --------------------------------- " << endl;
  cout << "  c                = " << udata.c << endl;
  cout << "  d                = " << udata.d << endl;
  cout << "  A                = " << udata.A << endl;
  cout << "  B                = " << udata.B << endl;
  cout << " --------------------------------- " << endl;
  cout << "  tf               = " << udata.tf << endl;
  cout << "  xl               = " << udata.xl << endl;
  cout << "  xu               = " << udata.xu << endl;
  cout << "  nx               = " << udata.nx << endl;
  cout << "  dx               = " << udata.dx << endl;
  cout << " --------------------------------- " << endl;

  if (uopts.integrator == 0)
  {
    cout << "  integrator       = ERK" << endl;
    if (udata.advection) { cout << "  advection        = Explicit" << endl; }
    else { cout << "  advection        = OFF" << endl; }
    if (udata.reaction) {  cout << "  reaction         = Explicit" << endl; }
    else { cout << "  reaction         = OFF" << endl; }
    cout << "  diffusion        = Explicit" << endl;
  }
  else if (uopts.integrator == 1)
  {
    cout << "  integrator       = ARK" << endl;
    if (udata.advection) { cout << "  advection        = Explicit" << endl; }
    else { cout << "  advection        = OFF" << endl; }
    if (udata.reaction) {  cout << "  reaction         = Implicit" << endl; }
    else { cout << "  reaction         = OFF" << endl; }
    cout << "  diffusion        = Implicit" << endl;
    cout << "  ARK table ID     = ";
    switch(uopts.table_id)
    {
      case 1: cout << "ARS(2,2,2)" << endl; break;
      case 2: cout << "Giraldo ARK2" << endl; break;
      case 3: cout << "Ralston" << endl; break;
      case 4: cout << "Heun-Euler" << endl; break;
      case 5: cout << "SSP SDIRK 2" << endl; break;
      case 6: cout << "Giraldo DIRK2" << endl; break;
      default: cout << "default" << endl;
    }
  }
  else if (uopts.integrator == 2)
  {
    cout << "  integrator       = ExtSTS" << endl;
    if (udata.advection) { cout << "  advection        = Explicit" << endl; }
    else { cout << "  advection        = OFF" << endl; }
    if (udata.reaction) {  cout << "  reaction         = Implicit" << endl; }
    else { cout << "  reaction         = OFF" << endl; }
    cout << "  diffusion        = Explicit" << endl;
  }
  else if (uopts.integrator == 3)
  {
    cout << "  integrator       = Strang" << endl;
    if (udata.advection) { cout << "  advection        = Explicit" << endl; }
    else { cout << "  advection        = OFF" << endl; }
    if (udata.reaction) {  cout << "  reaction         = Implicit" << endl; }
    else { cout << "  reaction         = OFF" << endl; }
    cout << "  diffusion        = Explicit" << endl;
  }
  else
  {
    cerr << "ERROR: Invalid integrator option" << endl;
    return -1;
  }

  cout << "  order            = " << uopts.order << endl;
  cout << "  rtol             = " << uopts.rtol << endl;
  cout << "  atol             = " << uopts.atol << endl;
  cout << "  fixed h          = " << uopts.fixed_h << endl;
  if (uopts.integrator > 0)
  {
    if (uopts.predictor == 0)
    {
      cout << "  predictor        = trivial" << endl;
    }
    else if (uopts.predictor == 1)
    {
      cout << "  predictor        = max order" << endl;
    }
    else if (uopts.predictor == 2)
    {
      cout << "  predictor        = variable order" << endl;
    }
    else if (uopts.predictor == 3)
    {
      cout << "  predictor        = cutoff order" << endl;
    }
    else { cout << "  predictor        = " << uopts.predictor << endl; }
    cout << "  ls setup freq    = " << uopts.ls_setup_freq << endl;
    cout << "  max GMRES iters  = " << uopts.maxl << endl;
    cout << "  linear           = " << uopts.linear << endl;
  }
  if (uopts.integrator == 2)
  {
    cout << " --------------------------------- " << endl;
    if (udata.advection && udata.reaction)  // advection + diffusion + reaction
    {
      if (uopts.extsts_method == 0)
      { cout << "  ExtSTS method    = ARKS(2,2,2)" << endl; }
      else
      { cout << "  ExtSTS method    = Giraldo ARK2" << endl; }
    }
    else if (!udata.reaction)  // advection + diffusion -or- just diffusion (both are fully explicit)
    {
      if (uopts.extsts_method == 0)
      { cout << "  ExtSTS method    = ARS(2,2,2) ERK" << endl; }
      else if (uopts.extsts_method == 1)
      { cout << "  ExtSTS method    = Giraldo ERK2" << endl; }
      else if (uopts.extsts_method == 2)
      { cout << "  ExtSTS method    = Ralston" << endl; }
      else
      { cout << "  ExtSTS method    = Heun-Euler" << endl; }
    }
    else if (!udata.advection && udata.reaction)  // diffusion + reaction
    {
      if (uopts.extsts_method == 0)
      { cout << "  ExtSTS method    = ARS(2,2,2) SDIRK" << endl; }
      else if (uopts.extsts_method == 1)
      { cout << "  ExtSTS method    = Giraldo DIRK2" << endl; }
      else
      { cout << "  ExtSTS method    = SSP SDIRK 2" << endl; }
    }
    if (uopts.sts_method == 0)
    {
      cout << "  STS method       = RKC" << endl;
    }
    else
    {
      cout << "  STS method       = RKL" << endl;
    }
  }
  if (uopts.integrator == 3)
  {
    cout << " --------------------------------- " << endl;
    cout << "  Strang:" << endl;
    cout << "     STS diffusion" << endl;
    if (udata.advection && udata.reaction)  // advection + diffusion + reaction
    {
      cout << "     ARS(2,2,2) ARK: implicit reaction + explicit advection" << endl;
    }
    else if (!udata.reaction)  // advection + diffusion
    {
      cout << "     ARS(2,2,2) ERK: explicit advection" << endl;
    }
    else if (!udata.advection && udata.reaction)  // diffusion + reaction
    {
      cout << "     ARS(2,2,2) DIRK: implicit reaction" << endl;
    }
  }
  cout << " --------------------------------- " << endl;
  if (uopts.calc_error)
  {
    cout << "  reference solver = ARK" << endl;
  }
  cout << "  output           = " << uopts.output << endl;
  cout << " --------------------------------- " << endl;
  cout << endl;

  return 0;
}

// Initialize output
static int OpenOutput(UserData& udata, UserOptions& uopts)
{
  // Header for status output
  if (uopts.output)
  {
    cout << scientific;
    cout << setprecision(numeric_limits<sunrealtype>::digits10);
    cout << "          t           ";
    cout << "          ||y||_rms      ";
    if (uopts.calc_error)
    {
      cout << "   ||yerr||_rms";
    }
    cout << endl;
    cout << " ---------------------";
    if (uopts.calc_error)
    {
      cout << "---------------";
    }
    cout << "-------------------------" << endl;
  }

  // Open output stream and output problem information
  if (uopts.output >= 2)
  {
    // Open output stream
    stringstream fname;
    fname << "advection_diffusion_reaction.out";
    uopts.uout.open(fname.str());

    uopts.uout << scientific;
    uopts.uout << setprecision(numeric_limits<sunrealtype>::digits10);
    uopts.uout << "# title Advection-Diffusion-Reaction (Brusselator)" << endl;
    uopts.uout << "# nvar 3" << endl;
    uopts.uout << "# vars u v w" << endl;
    uopts.uout << "# nt " << uopts.nout + 1 << endl;
    uopts.uout << "# nx " << udata.nx << endl;
    uopts.uout << "# xl " << udata.xl << endl;
    uopts.uout << "# xu " << udata.xu << endl;
  }

  return 0;
}

// Write output
static int WriteOutput(sunrealtype t, N_Vector y, UserData& udata,
                       UserOptions& uopts)
{
  if (uopts.output)
  {
    // Compute rms norm of the state
    sunrealtype urms = sqrt(N_VDotProd(y, y) / udata.nx);
    cout << setw(22) << t << setw(25) << urms << endl;

    // Write solution to disk
    if (uopts.output >= 2)
    {
      sunrealtype* ydata = N_VGetArrayPointer(y);
      if (check_ptr(ydata, "N_VGetArrayPointer")) { return -1; }

      uopts.uout << t;
      for (sunindextype i = 0; i < udata.nx; i++)
      {
        uopts.uout << setw(WIDTH) << ydata[UIDX(i)];
        uopts.uout << setw(WIDTH) << ydata[VIDX(i)];
        uopts.uout << setw(WIDTH) << ydata[WIDX(i)];
      }
      uopts.uout << endl;
    }
  }

  return 0;
}

// Write output
static int WriteOutput(sunrealtype t, N_Vector y, N_Vector yerr,
                       UserData& udata, UserOptions& uopts)
{
  if (uopts.output)
  {
    // Compute rms norm of the state and error
    sunrealtype urms = sqrt(N_VDotProd(y, y) / udata.nx / 3);
    sunrealtype erms = sqrt(N_VDotProd(yerr, yerr) / udata.nx / 3);
    cout << setw(22) << t << setw(25) << urms
         << setprecision(2) << setw(12) << erms
         << setprecision(numeric_limits<sunrealtype>::digits10) << endl;

    // Write solution to disk
    if (uopts.output >= 2)
    {
      sunrealtype* ydata = N_VGetArrayPointer(y);
      if (check_ptr(ydata, "N_VGetArrayPointer")) { return -1; }

      uopts.uout << t;
      for (sunindextype i = 0; i < udata.nx; i++)
      {
        uopts.uout << setw(WIDTH) << ydata[UIDX(i)];
        uopts.uout << setw(WIDTH) << ydata[VIDX(i)];
        uopts.uout << setw(WIDTH) << ydata[WIDX(i)];
      }
      uopts.uout << endl;
    }
  }

  return 0;
}

// Write solution to disk
static int WriteSolution(sunrealtype t, N_Vector y,
                        UserData& udata, UserOptions& uopts)
{
  if (uopts.write_solution)
  {
    sunrealtype* ydata = N_VGetArrayPointer(y);
    if (check_ptr(ydata, "N_VGetArrayPointer")) { return -1; }
    stringstream fname;
    fname << "reference.dat";
    ofstream uref;
    uref.open(fname.str());
    uref << setprecision(numeric_limits<sunrealtype>::digits10) << t;
    for (sunindextype i = 0; i< udata.nx; i++)
    {
      uref << setprecision(numeric_limits<sunrealtype>::digits10)
           << " " << ydata[UIDX(i)];
    }
    for (sunindextype i = 0; i< udata.nx; i++)
    {
      uref << setprecision(numeric_limits<sunrealtype>::digits10)
           << " " << ydata[VIDX(i)];
    }
    for (sunindextype i = 0; i< udata.nx; i++)
    {
      uref << setprecision(numeric_limits<sunrealtype>::digits10)
           << " " << ydata[WIDX(i)];
    }
    uref << endl;
    uref.close();
    cout << "Reference solution is written to reference.dat" << endl;
  }

  return 0;
}

// Finalize output
static int CloseOutput(UserOptions& uopts)
{
  // Footer for status output
  if (uopts.output)
  {
    cout << " ---------------------";
    if (uopts.calc_error)
    {
      cout << "---------------";
    }
    cout << "-------------------------" << endl;
    cout << endl;
  }

  // Close output streams
  if (uopts.output >= 2) { uopts.uout.close(); }

  return 0;
}

//---- end of file ----
