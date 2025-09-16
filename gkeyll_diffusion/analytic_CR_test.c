#include <float.h>
#include <gkyl_array.h>
#include <gkyl_array_ops.h>
#include <gkyl_comm_io.h>
#include <gkyl_dg_bin_ops.h>
#include <gkyl_dynvec.h>
#include <gkyl_eval_on_nodes.h>
#include <gkyl_gk_geometry.h>
#include <gkyl_gk_geometry_mapc2p.h>
#include <gkyl_null_comm.h>
#include <gkyl_proj_on_basis.h>
#include <gkyl_range.h>
#include <gkyl_rect_decomp.h>
#include <gkyl_rect_grid.h>
#include <gkyl_util.h>
#include <gkyl_velocity_map.h>
#include <mpack.h>

#include <rt_arg_parse.h>
#include <time.h>

#include "src/nvector_gkylzero.h"
#include "src/input_handler.h"

#include <arkode/arkode_erkstep.h>  /* prototypes for ERKStep fcts., consts */
#include <arkode/arkode_lsrkstep.h> /* prototypes for LSRKStep fcts., consts */

#include <sundomeigest/sundomeigest_power.h> /* access to Power Iteration module */
#include <sundomeigest/sundomeigest_arnoldi.h> /* access to Arnoldi Iteration module */

// Struct with context parameters.
struct analytic_ctx
{
  char name[128]; // Simulation name.

  int cells[1]; // Number of cells.
  double t_end;   // Final simulation time.
};

struct analytic_ctx create_analytic_ctx(void)
{
  // Create the context with all the inputs for this simulation.
  struct analytic_ctx ctx = {
    .name  = "analytic_CR_test", // App name.
    .t_end = 10.0, // Final simulation time.
  };
  return ctx;
}

// Struct with inputs to our app.
struct gkyl_analytic_app_inp
{
  char name[128]; // Name of the app.
  bool use_gpu;   // Whether to run on GPU.
  struct gkyl_comm *comm; // Communicator to use.

  int cells[1]; // Number of cells.
  sunrealtype lambda; // Stiffness parameter.

  // Initial condition.
  void (*initial_f_func)(double t, const double* xn, double* fout, void* ctx);
  void* initial_f_ctx; // Context.
};


void init_distf_1x1v(double t, const double* xn, double* restrict fout, void* ctx)
{
  // Initial condition.
  double x = xn[0], vpar = xn[1];

  struct analytic_ctx* dctx = ctx;

  fout[0] = 0.0;
}

// Main struct containing all our objects.
struct gkyl_analytic_app
{
  char name[128]; // Name of the app.
  bool use_gpu;   // Whether to run on the GPU.

  sunrealtype lambda; // Stiffness parameter.
  struct gkyl_comm* comm;          // Communicator object.

  struct gkyl_range local; // Local range. 
  struct gkyl_array *f;
  
  double tcurr; // Current simulation time.
};

struct gkyl_analytic_app* gkyl_analytic_app_new(struct gkyl_analytic_app_inp* inp)
{
  // Create the analytic app.
  struct gkyl_analytic_app* app = gkyl_malloc(sizeof(struct gkyl_analytic_app));

  strcpy(app->name, inp->name);

  app->use_gpu = inp->use_gpu;
  app->comm = inp->comm;

  app->lambda = inp->lambda;

  int lower[] = {1}, upper[] = {2};
  gkyl_range_init(&app->local, 1, lower, upper);
  printf("vol = %d\n",app->local.volume);

  app->f = mkarr(app->use_gpu, 1, app->local.volume);

  // Aliases for simplicity.
  bool use_gpu = app->use_gpu;

  return app;
}

int flag; /* reusable error-checking flag */

/* Check function return value...
    opt == 0 means function allocates memory so check if
             returned NULL pointer
    opt == 1 means function returns a flag so check if
             flag >= 0
    opt == 2 means function allocates memory so check if returned
             NULL pointer
*/
static int check_flag(void* flagvalue, const char* funcname, int opt)
{
  int* errflag;

  /* Check if function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL)
  {
    fprintf(stderr, "\nERROR: %s() failed - returned NULL pointer\n\n", funcname);
    return 1;
  }

  /* Check if flag != 0 */
  else if (opt == 1)
  {
    errflag = (int*)flagvalue;
    if (*errflag != 0)
    {
      fprintf(stderr, "\nERROR: %s() failed with flag = %d\n\n", funcname,
              *errflag);
      return 1;
    }
  }

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL)
  {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return 1;
  }

  return 0;
}

/* ----------------------------------------------------------------------------------
   ----------------------------------------------------------------------------------

                  Below NVector involving interface functions are defined.

   ----------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------- */

// Test the NVector interface
void test_NVector(bool use_gpu);

/* f routine to compute the ODE RHS function f(t,y). */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
  struct gkyl_analytic_app* app = (struct gkyl_analytic_app*)user_data;

  struct gkyl_array* fin  = N_VGetVector_Gkylzero(y);
  struct gkyl_array* fout = N_VGetVector_Gkylzero(ydot);

  gkyl_array_clear(fout, 0.0);

  sunrealtype lambda = app->lambda;

  gkyl_array_set(fout, lambda, fin);
  gkyl_array_shiftc(fout, SUN_RCONST(1.0) / (SUN_RCONST(1.0) + t * t) - lambda * atan(t), 0);

  return 0; /* return with success */
}

/* true_sol routine to compute the exact solution. */
static int true_sol(sunrealtype t, N_Vector y, void* user_data)
{
  struct gkyl_analytic_app* app = (struct gkyl_analytic_app*)user_data;

  struct gkyl_array* f  = N_VGetVector_Gkylzero(y);

  gkyl_array_clear(f, 0.0);

  gkyl_array_shiftc(f, atan(t), 0);

  return 0; /* return with success */
}

/* dom_eig routine to estimate the dominated eigenvalue */
static int dom_eig(sunrealtype t, N_Vector y, N_Vector fn, sunrealtype* lambdaR,
                   sunrealtype* lambdaI, void* user_data, N_Vector temp1,
                   N_Vector temp2, N_Vector temp3)
{
  struct gkyl_analytic_app* app = (struct gkyl_analytic_app*)user_data;

  printf("Estimated dominant eigenvalue = %f\n", app->lambda);

  *lambdaR = -app->lambda;
  *lambdaI = SUN_RCONST(0.0);
  return 0; /* return with success */
}

sunrealtype reltol; /* tolerances */
sunrealtype abstol;

// Error weight function for cellwise norm of y_{n-1}
int efun_cell_norm(N_Vector x, N_Vector w, void* user_data)
{
  struct gkyl_array* xdptr = NV_CONTENT_GKZ(x)->dataptr;
  struct gkyl_array* wdptr = NV_CONTENT_GKZ(w)->dataptr;

  gkyl_array_error_denom_fac(wdptr, reltol, abstol, xdptr);

  return 0;
}

/* general problem parameters */
sunrealtype T0 = 0.0; /* initial time */

int STS_init(struct gkyl_analytic_app* app, UserData* udata, N_Vector* y, void** arkode_mem)
{
  /* Create the SUNDIALS context object for this simulation */
  SUNContext sunctx;
  flag = SUNContext_Create(SUN_COMM_NULL, &sunctx);
  if (check_flag(&flag, "SUNContext_Create", 1)) { return 1; }

  /* Check if *y is NULL */
  if (*y == NULL)
  {
    fprintf(stderr, "*y is NULL\n");
    return 1;
  }
  /* Call LSRKStepCreateSTS to initialize the ARK timestepper module and
     specify the right-hand side function in y'=f(t,y), the initial time
     T0, and the initial dependent variable vector y. */
  *arkode_mem = LSRKStepCreateSTS(f, T0, *y, sunctx);
  if (check_flag((void*)*arkode_mem, "LSRKStepCreateSTS", 0)) { return 1; }

  /* Set routines */
  flag = ARKodeSetUserData(*arkode_mem, (void*)app); /* Pass the user data */
  if (check_flag(&flag, "ARKodeSetUserData", 1)) { return 1; }

  /* Specify tolerances */
  flag = ARKodeSStolerances(*arkode_mem, udata->rtol, udata->atol);
  if (check_flag(&flag, "ARKStepSStolerances", 1)) { return 1; }

  SUNDomEigEstimator DEE     = NULL; /* domeig estimator object */

  if (udata->user_dom_eig)
  {
    /* Specify user provided spectral radius */
    flag = LSRKStepSetDomEigFn(*arkode_mem, dom_eig);
    if (check_flag(&flag, "LSRKStepSetDomEigFn", 1)) { return 1; }
  }
  else
  {
    /* Set the initial random eigenvector for the DEE */
    struct gkyl_array* fdde_init = mkarr(app->use_gpu, 1, 1);
    gkyl_array_shiftc(fdde_init, 1.0, 0);
    
    N_Vector ydde_init    = NULL;
    ydde_init = N_VMake_Gkylzero(fdde_init, app->use_gpu, app->comm, &app->local, sunctx);

    if(udata->dee_id == 0)
    {
      DEE = SUNDomEigEstimator_Power(ydde_init, udata->dee_max_iters, udata->dee_reltol, sunctx);
      if (check_flag(DEE, "SUNDomEigEstimator_Power", 0)) { return 1; }
    }
    else if(udata->dee_id == 1)
    {
      DEE = SUNDomEigEstimator_Arnoldi(ydde_init, udata->dee_krylov_dim, sunctx);
      if (check_flag(DEE, "SUNDomEigEstimator_Arnoldi", 0)) { return 1; }
    }
    else
    {
      fprintf(stderr, "ERROR: Invalid DEE id %d\n", udata->dee_id);
      return 1;
    }

    flag = LSRKStepSetDomEigEstimator(*arkode_mem, DEE);
    if (check_flag(&flag, "LSRKStepSetDomEigEstimator", 1)) { return 1; }

    flag = LSRKStepSetNumDomEigEstInitPreprocessIters(*arkode_mem, udata->dee_num_init_wups);
    if (check_flag(&flag, "LSRKStepSetNumDomEigEstInitPreprocessIters", 1)) { return 1; }

    flag = LSRKStepSetNumDomEigEstPreprocessIters(*arkode_mem, udata->dee_num_succ_wups);
    if (check_flag(&flag, "LSRKStepSetNumDomEigEstPreprocessIters", 1)) { return 1; }
  }

  /* Specify after how many successful steps dom_eig is recomputed
     Note that nsteps = 0 refers to constant dominant eigenvalue */
  flag = LSRKStepSetDomEigFrequency(*arkode_mem, udata->eigfrequency);
  if (check_flag(&flag, "LSRKStepSetDomEigFrequency", 1)) { return 1; }

  /* Specify max number of stages allowed */
  flag = LSRKStepSetMaxNumStages(*arkode_mem, udata->stage_max_limit);
  if (check_flag(&flag, "LSRKStepSetMaxNumStages", 1)) { return 1; }

  /* Specify max number of steps allowed */
  flag = ARKodeSetMaxNumSteps(*arkode_mem, udata->maxsteps);
  if (check_flag(&flag, "ARKodeSetMaxNumSteps", 1)) { return 1; }

  /* Specify safety factor for user provided dom_eig */
  flag = LSRKStepSetDomEigSafetyFactor(*arkode_mem, udata->eigsafety);
  if (check_flag(&flag, "LSRKStepSetDomEigSafetyFactor", 1)) { return 1; }

  return 0;
}

int SSP_init(struct gkyl_analytic_app* app, UserData* udata, N_Vector* y, void** arkode_mem)
{
  /* Create the SUNDIALS context object for this simulation */
  SUNContext sunctx;
  flag = SUNContext_Create(SUN_COMM_NULL, &sunctx);
  if (check_flag(&flag, "SUNContext_Create", 1)) { return 1; }

  /* Check if *y is NULL */
  if (*y == NULL)
  {
    fprintf(stderr, "*y is NULL\n");
    return 1;
  }
  /* Call LSRKStepCreateSTS to initialize the ARK timestepper module and
     specify the right-hand side function in y'=f(t,y), the initial time
     T0, and the initial dependent variable vector y. */
  *arkode_mem = LSRKStepCreateSSP(f, T0, *y, sunctx);
  if (check_flag((void*)*arkode_mem, "LSRKStepCreateSSP", 0)) { return 1; }

  /* Set routines */
  flag = ARKodeSetUserData(*arkode_mem, (void*)app); /* Pass the user data */
  if (check_flag(&flag, "ARKodeSetUserData", 1)) { return 1; }

  /* Specify tolerances */
  flag = ARKodeSStolerances(*arkode_mem, udata->rtol, udata->atol);
  if (check_flag(&flag, "ARKStepSStolerances", 1)) { return 1; }

  /* Specify max number of steps allowed */
  flag = ARKodeSetMaxNumSteps(*arkode_mem, udata->maxsteps);
  if (check_flag(&flag, "ARKodeSetMaxNumSteps", 1)) { return 1; }

  return 0;
}

int gkyl_analytic_update(struct gkyl_analytic_app* app, void* arkode_mem, double tout, N_Vector y, sunrealtype* tcurr)
{
  // Call integrator to evolve the solution to time tout
  int flag = ARKodeEvolve(arkode_mem, tout, y, tcurr, ARK_NORMAL);
  if (check_flag(&flag, "ARKodeEvolve", 1)) { return 1; }

  // Check for any CUDA errors during time step
  if (app->use_gpu) checkCuda(cudaGetLastError());
  return 0;
}

double compute_max_error(N_Vector u, N_Vector v, sunrealtype  t_curr, struct gkyl_analytic_app* app)
{
  struct gkyl_array* udptr = NV_CONTENT_GKZ(u)->dataptr;
  struct gkyl_array* vdptr = NV_CONTENT_GKZ(v)->dataptr;
  double error             = -DBL_MAX;

  // TODO: change code so these allocations only happen once.
  int ncomp = udptr->ncomp;
  struct gkyl_array* wdptr; // Temporary buffer. Should change code to avoid this.
  double* red_ho = gkyl_malloc(ncomp * sizeof(double));
  double* red;
  if (app->use_gpu)
  {
    red   = gkyl_cu_malloc(ncomp * sizeof(double));
    wdptr = mkarr(true, ncomp, udptr->size);
  }
  else
  {
    red   = gkyl_malloc(ncomp * sizeof(double));
    wdptr = mkarr(false, ncomp, udptr->size);
  }

  gkyl_array_set(wdptr, 1.0, udptr);
  gkyl_array_accumulate(wdptr, -1.0, vdptr);
  gkyl_array_reduce(red, wdptr, GKYL_ABS_MAX);

  if (app->use_gpu)
    gkyl_cu_memcpy(red_ho, red, ncomp * sizeof(double), GKYL_CU_MEMCPY_D2H);
  else memcpy(red_ho, red, ncomp * sizeof(double));

  // Reduce over components.
  for (int i = 0; i < ncomp; i++) error = fmax(error, fabs(red_ho[i]));

  gkyl_free(red_ho);
  if (app->use_gpu) gkyl_cu_free(red);
  else gkyl_free(red);

  gkyl_array_release(wdptr);

  printf("\nmax error is %e at t = %g over the whole domain\n", error, t_curr);

  return error;
}

sunbooleantype is_SSP = SUNFALSE;

clock_t start, end, ref_start, ref_end;
double ref_time = 0.0;
double sol_time = 0.0;

int main(int argc, char* argv[])
{
  UserData* udata  = NULL; // user data structure

  // Allocate and initialize user data structure with default values. The
  // defaults may be overwritten by command line inputs in ReadInputs below.
  udata = (UserData*) malloc(sizeof(UserData));
  if (udata == NULL) {
    fprintf(stderr, "ERROR: failed to allocate memory for UserData\n");
    return 1;
  }

  flag = InitUserData(udata);
  if (check_flag(&flag, "InitUserData", 1)) { 
    free(udata);
    return 1;
  }

  // Parse command line inputs
  flag = ReadInputs(argc, argv, udata);
  if (flag != 0) { return 1; }

  struct gkyl_app_args app_args = parse_app_args(argc, argv);

  // Construct communicator for use in app.
  struct gkyl_comm *comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = app_args.use_gpu
    }
  );
   
  // Create the context struct.
  struct analytic_ctx ctx = create_analytic_ctx(); 

  // Update the context with user inputs.
  ctx.t_end = udata->tf;
  reltol = udata->rtol;
  abstol = udata->atol;

  if(udata->method == ARKODE_LSRK_RKC_2 || udata->method == ARKODE_LSRK_RKL_2)
  {
    is_SSP = SUNFALSE;
  }
  else if(udata->method == ARKODE_LSRK_SSP_S_2 ||
          udata->method == ARKODE_LSRK_SSP_S_3 ||
          udata->method == ARKODE_LSRK_SSP_10_4)
  {
    is_SSP = SUNTRUE;
    if(udata->method == ARKODE_LSRK_SSP_10_4 && udata->num_SSP_stages != 10)
    {
      udata->num_SSP_stages = 10; // Set to 10 for ARKODE_LSRK_SSP_10_
      fprintf(stderr, "\nWARNING: num_SSP_stages reset to default 10 for ARKODE_LSRK_SSP_10_4\n");
    }
  }
  else
  {
    fprintf(stderr, "ERROR: Invalid method %d\n", udata->method);
    return 1;
  }

  // Output problem setup/options
  flag = PrintUserData(udata, 0);
  if (check_flag(&flag, "PrintUserData", 1)) { return 1; }

  // Create the struct of app inputs. 
  struct gkyl_analytic_app_inp app_inp = {
    // Initial condition.
    .initial_f_func = init_distf_1x1v,
    .initial_f_ctx  = &ctx,
    .lambda = -10.0, // Stiffness parameter.
    
    .use_gpu = app_args.use_gpu, // Whether to run on GPU.
    .comm = comm,
  };
  strcpy(app_inp.name, ctx.name);

  // Create app object.
  struct gkyl_analytic_app* app = gkyl_analytic_app_new(&app_inp);

  // Initial and final simulation times.
  double t_curr = 0.0, t_end = ctx.t_end;

  double dt; // Initial tout time.

  void* arkode_mem = NULL; /* empty ARKode memory structure */
  N_Vector y       = NULL; /* empty vector for storing solution */
  N_Vector yref    = NULL; /* empty vector for storing reference solution */

  app->use_gpu = SUNFALSE; // For now, run everything on CPU.
  printf("Hardcoded app->use_gpu = false for now\n");

  struct gkyl_array*    f = mkarr(app->use_gpu, 1, 1);
  struct gkyl_array* fref = mkarr(app->use_gpu, 1, 1);

  // compute the reference solution and the error
  SUNContext sunctx;
  flag = SUNContext_Create(SUN_COMM_NULL, &sunctx);
  if (check_flag(&flag, "SUNContext_Create", 1)) { return 1; }

  app->f = gkyl_array_clone(f);
  y    = N_VMake_Gkylzero(app->f, app->use_gpu, app->comm, &app->local, sunctx);
  yref = N_VMake_Gkylzero(fref, app->use_gpu, app->comm, &app->local, sunctx);
  
  true_sol(ZERO, y, (void*)app); // Compute initial condition.

  if (!is_SSP)
  {
    int flag;
    flag = STS_init(app, udata, &y, &arkode_mem);
    if (check_flag(&flag, "STS_init", 1)) { return 1; }

    /* Specify the STS method */
    flag = LSRKStepSetSTSMethod(arkode_mem, udata->method);
    if (check_flag(&flag, "LSRKStepSetSTSMethod", 1)) { return 1; }
  }
  else if (is_SSP)
  {
    int flag;
    flag = SSP_init(app, udata, &y, &arkode_mem);
    if (check_flag(&flag, "SSP_init", 1)) { return 1; }

    /* Specify the SSP method */
    flag = LSRKStepSetSSPMethod(arkode_mem, udata->method);
    if (check_flag(&flag, "LSRKStepSetSSPMethod", 1)) { return 1; }

    /* Specify the number of SSP stages */
    flag = LSRKStepSetNumSSPStages(arkode_mem, udata->num_SSP_stages);
    if (check_flag(&flag, "LSRKStepSetNumSSPStages", 1)) { return 1; }
  }

  // Set fixed step size or adaptivity method
  if (udata->hfixed > ZERO)
  {
    flag = ARKodeSetFixedStep(arkode_mem, udata->hfixed);
    if (check_flag(&flag, "ARKodeSetFixedStep", 1)) { return 1; }
  }

  /* Specify the Ewt function */
  switch (udata->wrms_norm_type)
  {
  case 1:
    y->ops->nvwrmsnorm = N_VWrmsNorm_abs_comp_Gkylzero;
    printf("\nUsing WRMSNorm with componentwise absolute values\n");
    break;

  case 2:
    flag = ARKodeWFtolerances(arkode_mem, efun_cell_norm);
    if (check_flag(&flag, "ARKodeWFtolerances", 1)) { return 1; }
    y->ops->nvwrmsnorm = N_VWrmsNorm_cell_norm_Gkylzero;
    printf("\nUsing WRMSNorm with cellwise norm values\n");
    break;
  }

  printf("\nNumber of cells             = %ld", app->f->size);
  printf("\nNumber of DoFs in each cell = %ld", app->f->ncomp);
  printf("\nNumber of DoFs              = %ld\n", app->f->size * app->f->ncomp);

  double tout      = 0;
  double max_error = 0.0;

  long step = 1;
  while ((t_end - t_curr > 1.0e-10) && (step <= app_args.num_steps))
  {
    if (step == 1)
    {
      dt   = udata->tf / udata->nout;
      flag = ARKodeGetCurrentTime(arkode_mem, &t_curr);
      if (check_flag(&flag, "ARKodeGetCurrentTime", 1)) { return 1; }
    }
    tout += dt;

    fprintf(stdout, "\nTaking time-step %ld at t = %g ...", step, t_curr);

    // Update the reference solution
    ref_start = clock();
    true_sol(tout, yref, (void*)app);
    ref_end   = clock();
    ref_time += ((double)(ref_end - ref_start)) / CLOCKS_PER_SEC;

    // Update the computed solution
    start = clock();
    flag  = gkyl_analytic_update(app, arkode_mem, tout, y, &t_curr);
    if (check_flag(&flag, "gkyl_analytic_update", 1)) 
    {
      fprintf(stdout, "** Update method failed! Aborting simulation ....\n");
      break;
    }
    end = clock();
    sol_time += ((double)(end - start)) / CLOCKS_PER_SEC;

    // Compute the error between the reference and computed solutions
    max_error = fmax(compute_max_error(y, yref, t_curr, app), max_error);

    step++;
  }

  // printf("\nReference Solution Stats\n");
  // ARKodePrintAllStats(arkode_mem_ref, stdout, SUN_OUTPUTFORMAT_TABLE);
  printf("\nComputed Solution Stats\n");
  ARKodePrintAllStats(arkode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);

  printf("\nmax-in-space and max-in-time error is %e over D x [%g, %g]\n\n", max_error, T0, t_curr);

  printf("Reference solution CPU time: %f seconds\n", ref_time);
  printf(" Computed solution CPU time: %f seconds\n", sol_time);

  // Free the app.
  gkyl_array_release(fref);
  gkyl_comm_release(comm);

  return 0;
}
