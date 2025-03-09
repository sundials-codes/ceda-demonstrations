#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Header files */
#include <arkode/arkode_lsrkstep.h> /* prototypes for LSRKStep fcts., consts */
#include <math.h>
#include <nvector/nvector_serial.h> /* serial N_Vector types, fcts., macros */
#include <stdio.h>
#include <sundials/sundials_math.h> /* def. of SUNRsqrt, etc. */
#include <sundials/sundials_types.h> /* definition of type sunrealtype          */

#include <time.h>

// Macro to access (x,y) location in 1D NVector array
#define IDX(x, y, n) ((n) * (y) + (x))

typedef struct {
  int cells[2];
  double xmax;
  double xmin;
  double ymax;
  double ymin;

  int nx;
  int ny;
  double diffD0;
  double* kx;
  double* ky;
  double kxmax;
  double kymax;
  double dx;
  double dy;
  double t0;
  double dtfixed;
  double ft;
  double rtol;
  double atol;
  int n_steps;
  sunbooleantype compute_error;
  sunbooleantype is_STS;

} UserData;

static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);
static int dom_eig(sunrealtype t, N_Vector y, N_Vector fn, sunrealtype* lambdaR,
                   sunrealtype* lambdaI, void* user_data, N_Vector temp1,
                   N_Vector temp2, N_Vector temp3);
static int check_flag(void* flagvalue, const char* funcname, int opt);
static void compute_error(N_Vector u, N_Vector v, UserData* data);
static void interpolate_initial_values(N_Vector* u, UserData* data);
static void compute_rhs(UserData* data, N_Vector rhs, N_Vector u);

/*-------------------------------
 * Private helper functions
 *-------------------------------*/

/* Check function return value...
    opt == 0 means SUNDIALS function allocates memory so check if
             returned NULL pointer
    opt == 1 means SUNDIALS function returns a flag so check if
             flag >= 0
    opt == 2 means function allocates memory so check if returned
             NULL pointer
*/
static int check_flag(void* flagvalue, const char* funcname, int opt)
{
  int* errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL)
  {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return 1;
  }

  /* Check if flag < 0 */
  else if (opt == 1)
  {
    errflag = (int*)flagvalue;
    if (*errflag < 0)
    {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
              funcname, *errflag);
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

static void interpolate_initial_values(N_Vector* u, UserData* data) {
  sunrealtype* uptr, vptr;
  uptr = N_VGetArrayPointer(*u);

  for (int j = 0; j < data->ny; j++) {
      for (int i = 0; i < data->nx; i++) {

        const sunrealtype x = data->xmin + i * data->dx;
        const sunrealtype y = data->ymin + j * data->dy;

        uptr[IDX(i, j, data->nx)] = (1.0 + 0.3*sin(2.0*x))/sqrt(5.5*M_PI)*exp(-(y*y)/5.5);
      }
  }
}

static void output_plot(N_Vector* u, UserData* data, int output_ID) {
  sunrealtype* uptr;
  uptr = N_VGetArrayPointer(*u);

  char fileName[100];
  sprintf(fileName, "output_%d.csv", output_ID);

  // Export the array to a CSV file
  FILE* fp = fopen(fileName, "w");
  if (fp == NULL) {
      printf("Error opening file for writing\n");
      return;
  }

  // Write header (optional)
  fprintf(fp, "x,y,value\n");

  // Write the data
  for (int j = 0; j < data->ny; j++) {
      for (int i = 0; i < data->nx; i++) {
          const sunrealtype x = data->xmin + i * data->dx;
          const sunrealtype y = data->ymin + j * data->dy;
          fprintf(fp, "%10.16f,%10.16f,%10.16f\n", x, y, uptr[IDX(i, j, data->nx)]);
      }
  }

  fclose(fp);
}

sunbooleantype first_RHS_call = SUNTRUE;
/* f routine to compute the ODE RHS function f(t,y). */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* data)
{
  UserData* user_data = (UserData*)data;

  if(first_RHS_call)
  first_RHS_call = SUNFALSE;

  compute_rhs(user_data, ydot, y);

  return 0; /* return with success */
}

/* dom_eig routine to estimate the dominated eigenvalue */
static int dom_eig(sunrealtype t, N_Vector y, N_Vector fn, sunrealtype* lambdaR,
    sunrealtype* lambdaI, void* user_data, N_Vector temp1,
    N_Vector temp2, N_Vector temp3)
{
  // Access problem data
  UserData* udata = (UserData*)user_data;

  if(first_RHS_call)
    f(t, y, fn, user_data);

  // Fill in spectral radius value
  *lambdaR = -SUN_RCONST(8.0) * fmax(udata->kxmax / udata->dx / udata->dx,
                                     udata->kymax / udata->dy / udata->dy);
  *lambdaI = SUN_RCONST(0.0);
  return 0; /* return with success */
}

int flag;

int STS_init(UserData* data, N_Vector* y, void** arkode_mem)
{
  /* Create the SUNDIALS context object for this simulation */
  SUNContext sunctx;
  flag = SUNContext_Create(SUN_COMM_NULL, &sunctx);
  if (check_flag(&flag, "SUNContext_Create", 1)) { return 1; }

  *y = N_VNew_Serial(data->nx * data->ny, sunctx);
  if (check_flag((void*)*y, "N_VMake_Serial", 0)) { return 1; }

  interpolate_initial_values(y, data);

  /* Call LSRKStepCreateSTS to initialize the ARK timestepper module and
     specify the right-hand side function in y'=f(t,y), the initial time
     T0, and the initial dependent variable vector y. */
  *arkode_mem = LSRKStepCreateSTS(f, data->t0, *y, sunctx);
  if (check_flag((void*)*arkode_mem, "LSRKStepCreateSTS", 0)) { return 1; }

  /* Set routines */
  flag = ARKodeSetUserData(*arkode_mem,
                           (void*)data); /* Pass the user data */
  if (check_flag(&flag, "ARKodeSetUserData", 1)) { return 1; }

  /* Specify tolerances */
  flag = ARKodeSStolerances(*arkode_mem, data->rtol, data->atol);
  if (check_flag(&flag, "ARKStepSStolerances", 1)) { return 1; }

  /* Specify user provided spectral radius */
  flag = LSRKStepSetDomEigFn(*arkode_mem, dom_eig);
  if (check_flag(&flag, "LSRKStepSetDomEigFn", 1)) { return 1; }

  /* Specify after how many successful steps dom_eig is recomputed
     Note that nsteps = 0 refers to constant dominant eigenvalue */
  flag = LSRKStepSetDomEigFrequency(*arkode_mem, 0);
  if (check_flag(&flag, "LSRKStepSetDomEigFrequency", 1)) { return 1; }

  /* Specify max number of stages allowed */
  flag = LSRKStepSetMaxNumStages(*arkode_mem, 200);
  if (check_flag(&flag, "LSRKStepSetMaxNumStages", 1)) { return 1; }

  /* Specify max number of steps allowed */
  flag = ARKodeSetMaxNumSteps(*arkode_mem, 100000);
  if (check_flag(&flag, "ARKodeSetMaxNumSteps", 1)) { return 1; }

  /* Specify safety factor for user provided dom_eig */
  flag = LSRKStepSetDomEigSafetyFactor(*arkode_mem, SUN_RCONST(1.01));
  if (check_flag(&flag, "LSRKStepSetDomEigSafetyFactor", 1)) { return 1; }

  /* Specify the Runge--Kutta--Legendre LSRK method */
  flag = LSRKStepSetSTSMethod(*arkode_mem, ARKODE_LSRK_RKL_2);
  if (check_flag(&flag, "LSRKStepSetSTSMethod", 1)) { return 1; }

  // /* Specify the fixed step size */
  // flag = ARKodeSetFixedStep(*arkode_mem, 1.0e-5);
  // if (check_flag(&flag, "ARKodeSetFixedStep", 1)) { return 1; }

  return 0;
}

int SSP_init(UserData* data, N_Vector* y, void** arkode_mem)
{
  /* Create the SUNDIALS context object for this simulation */
  SUNContext sunctx;
  flag = SUNContext_Create(SUN_COMM_NULL, &sunctx);
  if (check_flag(&flag, "SUNContext_Create", 1)) { return 1; }

  /* Initialize data structures */
  *y = N_VNew_Serial(data->nx * data->ny, sunctx);
  if (check_flag((void*)*y, "N_VMake_Serial", 0)) { return 1; }

  interpolate_initial_values(y, data);

  /* Call LSRKStepCreateSTS to initialize the ARK timestepper module and
     specify the right-hand side function in y'=f(t,y), the initial time
     T0, and the initial dependent variable vector y. */
  *arkode_mem = LSRKStepCreateSSP(f, data->t0, *y, sunctx);
  if (check_flag((void*)*arkode_mem, "LSRKStepCreateSTS", 0)) { return 1; }

  /* Set routines */
  flag = ARKodeSetUserData(*arkode_mem,
                           (void*)data); /* Pass the user data */
  if (check_flag(&flag, "ARKodeSetUserData", 1)) { return 1; }

  /* Specify tolerances */
  flag = ARKodeSStolerances(*arkode_mem, data->rtol, data->atol);
  if (check_flag(&flag, "ARKStepSStolerances", 1)) { return 1; }

  /* Specify max number of steps allowed */
  flag = ARKodeSetMaxNumSteps(*arkode_mem, 100000);
  if (check_flag(&flag, "ARKodeSetMaxNumSteps", 1)) { return 1; }

  /* Specify the Runge--Kutta--Legendre LSRK method */
  flag = LSRKStepSetSSPMethod(*arkode_mem, ARKODE_LSRK_SSP_S_3);
  if (check_flag(&flag, "LSRKStepSetSSPMethod", 1)) { return 1; }

  /* Specify the number of SSP stages */
  flag = LSRKStepSetNumSSPStages(*arkode_mem, 4);
  if (check_flag(&flag, "ARKodeSetOrder", 1)) { return 1; }

  // /* Specify the fixed step size */
  // flag = ARKodeSetFixedStep(*arkode_mem, 1.0e-5);
  // if (check_flag(&flag, "ARKodeSetFixedStep", 1)) { return 1; }

  return 0;
}

static int fd_diffusion_update_LSRK(UserData* data, void* arkode_mem,  double tout, N_Vector y, sunrealtype* tcurr)
{
  flag = ARKodeEvolve(arkode_mem, tout, y, tcurr, ARK_NORMAL); /* call integrator */
  if (check_flag(&flag, "ARKodeEvolve", 1)) {return 1; }

  return 0;
}

static void compute_rhs(UserData* data, N_Vector rhs, N_Vector u) {
  sunrealtype *uptr, *rhsptr;
  uptr = N_VGetArrayPointer(u);
  rhsptr = N_VGetArrayPointer(rhs);

  for (int j = 1; j < data->ny - 1; j++) {
    for (int i = 1; i < data->nx - 1; i++) {
      rhsptr[IDX(i, j, data->nx)] =
      (data->kx[i + 1] * uptr[IDX(i + 1, j, data->nx)] - 2.0 * data->kx[i] * uptr[IDX(i, j, data->nx)] + data->kx[i - 1] * uptr[IDX(i - 1, j, data->nx)]) / (data->dx * data->dx) +
      (data->ky[j + 1] * uptr[IDX(i, j + 1, data->nx)] - 2.0 * data->ky[j] * uptr[IDX(i, j, data->nx)] + data->ky[j - 1] * uptr[IDX(i, j - 1, data->nx)]) / (data->dy * data->dy);
    }
  }

  // i = 0 wall
  for (int j = 1; j < data->ny - 1; j++) {
    rhsptr[IDX(0, j, data->nx)] =
    (data->kx[data->nx - 1] * uptr[IDX(data->nx - 1 , j, data->nx)] - 2.0 * data->kx[0] * uptr[IDX(0, j, data->nx)] + data->kx[1] * uptr[IDX(1, j, data->nx)]) / (data->dx * data->dx) +
    (data->ky[j + 1] * uptr[IDX(0, j + 1, data->nx)] - 2.0 * data->ky[j] * uptr[IDX(0, j, data->nx)] + data->ky[j - 1] * uptr[IDX(0, j - 1, data->nx)]) / (data->dy * data->dy);
  }

  // i = nx - 1 wall
  for (int j = 1; j < data->ny - 1; j++) {
    rhsptr[IDX(data->nx - 1, j, data->nx)] =
    (data->kx[data->nx - 2] * uptr[IDX(data->nx - 2 , j, data->nx)] - 2.0 * data->kx[data->nx - 1] * uptr[IDX(data->nx - 1, j, data->nx)] + data->kx[0] * uptr[IDX(0, j, data->nx)]) / (data->dx * data->dx) +
    (data->ky[j + 1] * uptr[IDX(data->nx - 1, j + 1, data->nx)] - 2.0 * data->ky[j] * uptr[IDX(data->nx - 1, j, data->nx)] + data->ky[j - 1] * uptr[IDX(data->nx - 1, j - 1, data->nx)]) / (data->dy * data->dy);
  }

  // j = 0 wall
  for (int i = 1; i < data->nx - 1; i++) {
    rhsptr[IDX(i, 0, data->nx)] =
    (data->ky[data->ny - 1] * uptr[IDX(i, data->ny - 1, data->nx)] - 2.0 * data->ky[0] * uptr[IDX(i, 0, data->nx)] + data->ky[1] * uptr[IDX(i, 1, data->nx)]) / (data->dy * data->dy) +
    (data->kx[i - 1] * uptr[IDX(i - 1, 0, data->nx)] - 2.0 * data->kx[i] * uptr[IDX(i, 0, data->nx)] + data->kx[i + 1] * uptr[IDX(i + 1, 0, data->nx)]) / (data->dx * data->dx);
  }

  // j = ny - 1 wall
  for (int i = 1; i < data->nx - 1; i++) {
    rhsptr[IDX(i, data->ny - 1, data->nx)] =
    (data->ky[data->ny - 2] * uptr[IDX(i, data->ny - 2, data->nx)] - 2.0 * data->ky[data->ny - 1] * uptr[IDX(i, data->ny - 1, data->nx)] + data->ky[0] * uptr[IDX(i, 0, data->nx)]) / (data->dy * data->dy) +
    (data->kx[i - 1] * uptr[IDX(i - 1, data->ny - 1, data->nx)] - 2.0 * data->kx[i] * uptr[IDX(i, data->ny - 1, data->nx)] + data->kx[i + 1] * uptr[IDX(i + 1, data->ny - 1, data->nx)]) / (data->dx * data->dx);
  }

  // origin
  rhsptr[IDX(0, 0, data->nx)] =
  (data->ky[data->ny - 1] * uptr[IDX(0, data->ny - 1, data->nx)] - 2.0 * data->ky[0] * uptr[IDX(0, 0, data->nx)] + data->ky[1] * uptr[IDX(0, 1, data->nx)]) / (data->dy * data->dy) +
  (data->kx[data->nx - 1] * uptr[IDX(data->nx - 1, 0, data->nx)] - 2.0 * data->kx[0] * uptr[IDX(0, 0, data->nx)] + data->kx[1] * uptr[IDX(1, 0, data->nx)]) / (data->dx * data->dx);

  // lu corner
  rhsptr[IDX(0, data->ny - 1, data->nx)] =
  (data->ky[data->ny - 2] * uptr[IDX(0, data->ny - 2, data->nx)] - 2.0 * data->ky[data->ny - 1] * uptr[IDX(0, data->ny - 1, data->nx)] + data->ky[0] * uptr[IDX(0, 0, data->nx)]) / (data->dy * data->dy) +
  (data->kx[data->nx - 1] * uptr[IDX(data->nx - 1, data->ny - 1, data->nx)] - 2.0 * data->kx[0] * uptr[IDX(0, data->ny - 1, data->nx)] + data->kx[1] * uptr[IDX(1, data->ny - 1, data->nx)]) / (data->dx * data->dx);

  // rl corner
  rhsptr[IDX(data->nx - 1, 0, data->nx)] =
  (data->ky[data->ny - 1] * uptr[IDX(data->nx - 1, data->ny - 1, data->nx)] - 2.0 * data->ky[0] * uptr[IDX(data->nx - 1, 0, data->nx)] + data->ky[1] * uptr[IDX(data->nx - 1, 1, data->nx)]) / (data->dy * data->dy) +
  (data->kx[0] * uptr[IDX(0, 0, data->nx)] - 2.0 * data->kx[data->nx - 1] * uptr[IDX(data->nx - 1, 0, data->nx)] + data->kx[data->nx - 2] * uptr[IDX(data->nx - 2, 0, data->nx)]) / (data->dx * data->dx);

  // ru corner
  rhsptr[IDX(data->nx - 1, data->ny - 1, data->nx)] =
  (data->ky[data->ny - 2] * uptr[IDX(data->nx - 1, data->ny - 2, data->nx)] - 2.0 * data->ky[data->ny - 1] * uptr[IDX(data->nx - 1, data->ny - 1, data->nx)] + data->ky[0] * uptr[IDX(data->nx - 1, 0, data->nx)]) / (data->dy * data->dy) +
  (data->kx[data->nx - 2] * uptr[IDX(data->nx - 2, data->ny - 1, data->nx)] - 2.0 * data->kx[data->nx - 1] * uptr[IDX(data->nx - 1, data->ny - 1, data->nx)] + data->kx[0] * uptr[IDX(0, data->ny - 1, data->nx)]) / (data->dx * data->dx);
}

static void compute_error(N_Vector u, N_Vector v, UserData* data) {
    double error = 0;

    sunrealtype *uptr, *vptr;
    uptr = N_VGetArrayPointer(u);
    vptr = N_VGetArrayPointer(v);

    for (int j = 0; j < data->ny; j++) {
        for (int i = 0; i < data->nx; i++) {
            error = fmax(error, fabs(uptr[IDX(i, j, data->nx)] - vptr[IDX(i, j, data->nx)]));
        }
    }
    printf("\nerror = %e\n", error);
}

void diffusion_coefficients(UserData* data) {
  data->kxmax = 0.0;
  data->kymax = 0.0;

  for (int i = 0; i < data->nx; i++) {
    const sunrealtype x = data->xmin + i * data->dx;

    data->kx[i] = data->diffD0 * (1.0 + 0.99 * sin(x));
    data->kxmax = fmax(data->kxmax, fabs(data->kx[i]));
  }

  for (int j = 0; j < data->ny; j++) {
    const sunrealtype y = data->ymin + j * data->dy;

    data->ky[j] = 0.0;
    data->kymax = fmax(data->kymax, fabs(data->ky[j]));
  }
}

UserData create_userdata() {
  UserData data;
  data.cells[0] = 244;
  data.cells[1] = 60;
  data.xmax =  M_PI;
  data.xmin = -M_PI;
  data.ymax =  6.0;
  data.ymin = -6.0;
  data.nx = data.cells[0];
  data.ny = data.cells[1];
  data.dx = (data.xmax - data.xmin)/data.cells[0];
  data.dy = (data.ymax - data.ymin)/data.cells[1];

  data.t0 = 0.0;
  data.ft = 1.0;
  data.dtfixed = 1.0e-05;
  data.n_steps = 1000;

  data.rtol = 1.0e-05;
  data.atol = 1.0e-08;

  data.diffD0 = 10.0;

  data.compute_error = SUNFALSE;

  data.kx = (double*)malloc(data.nx * sizeof(double));
  data.ky = (double*)malloc(data.ny * sizeof(double));

  diffusion_coefficients(&data);

  printf("  cells = [%d, %d]\n", data.cells[0], data.cells[1]);
  printf("   xmin = %2.2f\n", data.xmin);
  printf("   xmax =  %2.2f\n", data.xmax);
  printf("   ymin = %2.2f\n", data.ymin);
  printf("   ymax =  %2.2f\n", data.ymax);
  printf("     nx =  %d\n", data.nx);
  printf("     ny =  %d\n", data.ny);
  printf("     dx = %2.5f\n", data.dx);
  printf("     dy = %2.5f\n", data.dy);
  printf("\n");
  printf("     t0 = %2.2f\n", data.t0);
  printf("     ft = %2.2f\n", data.ft);
  printf("dtfixed = %e\n", data.dtfixed);
  printf("n_steps = %d\n", data.n_steps);
  printf("   rtol = %e\n", data.rtol);
  printf("   atol = %e\n", data.atol);
  printf("\n");
  printf(" diffD0 = %2.2f\n", data.diffD0);
  printf("  kxmax = %2.2f\n", data.kxmax);
  printf("  kymax = %2.2f\n", data.kymax);

  printf("\n# of DoFs = %d\n", data.nx * data.ny);

  return data;
}

// Main function to run the simulation
int main() {

    UserData data = create_userdata();

    data.is_STS = SUNFALSE; // Either STS or SSP

    N_Vector yref    = NULL; /* empty vector for storing solution */

    clock_t begin = clock();
    // compute the reference solution and the error
    if(data.compute_error) {

      double dt = 0.0;
      double tout = 0;
      double t_curr = 0.0;
      int flag;

      data.t0 = 0.0;

      void* arkode_mem_ref = NULL;
      flag = SSP_init(&data, &yref, &arkode_mem_ref);
      if (check_flag(&flag, "SSP_init", 1)) { return 1; }

      /* Specify the fixed step size */
      flag = ARKodeSetFixedStep(arkode_mem_ref, data.dtfixed);
      if (check_flag(&flag, "ARKodeSetFixedStep", 1)) { return 1; }

      long int step = 1;
      while ((data.ft - t_curr > 1.0e-10) && (step <= data.n_steps)) {

        if(step == 1) {
        dt = 0.1;
        flag = ARKodeGetCurrentTime(arkode_mem_ref, &t_curr);
        if (check_flag(&flag, "ARKodeGetCurrentTime", 1)) { return 1; }
        }
        tout += dt;

        flag = fd_diffusion_update_LSRK(&data, arkode_mem_ref, tout, yref, &t_curr);

        if (!(flag == 0)) {
          fprintf(stdout, "** Update method failed! Aborting simulation ....\n");
          break;
        }
        step += 1;
      }
    }

    clock_t end = clock();
    double cpu_time_for_ref = (double)(end - begin) / CLOCKS_PER_SEC;

    begin = clock();
    void* arkode_mem = NULL; /* empty ARKode memory structure */
    N_Vector y       = NULL; /* empty vector for storing solution */

    if (data.is_STS) {
      flag = STS_init(&data, &y, &arkode_mem);
      if (check_flag(&flag, "STS_init", 1)) { return 1; }
    }
    else
    {
      flag = SSP_init(&data, &y, &arkode_mem);
      if (check_flag(&flag, "SSP_init", 1)) { return 1; }
    }

    output_plot(&y, &data, 0);

    double dt = 0.0;
    double tout = 0;
    double t_curr = 0.0;
    int flag;

    data.t0 = 0.0;

    long int step = 1;
    while ((data.ft - t_curr > 1.0e-10) && (step <= data.n_steps)) {
      fprintf(stdout, "Taking time-step %ld at t = %g ...\n", step, t_curr);

      if(step == 1) {
      dt = 0.1;
      flag = ARKodeGetCurrentTime(arkode_mem, &t_curr);
      if (check_flag(&flag, "ARKodeGetCurrentTime", 1)) { return 1; }
      }
      tout += dt;

      flag = fd_diffusion_update_LSRK(&data, arkode_mem, tout, y, &t_curr);

      if (!(flag == 0)) {
        fprintf(stdout, "** Update method failed! Aborting simulation ....\n");
        break;
      }
      output_plot(&y, &data, step);
      step += 1;
    }

    ARKodePrintAllStats(arkode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);

    end = clock();
    double cpu_time_used = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("\ncomputation time for the reference solution is %10.5f\n", cpu_time_for_ref);
    printf("\ncomputation time is %10.5f\n", cpu_time_used);

    if(data.compute_error) {
      compute_error(y, yref, &data);
      N_VLinearSum(1.0, y, -1.0, yref, y);
      // output_plot(&y, &data);
    }

    return 0;
}
