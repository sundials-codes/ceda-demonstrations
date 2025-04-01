#include <gkyl_array.h>
#include <gkyl_dynvec.h>
#include <gkyl_range.h>
#include <gkyl_rect_decomp.h>
#include <gkyl_rect_grid.h>
#include <gkyl_array_ops.h>
#include <gkyl_gk_geometry.h>
#include <gkyl_gk_geometry_mapc2p.h>
#include <gkyl_velocity_map.h>
#include <gkyl_proj_on_basis.h>
#include <gkyl_eval_on_nodes.h>
#include <gkyl_dg_updater_diffusion_gyrokinetic.h>
#include <gkyl_util.h>
#include <gkyl_null_comm.h>
#include <gkyl_comm_io.h>
#include <gkyl_dg_bin_ops.h>
#include <float.h>
#include <mpack.h>

#include "nvector_gkylzero.h"

#include <arkode/arkode_lsrkstep.h> /* prototypes for LSRKStep fcts., consts */
#include <arkode/arkode_erkstep.h>  /* prototypes for ERKStep fcts., consts */
#include <math.h>
#include <nvector/nvector_serial.h> /* serial N_Vector types, fcts., macros */
#include <stdio.h>
#include <sundials/sundials_math.h> /* def. of SUNRsqrt, etc. */
#include <sundials/sundials_types.h> /* definition of type sunrealtype          */

#include <rt_arg_parse.h>
#include <time.h>

void test_nvector_gkylzero (bool use_gpu) {

  int num_of_failures = 0;
  long int num_basis = 100;
  long int size = 100;
  double eq_check_tol = 1e-10;

  printf("\nTESTING nvector_gkylzero:\n");

  /* Create the SUNDIALS context object for this simulation */
  SUNContext sunctx;
  SUNContext_Create(SUN_COMM_NULL, &sunctx);

  struct gkyl_array *testarray;

  testarray = mkarr(use_gpu, num_basis, size);

  double *ta_data = testarray->data;

  for (unsigned int i=0; i<(testarray->size*testarray->ncomp); ++i) {
    ta_data[i] = (double)i;
  }

  N_Vector NV_test = N_VMake_Gkylzero(testarray, use_gpu, sunctx);

  struct gkyl_array *testarrayreturn;

  testarrayreturn = N_VGetVector_Gkylzero(NV_test);

  double *tar_data = testarrayreturn->data;

  bool failure = ((testarray->size != testarrayreturn->size) ||
                  (testarray->ncomp != testarrayreturn->ncomp));


  for (unsigned int i=0; i<(testarrayreturn->size*testarrayreturn->ncomp); ++i) {
    failure = (fabs(tar_data[i] - ta_data[i]) > eq_check_tol) || failure;
  }

  if(failure)
  {
    printf("\n      FAILED in N_VMake_Gkylzero or N_VGetVector_Gkylzero");
    num_of_failures++;
  }
  else
  {
    printf("\n      N_VMake_Gkylzero and N_VGetVector_Gkylzero PASSED the test");
  }

  /* ----------------------------------------------------------------------
  * N_VCloneEmpty_Gkylzero Test
  * --------------------------------------------------------------------*/

  N_Vector NV_test_clone = N_VClone_Gkylzero(NV_test);

  struct gkyl_array *testarrayclone;

  testarrayclone = N_VGetVector_Gkylzero(NV_test_clone);

  double *tac_data = testarrayclone->data;

  failure = ((testarray->size != testarrayclone->size) ||
                  (testarray->ncomp != testarrayclone->ncomp) ||
                  (tac_data == ta_data));

  if(failure)
  {
    printf("\n      FAILED in N_VCloneEmpty_Gkylzero or N_VClone_Gkylzero");
    num_of_failures++;
  }
  else
  {
    printf("\n      N_VCloneEmpty_Gkylzero and N_VClone_Gkylzero PASSED the test");
  }

  /* ----------------------------------------------------------------------
  * N_VScale_Gkylzero Test
  * --------------------------------------------------------------------*/

  N_Vector NV_test_return = N_VClone_Gkylzero(NV_test);

  N_VScale_Gkylzero(2.0, NV_test, NV_test_return);

  testarrayreturn = N_VGetVector_Gkylzero(NV_test_return);

  double *tas_data = testarrayreturn->data;

  failure = ((testarray->size != testarrayreturn->size) ||
                  (testarray->ncomp != testarrayreturn->ncomp));

  for (unsigned int i=0; i<(testarray->size*testarray->ncomp); ++i) {
    failure = (fabs(tas_data[i] - 2.0*ta_data[i]) > eq_check_tol) || failure;
  }

  if(failure)
  {
    printf("\n      FAILED in N_VScale_Gkylzero");
    num_of_failures++;
  }
  else
  {
    printf("\n      N_VScale_Gkylzero PASSED the test");
  }

  /* ----------------------------------------------------------------------
  * N_VConst_Gkylzero Test
  * --------------------------------------------------------------------*/

  N_VConst_Gkylzero(173.0, NV_test_return);

  testarrayreturn = N_VGetVector_Gkylzero(NV_test_return);

  double *tacon_data = testarrayreturn->data;

  failure = false;
  for (unsigned int i=0; i<(testarray->size*testarray->ncomp); ++i) {
    failure = (fabs(tacon_data[i] - 173.0) > eq_check_tol) || failure;
  }

  if(failure)
  {
    printf("\n            FAILED in N_VConst_Gkylzero");
    num_of_failures++;
  }
  else
  {
    printf("\n      N_VConst_Gkylzero PASSED the test");
  }

  /* ----------------------------------------------------------------------
  * N_VLinearSum_Gkylzero Test
  * --------------------------------------------------------------------*/

  double a = 2.0;
  double b = 3.0;
  double c = 1.75;
  double d = 2.89;

  struct gkyl_array* v1 = mkarr(use_gpu, num_basis, size);
  struct gkyl_array* v2 = mkarr(use_gpu, num_basis, size);
  struct gkyl_array *lin_sum = mkarr(use_gpu, num_basis, size);

  N_Vector Nv1      = N_VMake_Gkylzero(v1, use_gpu, sunctx);
  N_Vector Nv2      = N_VMake_Gkylzero(v2, use_gpu, sunctx);
  N_Vector Nlin_sum = N_VMake_Gkylzero(lin_sum, use_gpu, sunctx);

  N_VConst_Gkylzero(c, Nv1);
  N_VConst_Gkylzero(d, Nv2);

  N_VLinearSum_Gkylzero(a, Nv1, b, Nv2, Nlin_sum);

  lin_sum = N_VGetVector_Gkylzero(Nlin_sum);

  double *lin_sum_data = lin_sum->data;

  failure = false;
  for (unsigned int i=0; i<(testarrayreturn->size*testarrayreturn->ncomp); ++i) {
    failure = (fabs(lin_sum_data[i] - (a*c + b*d)) > eq_check_tol) || failure;
  }

  if(failure)
  {
    printf("\n      FAILED in N_VLinearSum_Gkylzero");
    num_of_failures++;
  }
  else
  {
    printf("\n      N_VLinearSum_Gkylzero PASSED the test");
  }

  /* ----------------------------------------------------------------------
  * N_VWrmsNorm_abs_comp_Gkylzero Test
  * --------------------------------------------------------------------*/

  N_VConst_Gkylzero(-0.5, Nv1);
  N_VConst_Gkylzero( 0.5, Nv2);

  double wrmsnorm = N_VWrmsNorm_abs_comp_Gkylzero(Nv1, Nv2);

  /* ans should equal 1/4 */
  failure = (wrmsnorm < 0.0) ? 1 : (fabs(wrmsnorm - 1.0/4.0) > eq_check_tol);

  if(failure)
  {
    printf("\n      FAILED in N_VWrmsNorm_abs_comp_Gkylzero");
    num_of_failures++;
  }
  else
  {
    printf("\n      N_VWrmsNorm_abs_comp_Gkylzero PASSED the test");
  }


  /* ----------------------------------------------------------------------
  * N_VDiv_Gkylzero Test
  * --------------------------------------------------------------------*/

  struct gkyl_array *nvdiv = mkarr(use_gpu, num_basis, size);

  N_Vector Nvdiv = N_VMake_Gkylzero(nvdiv, use_gpu, sunctx);

  N_VConst_Gkylzero(c, Nv1);
  N_VConst_Gkylzero(d, Nv2);

  N_VDiv_Gkylzero(Nv1, Nv2, Nvdiv);

  nvdiv = N_VGetVector_Gkylzero(Nvdiv);

  double *nvdiv_data = nvdiv->data;

  failure = false;
  for (unsigned int i=0; i<(nvdiv->size*nvdiv->ncomp); ++i) {
    failure = (fabs(nvdiv_data[i] - c/d) > eq_check_tol) || failure;
  }

  if(failure)
  {
    printf("\n      FAILED in N_VDiv_Gkylzero");
    num_of_failures++;
  }
  else
  {
    printf("\n      N_VDiv_Gkylzero PASSED the test");
  }

  /* ----------------------------------------------------------------------
  * N_VAbs_Gkylzero Test
  * --------------------------------------------------------------------*/

  struct gkyl_array *nvabs = mkarr(use_gpu, num_basis, size);

  N_Vector Nvabs = N_VMake_Gkylzero(nvabs, use_gpu, sunctx);

  N_VConst_Gkylzero(-1.0, Nv1);

  N_VAbs_Gkylzero(Nv1, Nvabs);

  nvabs = N_VGetVector_Gkylzero(Nvabs);

  double *nvabs_data = nvabs->data;

  failure = false;
  for (unsigned int i=0; i<(nvdiv->size*nvdiv->ncomp); ++i) {
    failure = (fabs(nvabs_data[i] - 1.0) > eq_check_tol) || failure;
  }

  if(failure)
  {
    printf("\n      FAILED in N_VAbs_Gkylzero");
    num_of_failures++;
  }
  else
  {
    printf("\n      N_VAbs_Gkylzero PASSED the test");
  }

  /* ----------------------------------------------------------------------
  * N_VInv_Gkylzero Test
  * --------------------------------------------------------------------*/

  struct gkyl_array *nvinv = mkarr(use_gpu, num_basis, size);

  N_Vector Nvinv = N_VMake_Gkylzero(nvinv, use_gpu, sunctx);

  N_VConst_Gkylzero(c, Nv1);

  N_VInv_Gkylzero(Nv1, Nvinv);

  nvinv = N_VGetVector_Gkylzero(Nvinv);

  double *nvinv_data = nvinv->data;

  failure = false;
  for (unsigned int i=0; i<(nvdiv->size*nvdiv->ncomp); ++i) {
    failure = (fabs(nvinv_data[i] - 1.0/c) > eq_check_tol) || failure;
  }

  if(failure)
  {
    printf("\n      FAILED in N_VInv_Gkylzero");
    num_of_failures++;
  }
  else
  {
    printf("\n      N_VInv_Gkylzero PASSED the test");
  }

  /* ----------------------------------------------------------------------
  * N_VMaxnorm_Gkylzero Test
  * --------------------------------------------------------------------*/

  N_VConst_Gkylzero(0.0, Nv1);

  sunrealtype nvmaxnorm = N_VMaxnorm_Gkylzero(Nv1);

  if (nvmaxnorm < 0.0 || nvmaxnorm >= eq_check_tol) { failure = 1; }

  if(failure)
  {
    printf("\n      FAILED in N_VMaxnorm_Gkylzero");
    num_of_failures++;
  }
  else
  {
    printf("\n      N_VMaxnorm_Gkylzero PASSED the test");
  }

  /* ----------------------------------------------------------------------
  * N_VAddconst_Gkylzero Test
  * --------------------------------------------------------------------*/

  struct gkyl_array *nvadd = mkarr(use_gpu, num_basis, size);

  N_Vector Nvadd = N_VMake_Gkylzero(nvadd, use_gpu, sunctx);

  N_VConst_Gkylzero(c, Nv1);

  N_VAddconst_Gkylzero(Nv1, d, Nvadd);

  nvadd = N_VGetVector_Gkylzero(Nvadd);

  double *nvadd_data = nvadd->data;

  failure = false;
  for (unsigned int i=0; i<(nvdiv->size*nvdiv->ncomp); ++i) {
    failure = (fabs(nvadd_data[i] - (c + d)) > eq_check_tol) || failure;
  }

  if(failure)
  {
    printf("\n      FAILED in N_VAddconst_Gkylzero");
    num_of_failures++;
  }
  else
  {
    printf("\n      N_VAddconst_Gkylzero PASSED the test");
  }

  if(num_of_failures == 0) {
    printf("\n\n nvector_gkylzero PASSED all tests!\n\n");
  }
  else {
    printf("\n\n nvector_gkylzero failed in %d test(s)!\n\n", num_of_failures);
  }

  N_VDestroy_Gkylzero(NV_test);
  N_VDestroy_Gkylzero(NV_test_clone);
  N_VDestroy_Gkylzero(NV_test_return);
  N_VDestroy_Gkylzero(Nv1);
  N_VDestroy_Gkylzero(Nv2);
  N_VDestroy_Gkylzero(Nlin_sum);
}

// Struct with context parameters.
struct diffusion_ctx {
  char name[128]; // Simulation name.
  int cdim, vdim; // Conf- and vel-space dimensions.

  double n0; // Density.
  double upar; // Parallel flow speed.
  double temp; // Temperature.
  double mass; // Species mass.
  double B0; // Magnetic field.
  double diffD0; // Diffusion amplitude.

  double x_min; // Minimum x of the grid.
  double x_max; // Maximum x of the grid.
  double vpar_min; // Minimum vpar of the grid.
  double vpar_max; // Maximum vpar of the grid.
  int cells[GKYL_MAX_DIM]; // Number of cells.
  int poly_order; // Polynomial order of the basis.

  double t_end; // Final simulation time.
  int num_frames; // Number of output frames.
  int int_diag_calc_num; // Number of integrated diagnostics computations (=INT_MAX for every step).
  double dt_failure_tol; // Minimum allowable fraction of initial time-step.
  int num_failures_max; // Maximum allowable number of consecutive small time-steps.
};

struct diffusion_ctx
create_diffusion_ctx(void)
{
  // Create the context with all the inputs for this simulation.
  struct diffusion_ctx ctx = {
    .name = "gk_diffusion_1x1v_p1", // App name.
    .cdim = 1, // Number of configuration space dimensions.
    .vdim = 1, // Number of velocity space dimensions.

    .n0 = 1.0, // Density.
    .upar = 0.0, // Parallel flow speed.
    .temp = 2.75, // Temperature.
    .mass = 1.0, // Species mass.
    .B0 = 1.0, // Magnetic field.
    .diffD0 = 0.1, // Diffusion amplitude.

    .x_min = -M_PI, // Minimum x of the grid.
    .x_max =  M_PI, // Maximum x of the grid.
    .vpar_min = -6.0, // Minimum vpar of the grid.
    .vpar_max =  6.0, // Maximum vpar of the grid.
    .poly_order = 1, // Polynomial order of the DG basis.
    .cells = {120, 20}, // Number of cells in each direction.

    .t_end = 1.0, // Final simulation time.
    .num_frames = 10, // Number of output frames.
    .int_diag_calc_num = 1000, // Number of times to compute integrated diagnostics.
    .dt_failure_tol = 1.0e-4, // Minimum allowable fraction of initial time-step.
    .num_failures_max = 20, // Maximum allowable number of consecutive small time-steps.
  };
  return ctx;
}

// Struct with inputs to our app.
struct gkyl_diffusion_app_inp {
  char name[128]; // Name of the app.

  int cdim, vdim; // Conf- and vel-space dimensions.
  double lower[GKYL_MAX_DIM], upper[GKYL_MAX_DIM]; // Grid extents.
  int cells[GKYL_MAX_DIM]; // Number of cells.
  int poly_order; // Polynomial order of the basis.
  bool use_gpu; // Whether to run on GPU.

  double cfl_frac; // Factor on RHS of the CFL constraint.

  // Mapping from computational to physical space.
  void (*mapc2p_func)(double t, const double *xn, double *fout, void *ctx);
  void *mapc2p_ctx; // Context.

  // Magnetic field amplitude.
  void (*bmag_func)(double t, const double *xn, double *fout, void *ctx);
  void *bmag_ctx; // Context.

  // Diffusion coefficient.
  void (*diffusion_coefficient_func)(double t, const double *xn, double *fout, void *ctx);
  void *diffusion_coefficient_ctx; // Context.

  // Initial condition.
  void (*initial_f_func)(double t, const double *xn, double *fout, void *ctx);
  void *initial_f_ctx; // Context.
};

void
mapc2p(double t, const double *xc, double* GKYL_RESTRICT xp, void *ctx)
{
  // Mapping from computational to physical space.
  xp[0] = xc[0]; xp[1] = xc[1]; xp[2] = xc[2];
}

void
bmag_1x(double t, const double *xn, double* restrict fout, void *ctx)
{
  // Magnetic field magnitude.
  double x = xn[0];

  struct diffusion_ctx *dctx = ctx;
  double B0 = dctx->B0;

  fout[0] = B0;
}

void
diffusion_coeff_1x(double t, const double *xn, double* restrict fout, void *ctx)
{
  // Diffusion coefficient profile.
  double x = xn[0];

  struct diffusion_ctx *dctx = ctx;
  double diffD0 = dctx->diffD0;

  fout[0] = diffD0 * (1.0 + 0.99 * sin(x));
}

void
init_distf_1x1v(double t, const double *xn, double* restrict fout, void *ctx)
{
  // Initial condition.
  double x = xn[0], vpar = xn[1];

  struct diffusion_ctx *dctx = ctx;
  double n0 = dctx->n0;
  double upar = dctx->upar;
  double vtsq = dctx->temp/dctx->mass;
  int vdim = dctx->vdim;
  double Lx = dctx->x_max - dctx->x_min;

  double den = n0*(1.0+0.3*sin(2*(2.0*M_PI/Lx)*x)); //change the initial condition to see if the error localizes somewhere else

  fout[0] = (den/pow(2.0*M_PI*vtsq,vdim/2.0)) * exp(-(pow(vpar-upar,2))/(2.0*vtsq));
}

// Time-stepping update status.
struct gkyl_update_status {
  bool success; // status of update
  double dt_actual; // actual time-step taken
  double dt_suggested; // suggested stable time-step
};

// Main struct containing all our objects.
struct gkyl_diffusion_app {
  char name[128]; // Name of the app.
  bool use_gpu; // Whether to run on the GPU.

  int cdim, vdim; // Conf- and vel-space dimensions.

  struct gkyl_rect_grid grid, grid_conf, grid_vel; // Phase-, conf- and vel-space grids.

  struct gkyl_basis basis, basis_conf; // Phase- and conf-space bases.

  struct gkyl_range local_conf, local_conf_ext; // Conf-space ranges.
  struct gkyl_range local_vel, local_vel_ext; // Vel-space ranges.
  struct gkyl_range local, local_ext; // Phase-space ranges.

  struct gkyl_comm *comm; // Communicator object.
  struct gkyl_rect_decomp *decomp; // Decomposition object.

  struct gkyl_array *bmag; // Magnetic field magnitude.
  struct gk_geometry *gk_geom; // Gyrokinetic geometry.

  struct gkyl_velocity_map *gvm; // Gyrokinetic velocity map.

  struct gkyl_array *f, *f1, *fnew; // Distribution functions (3 for ssp-rk3).
  struct gkyl_array *f_ho; // Host distribution functions for ICs and I/O.

  double cfl; // CFL factor (default: 1.0).
  struct gkyl_array *cflrate; // CFL rate in phase-space.
  double *omega_cfl; // Reduced CFL frequency.

  struct gkyl_array *diffD; // Diffusion coefficient field.
  struct gkyl_dg_updater_diffusion_gyrokinetic *diff_slvr; // Diffusion solver.

  int num_periodic_dir; // Number of periodic directions.
  int periodic_dirs[GKYL_MAX_DIM]; // List of periodic directions.

  struct gkyl_array *L2_f; // L2 norm f^2.
  double *red_L2_f; // For reduction of integrated L^2 norm on GPU.
  gkyl_dynvec integ_L2_f; // integrated L^2 norm reduced across grid.
  bool is_first_integ_L2_write_call; // Flag for integrated L^2 norm dynvec written first time.

  double tcurr; // Current simulation time.
};

static void
apply_bc(struct gkyl_diffusion_app* app, double tcurr, struct gkyl_array *distf)
{
  // Apply boundary conditions.
  int num_periodic_dir = app->num_periodic_dir, cdim = app->cdim;
  gkyl_comm_array_per_sync(app->comm, &app->local, &app->local_ext,
    num_periodic_dir, app->periodic_dirs, distf);
}

struct gkyl_diffusion_app *
gkyl_diffusion_app_new(struct gkyl_diffusion_app_inp *inp)
{
  // Create the diffusion app.
  struct gkyl_diffusion_app *app = gkyl_malloc(sizeof(struct gkyl_diffusion_app));

  strcpy(app->name, inp->name);

  app->cdim = inp->cdim;
  app->vdim = inp->vdim;
  app->use_gpu = inp->use_gpu;

  // Aliases for simplicity.
  int cdim = app->cdim,  vdim = app->vdim;
  bool use_gpu = app->use_gpu;

  double lower_conf[cdim], upper_conf[cdim];
  int cells_conf[cdim];
  for (int d=0; d<cdim; d++) {
    lower_conf[d] = inp->lower[d];
    upper_conf[d] = inp->upper[d];
    cells_conf[d] = inp->cells[d];
  }
  double lower_vel[vdim], upper_vel[vdim];
  int cells_vel[vdim];
  for (int d=0; d<vdim; d++) {
    lower_vel[d] = inp->lower[cdim+d];
    upper_vel[d] = inp->upper[cdim+d];
    cells_vel[d] = inp->cells[cdim+d];
  }

  // Grids.
  gkyl_rect_grid_init(&app->grid, cdim+vdim, inp->lower, inp->upper, inp->cells);
  gkyl_rect_grid_init(&app->grid_conf, cdim, lower_conf, upper_conf, cells_conf);
  gkyl_rect_grid_init(&app->grid_vel, vdim, lower_vel, upper_vel, cells_vel);

  // Basis functions.
  if (inp->poly_order == 1)
    gkyl_cart_modal_gkhybrid(&app->basis, cdim, vdim);
  else
    gkyl_cart_modal_serendip(&app->basis, cdim+vdim, inp->poly_order);
  gkyl_cart_modal_serendip(&app->basis_conf, cdim, inp->poly_order);

  // Ranges
  int ghost_conf[GKYL_MAX_CDIM]; // Number of ghost cells in conf-space.
  int ghost_vel[3] = {0}; // Number of ghost cells in vel-space.
  int ghost[GKYL_MAX_DIM] = {0}; // Number of ghost cells in phase-space.
  for (int d=0; d<cdim; d++) ghost_conf[d] = 1;
  for (int d=0; d<vdim; d++) ghost_vel[d] = 0;
  for (int d=0; d<cdim; d++) ghost[d] = ghost_conf[d];
  gkyl_create_grid_ranges(&app->grid_conf, ghost_conf, &app->local_conf_ext, &app->local_conf);
  gkyl_create_grid_ranges(&app->grid_vel, ghost_vel, &app->local_vel_ext, &app->local_vel);
  gkyl_create_grid_ranges(&app->grid, ghost, &app->local_ext, &app->local);

  // Communicator object.
  int cuts[GKYL_MAX_DIM];
  for (int d=0; d<cdim+vdim; d++) cuts[d] = 1;
  app->decomp = gkyl_rect_decomp_new_from_cuts(cdim+vdim, cuts, &app->local);
  app->comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .decomp = app->decomp,
      .use_gpu = app->use_gpu
    }
  );

  // Create bmag arrays.
  app->bmag = mkarr(use_gpu, app->basis_conf.num_basis, app->local_conf_ext.volume);
  struct gkyl_array *bmag_ho = use_gpu? mkarr(false, app->bmag->ncomp, app->bmag->size)
                                      : gkyl_array_acquire(app->bmag);
  gkyl_proj_on_basis *proj_bmag = gkyl_proj_on_basis_new(&app->grid_conf, &app->basis_conf,
    inp->poly_order+1, 1, inp->bmag_func, inp->bmag_ctx);
  gkyl_proj_on_basis_advance(proj_bmag, 0.0, &app->local_conf, bmag_ho);
  gkyl_array_copy(app->bmag, bmag_ho);
  gkyl_proj_on_basis_release(proj_bmag);
  gkyl_array_release(bmag_ho);

  // Initialize geometry
  struct gkyl_gk_geometry_inp geometry_input = {
    .geometry_id = GKYL_MAPC2P,
    .world = {0.0},  .mapc2p = inp->mapc2p_func,  .c2p_ctx = inp->mapc2p_ctx,
    .bmag_func = inp->bmag_func,  .bmag_ctx = inp->bmag_ctx,
    .basis = app->basis_conf,  .grid = app->grid_conf,
    .local = app->local_conf,  .local_ext = app->local_conf_ext,
    .global = app->local_conf, .global_ext = app->local_conf_ext,
  };
  int geo_ghost[3] = {1, 1, 1};
  geometry_input.geo_grid = gkyl_gk_geometry_augment_grid(app->grid_conf, geometry_input);
  gkyl_cart_modal_serendip(&geometry_input.geo_basis, 3, inp->poly_order);
  gkyl_create_grid_ranges(&geometry_input.geo_grid, geo_ghost, &geometry_input.geo_global_ext, &geometry_input.geo_global);
  memcpy(&geometry_input.geo_local, &geometry_input.geo_global, sizeof(struct gkyl_range));
  memcpy(&geometry_input.geo_local_ext, &geometry_input.geo_global_ext, sizeof(struct gkyl_range));
  // Deflate geometry.
  struct gk_geometry* gk_geom_3d = gkyl_gk_geometry_mapc2p_new(&geometry_input);
  app->gk_geom = gkyl_gk_geometry_deflate(gk_geom_3d, &geometry_input);
  gkyl_gk_geometry_release(gk_geom_3d);
  if (use_gpu) {
    // Copy geometry from host to device.
    struct gk_geometry* gk_geom_dev = gkyl_gk_geometry_new(app->gk_geom, &geometry_input, use_gpu);
    gkyl_gk_geometry_release(app->gk_geom);
    app->gk_geom = gkyl_gk_geometry_acquire(gk_geom_dev);
    gkyl_gk_geometry_release(gk_geom_dev);
  }

  // Velocity space mapping.
  struct gkyl_mapc2p_inp c2p_in = { };
  app->gvm = gkyl_velocity_map_new(c2p_in, app->grid, app->grid_vel,
    app->local, app->local_ext, app->local_vel, app->local_vel_ext, use_gpu);

  // Create distribution function arrays (3 for SSP-RK3).
  app->f = mkarr(use_gpu, app->basis.num_basis, app->local_ext.volume);
  app->f1 = mkarr(use_gpu, app->basis.num_basis, app->local_ext.volume);
  app->fnew = mkarr(use_gpu, app->basis.num_basis, app->local_ext.volume);
  app->f_ho = use_gpu? mkarr(false, app->f->ncomp, app->f->size)
                     : gkyl_array_acquire(app->f);
  gkyl_proj_on_basis *proj_distf = gkyl_proj_on_basis_new(&app->grid, &app->basis,
    inp->poly_order+1, 1, inp->initial_f_func, inp->initial_f_ctx);
  gkyl_proj_on_basis_advance(proj_distf, 0.0, &app->local, app->f_ho);
  gkyl_proj_on_basis_release(proj_distf);
  gkyl_array_copy(app->f, app->f_ho);

  // Things needed in ARKODE vector:
  //   1. cloning = mkarr & gkyl_array_copy
  //   2. g = a*f + b*f1 + c*fnew = wrap gkyl_array_accumulate(g, a, f)
  //   3. scale = gkyl_array_scale(f, a)
  //   4. dot/inner product: gkyl has an l2 norm operation
  //   5. weighted MRS norm
  //   6. set f = 1 (const) = gkyl_array_set(f, 1.0);

  app->cfl = inp->cfl_frac == 0? 1.0 : inp->cfl_frac; // CFL factor.

  // CFL frequency in phase-space.
  app->cflrate = mkarr(use_gpu, 1, app->local_ext.volume);

  if (use_gpu)
    app->omega_cfl = gkyl_cu_malloc(sizeof(double));
  else
    app->omega_cfl = gkyl_malloc(sizeof(double));

  // Create the diffusion coefficient array.
  // For now assume 2nd order diffusion in x only.
  int diffusion_order = 2;
  bool diff_dir[GKYL_MAX_CDIM] = {false};
  int num_diff_dir = 1; //number of diffusion directions
  diff_dir[0] = true; //direction of the diffusion
  bool is_zero_flux[2*GKYL_MAX_DIM] = {false}; // Whether to use zero-flux BCs.

  int szD = cdim * app->basis_conf.num_basis;
  app->diffD = mkarr(use_gpu, szD, app->local_conf_ext.volume);
  struct gkyl_array *diffD_ho = use_gpu? mkarr(false, app->diffD->ncomp, app->diffD->size)
                                       : gkyl_array_acquire(app->diffD);
  // Project the diffusion coefficient.
  gkyl_eval_on_nodes *proj_diffD = gkyl_eval_on_nodes_new(&app->grid_conf, &app->basis_conf,
    1, inp->diffusion_coefficient_func, inp->diffusion_coefficient_ctx);
  gkyl_eval_on_nodes_advance(proj_diffD, 0.0, &app->local_conf, diffD_ho);
  gkyl_eval_on_nodes_release(proj_diffD);
  gkyl_array_copy(app->diffD, diffD_ho);
  gkyl_array_release(diffD_ho);

  // Diffusion solver.
  app->diff_slvr = gkyl_dg_updater_diffusion_gyrokinetic_new(&app->grid,
      &app->basis, &app->basis_conf, false, diff_dir, diffusion_order, &app->local_conf, is_zero_flux, use_gpu);

  // Assume only periodic dir is x.
  app->num_periodic_dir = 1;
  app->periodic_dirs[0] = 0;

  // Things needed for L2 norm diagnostic.
  app->L2_f = mkarr(use_gpu, 1, app->local_ext.volume);
  if (use_gpu) {
    app->red_L2_f = gkyl_cu_malloc(sizeof(double));
  }
  app->integ_L2_f = gkyl_dynvec_new(GKYL_DOUBLE, 1); // Dynamic vector to store L2 norm in time.
  app->is_first_integ_L2_write_call = true;

  // Apply BC to the IC.
  apply_bc(app, 0.0, app->f);

  return app;
}

// Compute out = c1*arr1 + c2*arr2
static inline struct gkyl_array*
array_combine(struct gkyl_array *out, double c1, const struct gkyl_array *arr1,
  double c2, const struct gkyl_array *arr2, const struct gkyl_range *rng)
{
  return gkyl_array_accumulate_range(gkyl_array_set_range(out, c1, arr1, rng),
    c2, arr2, rng);
}

static void
forward_euler(struct gkyl_diffusion_app* app, double tcurr, double dt,
  const struct gkyl_array *fin, struct gkyl_array *fout,
  struct gkyl_update_status *st)
{
  // Take a forward Euler step with the suggested time-step dt. This may
  // not be the actual time-step taken. However, the function will never
  // take a time-step larger than dt even if it is allowed by
  // stability. The actual time-step and dt_suggested are returned in
  // the status object.
  double dtmin = DBL_MAX;

  while (true) {
    // Compute RHS of gyrokinetic equation and the minimum stable dt.
    gkyl_array_clear(app->cflrate, 0.0);
    gkyl_array_clear(fout, 0.0);

    gkyl_dg_updater_diffusion_gyrokinetic_advance(app->diff_slvr, &app->local,
      app->diffD, app->gk_geom->jacobgeo_inv, fin, app->cflrate, fout);

    gkyl_array_reduce_range(app->omega_cfl, app->cflrate, GKYL_MAX, &app->local);

    double omega_cfl_ho[1];
    if (app->use_gpu)
      gkyl_cu_memcpy(omega_cfl_ho, app->omega_cfl, sizeof(double), GKYL_CU_MEMCPY_D2H);
    else
      omega_cfl_ho[0] = app->omega_cfl[0];

    double dt1 = app->cfl/omega_cfl_ho[0];

    dtmin = fmin(dtmin, dt1);

    break;
  }

  double dt_max_rel_diff = 0.01;
  // Check if dtmin is slightly smaller than dt. Use dt if it is
  // (avoids retaking steps if dt changes are very small).
  double dt_rel_diff = (dt-dtmin)/dt;
  if (dt_rel_diff > 0 && dt_rel_diff < dt_max_rel_diff)
    dtmin = dt;

  // Compute minimum time-step across all processors.
  double dtmin_local = dtmin, dtmin_global;
  gkyl_comm_allreduce_host(app->comm, GKYL_DOUBLE, GKYL_MIN, 1, &dtmin_local, &dtmin_global);
  dtmin = dtmin_global;

  // Don't take a time-step larger that input dt.
  double dta = st->dt_actual = dt < dtmin ? dt : dtmin;
  st->dt_suggested = dtmin;

  // Complete update of distribution functions.
  gkyl_array_accumulate(gkyl_array_scale(fout, dta), 1.0, fin);

}


sunbooleantype first_RHS_call = SUNTRUE;
/* f routine to compute the ODE RHS function f(t,y). */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
  struct gkyl_diffusion_app *app = (struct gkyl_diffusion_app*) user_data;

  struct gkyl_array *fin = N_VGetVector_Gkylzero(y);
  struct gkyl_array *fout  = N_VGetVector_Gkylzero(ydot);

  if(first_RHS_call)
    first_RHS_call = SUNFALSE;

  gkyl_array_clear(app->cflrate, 0.0);
  gkyl_array_clear(fout, 0.0);

  apply_bc(app, t, fin); //apply_bc before computing the RHS

  gkyl_dg_updater_diffusion_gyrokinetic_advance(app->diff_slvr, &app->local,
    app->diffD, app->gk_geom->jacobgeo_inv, fin, app->cflrate, fout);

  return 0; /* return with success */
}

/* dom_eig routine to estimate the dominated eigenvalue */
static int dom_eig(sunrealtype t, N_Vector y, N_Vector fn, sunrealtype* lambdaR,
                   sunrealtype* lambdaI, void* user_data, N_Vector temp1,
                   N_Vector temp2, N_Vector temp3)
{
  if(first_RHS_call)
    f(t, y, fn, user_data);

  struct gkyl_diffusion_app *app = (struct gkyl_diffusion_app*) user_data;

  gkyl_array_reduce_range(app->omega_cfl, app->cflrate, GKYL_MAX, &app->local);

  double omega_cfl_ho[1];
  if (app->use_gpu)
    gkyl_cu_memcpy(omega_cfl_ho, app->omega_cfl, sizeof(double), GKYL_CU_MEMCPY_D2H);
  else
    omega_cfl_ho[0] = app->omega_cfl[0];

  *lambdaR           = -omega_cfl_ho[0];
  *lambdaI           = SUN_RCONST(0.0);
  return 0; /* return with success */
}


static int apply_bc_in_LSRK(sunrealtype t, N_Vector y, void* user_data) {

  struct gkyl_diffusion_app *app = (struct gkyl_diffusion_app*) user_data;
  struct gkyl_array* f = N_VGetVector_Gkylzero(y);

  apply_bc(app, t, f);

  return 0; /* return with success */
}

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

int flag;                /* reusable error-checking flag */

sunrealtype reltol = SUN_RCONST(1.0e-5); /* tolerances */
sunrealtype abstol = SUN_RCONST(1.0e-12);

// Error weight function for global norm of y_{n-1}
int efun_glob_norm(N_Vector x, N_Vector w, void *user_data) {

  sunrealtype xnorm;

  struct gkyl_array* xdptr = NV_CONTENT_GKZ(x)->dataptr;

  sunrealtype *x_data = xdptr->data;

  sunindextype N = (xdptr->size*xdptr->ncomp);
  xnorm = 0.0;

  for (sunindextype i=0; i<N; ++i) {
    xnorm += x_data[i] * x_data[i];
  }
  xnorm = reltol*SUNRsqrt(xnorm/N) + abstol;

  N_VConst(1.0/xnorm, w);

  return 0;
}

// Error weight function for cellwise norm of y_{n-1}
int efun_cell_norm(N_Vector x, N_Vector w, void *user_data) {

  sunrealtype xcnorm;

  struct gkyl_array* xdptr = NV_CONTENT_GKZ(x)->dataptr;
  struct gkyl_array* wdptr = NV_CONTENT_GKZ(w)->dataptr;

  sunrealtype *x_data = xdptr->data;
  sunrealtype *w_data = wdptr->data;

  for (sunindextype i=0; i<xdptr->size; ++i) {
    xcnorm = 0.0;
    for(sunindextype j=0; j<xdptr->ncomp; ++j) {
      xcnorm += x_data[i*xdptr->ncomp + j] * x_data[i*xdptr->ncomp + j];
    }
    xcnorm = reltol*SUNRsqrt(xcnorm/xdptr->ncomp) + abstol;
    for(sunindextype j=0; j<xdptr->ncomp; ++j) {
      w_data[i*xdptr->ncomp + j] = 1.0/xcnorm;
    }
  }

  return 0;
}

/* general problem parameters */
sunrealtype T0    = 0.0;  /* initial time */

int STS_init(struct gkyl_diffusion_app* app, N_Vector* y, void** arkode_mem)
{
  /* Create the SUNDIALS context object for this simulation */
  SUNContext sunctx;
  flag = SUNContext_Create(SUN_COMM_NULL, &sunctx);
  if (check_flag(&flag, "SUNContext_Create", 1)) { return 1; }

  /* Initialize data structures */
  if(*y == NULL) {
    *y = N_VMake_Gkylzero(app->f, app->use_gpu, sunctx);
    if (check_flag((void*)*y, "N_VMake_Gkylzero", 0)) { return 1; }
  }
  /* Call LSRKStepCreateSTS to initialize the ARK timestepper module and
     specify the right-hand side function in y'=f(t,y), the initial time
     T0, and the initial dependent variable vector y. */
  *arkode_mem = LSRKStepCreateSTS(f, T0, *y, sunctx);
  if (check_flag((void*)*arkode_mem, "LSRKStepCreateSTS", 0)) { return 1; }

  /* Set routines */
  flag = ARKodeSetUserData(*arkode_mem,
                           (void*)app); /* Pass the user data */
  if (check_flag(&flag, "ARKodeSetUserData", 1)) { return 1; }

  /* Specify tolerances */
  flag = ARKodeSStolerances(*arkode_mem, reltol, abstol);
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

int SSP_init(struct gkyl_diffusion_app* app, N_Vector* y, void** arkode_mem)
{
  /* Create the SUNDIALS context object for this simulation */
  SUNContext sunctx;
  flag = SUNContext_Create(SUN_COMM_NULL, &sunctx);
  if (check_flag(&flag, "SUNContext_Create", 1)) { return 1; }

  /* Initialize data structures */
  if(*y == NULL) {
    *y = N_VMake_Gkylzero(app->f, app->use_gpu, sunctx);
    if (check_flag((void*)*y, "N_VMake_Gkylzero", 0)) { return 1; }
  }
  /* Call LSRKStepCreateSTS to initialize the ARK timestepper module and
     specify the right-hand side function in y'=f(t,y), the initial time
     T0, and the initial dependent variable vector y. */
  *arkode_mem = LSRKStepCreateSSP(f, T0, *y, sunctx);
  if (check_flag((void*)*arkode_mem, "LSRKStepCreateSSP", 0)) { return 1; }

  /* Set routines */
  flag = ARKodeSetUserData(*arkode_mem,
                           (void*)app); /* Pass the user data */
  if (check_flag(&flag, "ARKodeSetUserData", 1)) { return 1; }

  /* Specify tolerances */
  flag = ARKodeSStolerances(*arkode_mem, reltol, abstol);
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

static struct gkyl_update_status
lsrk_step(struct gkyl_diffusion_app* app, void* arkode_mem, double tout, N_Vector y, sunrealtype* tcurr)
{
  // Take time-step using the STS methods. Also sets the status object
  // which has the actual and suggested dts used. These can be different
  // from the actual time-step.
  struct gkyl_update_status st = { .success = true };

  flag = ARKodeEvolve(arkode_mem, tout, y, tcurr, ARK_NORMAL); /* call integrator */
  if (check_flag(&flag, "ARKodeEvolve", 1)) {st.success = false; return st; }

  return st;
}

static struct gkyl_update_status
rk3(struct gkyl_diffusion_app* app, double dt0)
{
  // Take time-step using the RK3 method. Also sets the status object
  // which has the actual and suggested dts used. These can be different
  // from the actual time-step.
  const struct gkyl_array *fin;
  struct gkyl_array *fout;
  struct gkyl_update_status st = { .success = true };

  // time-stepper state
  enum { RK_STAGE_1, RK_STAGE_2, RK_STAGE_3, RK_COMPLETE } state = RK_STAGE_1;

  double tcurr = app->tcurr, dt = dt0;
  while (state != RK_COMPLETE) {
    switch (state) {
      case RK_STAGE_1:
        fin = app->f;
        fout = app->f1;

        forward_euler(app, tcurr, dt, fin, fout, &st);
        apply_bc(app, tcurr, fout);

        dt = st.dt_actual;
        state = RK_STAGE_2;
        break;

      case RK_STAGE_2:
        fin = app->f1;
        fout = app->fnew;

        forward_euler(app, tcurr+dt, dt, fin, fout, &st);

        if (st.dt_actual < dt) {
          dt = st.dt_actual;
          state = RK_STAGE_1; // restart from stage 1
        }
        else {
          array_combine(app->f1, 3.0/4.0, app->f, 1.0/4.0, app->fnew, &app->local_ext);

          fout = app->f1;
          apply_bc(app, tcurr, fout);

          state = RK_STAGE_3;
        }
        break;

      case RK_STAGE_3:
        fin = app->f1;
        fout = app->fnew;

        forward_euler(app, tcurr+dt/2, dt, fin, fout, &st);

        if (st.dt_actual < dt) {
          dt = st.dt_actual;
          state = RK_STAGE_1; // restart from stage 1
        }
        else {
          array_combine(app->f1, 1.0/3.0, app->f, 2.0/3.0, app->fnew, &app->local_ext);
          gkyl_array_copy_range(app->f, app->f1, &app->local_ext);

          fout = app->f;
          apply_bc(app, tcurr, fout);

          state = RK_COMPLETE;
        }
        break;

      case RK_COMPLETE: // can't happen: suppresses warning
        break;
    }
  }

  return st;
}

struct gkyl_update_status
gkyl_diffusion_update(struct gkyl_diffusion_app* app, double dt)
{
  // Update the state of the system by taking a single time step.
  struct gkyl_update_status status = rk3(app, dt);
  app->tcurr += status.dt_actual;

  // Check for any CUDA errors during time step
  if (app->use_gpu)
    checkCuda(cudaGetLastError());
  return status;
}

struct gkyl_update_status
gkyl_diffusion_update_STS(struct gkyl_diffusion_app* app, void* arkode_mem,  double tout, N_Vector y, sunrealtype* tcurr)
{
  // Update the state of the system by taking a single time step.

  struct gkyl_update_status status = lsrk_step(app, arkode_mem, tout, y, tcurr);

  app->tcurr += status.dt_actual;

  // Check for any CUDA errors during time step
  if (app->use_gpu)
    checkCuda(cudaGetLastError());
  return status;
}

void
gkyl_diffusion_app_calc_integrated_L2_f(struct gkyl_diffusion_app* app, double tm)
{
  // Calculate the L2 norm of f.
  gkyl_dg_calc_l2_range(app->basis, 0, app->L2_f, 0, app->f, app->local);
  gkyl_array_scale_range(app->L2_f, app->grid.cellVolume, &app->local);

  double L2[1] = { 0.0 };
  if (app->use_gpu) {
    gkyl_array_reduce_range(app->red_L2_f, app->L2_f, GKYL_SUM, &app->local);
    gkyl_cu_memcpy(L2, app->red_L2_f, sizeof(double), GKYL_CU_MEMCPY_D2H);
  }
  else {
    gkyl_array_reduce_range(L2, app->L2_f, GKYL_SUM, &app->local);
  }
  double L2_global[1] = { 0.0 };
  gkyl_comm_allreduce_host(app->comm, GKYL_DOUBLE, GKYL_SUM, 1, L2, L2_global);

  gkyl_dynvec_append(app->integ_L2_f, tm, L2_global);
}

void
gkyl_diffusion_app_write_integrated_L2_f(struct gkyl_diffusion_app* app)
{
  // Write the dynamic vector with the L2 norm of f.
  int rank;
  gkyl_comm_get_rank(app->comm, &rank);
  if (rank == 0) {
    // write out integrated L^2
    const char *fmt = "%s-f_%s.gkyl";
    int sz = gkyl_calc_strlen(fmt, app->name, "L2");
    char fileNm[sz+1]; // ensures no buffer overflow
    snprintf(fileNm, sizeof fileNm, fmt, app->name, "L2");

    if (app->is_first_integ_L2_write_call) {
      // Write to a new file (this ensure previous output is removed).
      gkyl_dynvec_write(app->integ_L2_f, fileNm);
      app->is_first_integ_L2_write_call = false;
    }
    else {
      // Append to existing file.
      gkyl_dynvec_awrite(app->integ_L2_f, fileNm);
    }
  }
  gkyl_dynvec_clear(app->integ_L2_f);
}

// Meta-data for IO.
struct diffusion_output_meta {
  int frame; // frame number
  double stime; // output time
  int poly_order; // polynomial order
  const char *basis_type; // name of basis functions
  char basis_type_nm[64]; // used during read
};

static struct gkyl_msgpack_data*
diffusion_array_meta_new(struct diffusion_output_meta meta)
{
  // Allocate new metadata to include in file.
  // Returned gkyl_msgpack_data must be freed using duffusion_array_meta_release.
  struct gkyl_msgpack_data *mt = gkyl_malloc(sizeof(*mt));

  mt->meta_sz = 0;
  mpack_writer_t writer;
  mpack_writer_init_growable(&writer, &mt->meta, &mt->meta_sz);

  // add some data to mpack
  mpack_build_map(&writer);

  mpack_write_cstr(&writer, "time");
  mpack_write_double(&writer, meta.stime);

  mpack_write_cstr(&writer, "frame");
  mpack_write_i64(&writer, meta.frame);

  mpack_write_cstr(&writer, "polyOrder");
  mpack_write_i64(&writer, meta.poly_order);

  mpack_write_cstr(&writer, "basisType");
  mpack_write_cstr(&writer, meta.basis_type);

  mpack_write_cstr(&writer, "Git_commit_hash");
  mpack_write_cstr(&writer, GIT_COMMIT_ID);

  mpack_complete_map(&writer);

  int status = mpack_writer_destroy(&writer);

  if (status != mpack_ok) {
    free(mt->meta); // we need to use free here as mpack does its own malloc
    gkyl_free(mt);
    mt = 0;
  }

  return mt;
}

static void
diffusion_array_meta_release(struct gkyl_msgpack_data *mt)
{
  // Release array meta data.
  if (!mt) return;
  MPACK_FREE(mt->meta);
  gkyl_free(mt);
}

void
gkyl_diffusion_app_write(struct gkyl_diffusion_app* app, double tm, int frame)
{
  // Write grid diagnostics for this app.
  struct gkyl_msgpack_data *mt = diffusion_array_meta_new( (struct diffusion_output_meta) {
      .frame = frame,
      .stime = tm,
      .poly_order = app->basis.poly_order,
      .basis_type = app->basis.id
    }
  );

  const char *fmt = "%s-f_%d.gkyl";
  int sz = gkyl_calc_strlen(fmt, app->name, frame);
  char fileNm[sz+1]; // ensures no buffer overflow
  snprintf(fileNm, sizeof fileNm, fmt, app->name, frame);

  // copy data from device to host before writing it out
  gkyl_array_copy(app->f_ho, app->f);

  gkyl_comm_array_write(app->comm, &app->grid, &app->local, mt, app->f_ho, fileNm);

  diffusion_array_meta_release(mt);
}

void
gkyl_diffusion_app_release(struct gkyl_diffusion_app *app)
{
  // Free memory associated with the app.
  gkyl_array_release(app->L2_f);
  gkyl_dynvec_release(app->integ_L2_f);
  if (app->use_gpu) {
    gkyl_cu_free(app->red_L2_f);
  }

  gkyl_dg_updater_diffusion_gyrokinetic_release(app->diff_slvr);
  gkyl_array_release(app->diffD);
  if (app->use_gpu) {
    gkyl_cu_free(app->omega_cfl);
  }
  else {
    gkyl_free(app->omega_cfl);
  }
  gkyl_array_release(app->cflrate);
  gkyl_array_release(app->f);
  gkyl_array_release(app->f1);
  gkyl_array_release(app->fnew);
  gkyl_array_release(app->f_ho);
  gkyl_array_release(app->bmag);
  gkyl_gk_geometry_release(app->gk_geom);
  gkyl_velocity_map_release(app->gvm);
  gkyl_comm_release(app->comm);
  gkyl_rect_decomp_release(app->decomp);
  gkyl_free(app);
}

void
calc_integrated_diagnostics(struct gkyl_tm_trigger* iot, struct gkyl_diffusion_app* app, double t_curr, bool force_calc)
{
  // Calculate diagnostics integrated over space.
  if (gkyl_tm_trigger_check_and_bump(iot, t_curr) || force_calc) {
    gkyl_diffusion_app_calc_integrated_L2_f(app, t_curr);
  }
}

void
write_data(struct gkyl_tm_trigger* iot, struct gkyl_diffusion_app* app, double t_curr, bool force_write)
{
  // Write grid and integrated diagnostics.
  bool trig_now = gkyl_tm_trigger_check_and_bump(iot, t_curr);
  if (trig_now || force_write) {
    int frame = (!trig_now) && force_write? iot->curr : iot->curr-1;

    gkyl_diffusion_app_write(app, t_curr, frame);

    gkyl_diffusion_app_calc_integrated_L2_f(app, t_curr);
    gkyl_diffusion_app_write_integrated_L2_f(app);
  }
}

double compute_max_error(N_Vector u, N_Vector v, struct gkyl_diffusion_app* app) {
  double error = 0;

  struct gkyl_array* udptr = NV_CONTENT_GKZ(u)->dataptr;
  struct gkyl_array* vdptr = NV_CONTENT_GKZ(v)->dataptr;

  sunrealtype *u_data = udptr->data;
  sunrealtype *v_data = vdptr->data;

  sunindextype N = (udptr->size*udptr->ncomp);

  for (int i = 0; i < N; i++) {
    error = fmax(error, fabs(u_data[i] - v_data[i]));
  }

  printf("\nerror = %e\n", error);

  return error;
}

int main(int argc, char **argv)
{
  bool is_STS = false;
  bool is_SSP = true;
  bool test_nvector = false;
  if(test_nvector)
    test_nvector_gkylzero(false);

  int wrms_norm_type = 1;

  bool compute_error = true;

  struct gkyl_app_args app_args = parse_app_args(argc, argv);

  // Create the context struct.
  struct diffusion_ctx ctx = create_diffusion_ctx();

  // Create the struct of app inputs.
  struct gkyl_diffusion_app_inp app_inp = {
    .cdim = ctx.cdim,  .vdim = ctx.vdim, // Conf- and vel-space basis.
    .lower = {ctx.x_min, ctx.vpar_min}, // Lower grid extents.
    .upper = {ctx.x_max, ctx.vpar_max}, // Upper grid extents.
    .cells = {ctx.cells[0], ctx.cells[1]}, // Number of cells.
    .poly_order = ctx.poly_order, // Polynomial order of DG basis.

    .cfl_frac = 0.5, // CFL factor.

    // Mapping from computational to physical space.
    .mapc2p_func = mapc2p,
    .mapc2p_ctx = &ctx,

    // Magnetic field amplitude.
    .bmag_func = bmag_1x,
    .bmag_ctx = &ctx,

    // Diffusion coefficient.
    .diffusion_coefficient_func = diffusion_coeff_1x,
    .diffusion_coefficient_ctx = &ctx,

    // Initial condition.
    .initial_f_func = init_distf_1x1v,
    .initial_f_ctx = &ctx,

    .use_gpu = app_args.use_gpu, // Whether to run on GPU.
  };
  strcpy(app_inp.name, ctx.name);

  // Create app object.
  struct gkyl_diffusion_app *app = gkyl_diffusion_app_new(&app_inp);

  // Initial and final simulation times.
  int frame_curr = 0;
  double t_curr = 0.0, t_end = ctx.t_end;

  // Create triggers for IO.
  int num_frames = ctx.num_frames, num_int_diag_calc = ctx.int_diag_calc_num;
  struct gkyl_tm_trigger trig_write = { .dt = t_end/num_frames, .tcurr = t_curr, .curr = frame_curr };
  struct gkyl_tm_trigger trig_calc_intdiag = { .dt = t_end/GKYL_MAX2(num_frames, num_int_diag_calc),
    .tcurr = t_curr, .curr = frame_curr };

  // Write out ICs (if restart, it overwrites the restart frame).
  calc_integrated_diagnostics(&trig_calc_intdiag, app, t_curr, false);
  write_data(&trig_write, app, t_curr, false);

  double dt = t_end-t_curr; // Initial time step.
  // Initialize small time-step check.
  double dt_init = -1.0, dt_failure_tol = ctx.dt_failure_tol;
  int num_failures = 0, num_failures_max = ctx.num_failures_max;

  void* arkode_mem = NULL; /* empty ARKode memory structure */
  N_Vector y       = NULL; /* empty vector for storing solution */
  N_Vector yref    = NULL; /* empty vector for storing solution */

  struct gkyl_array *fref = mkarr(false, app->basis.num_basis, app->local_ext.volume);
  gkyl_array_set(fref, 1.0, app->f);

  // compute the reference solution and the error
  SUNContext sunctx;
  flag = SUNContext_Create(SUN_COMM_NULL, &sunctx);
  if (check_flag(&flag, "SUNContext_Create", 1)) { return 1; }

  yref = N_VMake_Gkylzero(fref, app->use_gpu, sunctx);

  int flag;

  /* Create the reference solution memory*/
  void* arkode_mem_ref = NULL;
  flag = STS_init(app, &yref, &arkode_mem_ref);
  if (check_flag(&flag, "SSP_init", 1)) { return 1; }

  /* Specify the fixed step size for the reference STS solution */
  flag = ARKodeSetFixedStep(arkode_mem_ref, 1.0e-5);
  if (check_flag(&flag, "ARKodeSetFixedStep", 1)) { return 1; }

  if (is_STS) {
    int flag;
    flag = STS_init(app, &y, &arkode_mem);
    if (check_flag(&flag, "STS_init", 1)) { return 1; }
  }
  else if (is_SSP)
  {
    int flag;
    flag = SSP_init(app, &y, &arkode_mem);
    if (check_flag(&flag, "SSP_init", 1)) { return 1; }
  }

  /* Specify the Ewt function */
  switch (wrms_norm_type) {
    case 1:
      y->ops->nvwrmsnorm          = N_VWrmsNorm_abs_comp_Gkylzero;
      printf("\nUsing WRMSNorm with componentwise absolute values\n");
      break;

    case 2:
      flag = ARKodeWFtolerances(arkode_mem, efun_cell_norm);
      if (check_flag(&flag, "ARKodeWFtolerances", 1)) { return 1; }
      y->ops->nvwrmsnorm          = N_VWrmsNorm_cell_norm_Gkylzero;
      printf("\nUsing WRMSNorm with cellwise norm values\n");
      break;

    case 3:
      flag = ARKodeWFtolerances(arkode_mem, efun_glob_norm);
      if (check_flag(&flag, "ARKodeWFtolerances", 1)) { return 1; }
      y->ops->nvwrmsnorm          = N_VWrmsNorm_glob_norm_Gkylzero;
      printf("\nUsing WRMSNorm with global norm values\n");
      break;
  }

  printf("\nNumber of cells             = %ld", app->f->size);
  printf("\nNumber of DoFs in each cell = %ld", app->f->ncomp);
  printf("\nNumber of DoFs              = %ld\n", app->f->size*app->f->ncomp);

  double tout = 0;
  double max_error = 0.0;

  long step = 1;
  while ((t_end - t_curr > 1.0e-10) && (step <= app_args.num_steps)) {
    fprintf(stdout, "Taking time-step %ld at t = %g ...\n", step, t_curr);

    struct gkyl_update_status status;

    if(is_STS || is_SSP) {
      if(step == 1) {
        dt = 0.1;
        flag = ARKodeGetCurrentTime(arkode_mem_ref, &t_curr);
        if (check_flag(&flag, "ARKodeGetCurrentTime", 1)) { return 1; }
        flag = ARKodeGetCurrentTime(arkode_mem, &t_curr);
        if (check_flag(&flag, "ARKodeGetCurrentTime", 1)) { return 1; }
      }
      tout += dt;

               gkyl_diffusion_update_STS(app, arkode_mem_ref, tout, yref, &t_curr);
      status = gkyl_diffusion_update_STS(app, arkode_mem, tout, y, &t_curr);

      max_error = fmax(compute_max_error(y, yref, app), max_error);
    }
    else {
      status = gkyl_diffusion_update(app, dt);
      fprintf(stdout, " dt = %g\n", status.dt_actual);
    }

    if (!status.success) {
      fprintf(stdout, "** Update method failed! Aborting simulation ....\n");
      break;
    }

    if(!(is_STS || is_SSP)){
      t_curr += status.dt_actual;
      dt = status.dt_suggested;
    }

    calc_integrated_diagnostics(&trig_calc_intdiag, app, t_curr, t_curr > t_end);
    write_data(&trig_write, app, t_curr, t_curr > t_end);

    if(!(is_STS || is_SSP)) {
      if (dt_init < 0.0) {
        dt_init = status.dt_actual;
      }
      else if (status.dt_actual < dt_failure_tol * dt_init) {
        num_failures += 1;

        fprintf(stdout, "WARNING: Time-step dt = %g", status.dt_actual);
        fprintf(stdout, " is below %g*dt_init ...", dt_failure_tol);
        fprintf(stdout, " num_failures = %d\n", num_failures);
        if (num_failures >= num_failures_max) {
          fprintf(stdout, "ERROR: Time-step was below %g*dt_init ", dt_failure_tol);
          fprintf(stdout, "%d consecutive times. Aborting simulation ....\n", num_failures_max);
          break;
        }
      }
      else {
        num_failures = 0;
      }
    }

    step += 1;
  }

  if(is_STS || is_SSP) {
    printf("\nReference Solution Stats\n");
    ARKodePrintAllStats(arkode_mem_ref, stdout, SUN_OUTPUTFORMAT_TABLE);
    printf("\nComputed Solution Stats\n");
    ARKodePrintAllStats(arkode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
  }

  compute_max_error(y, yref, app);

  printf("\nmax error = %e\n", max_error);

  // Free the app.
  gkyl_diffusion_app_release(app);

  return 0;
}
