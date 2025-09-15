#include <float.h>
#include <gkyl_array.h>
#include <gkyl_array_ops.h>
#include <gkyl_comm_io.h>
#include <gkyl_dg_bin_ops.h>
#include <gkyl_dg_updater_diffusion_gyrokinetic.h>
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
struct diffusion_ctx
{
  char name[128]; // Simulation name.
  int cdim, vdim; // Conf- and vel-space dimensions.

  double n0;     // Density.
  double upar;   // Parallel flow speed.
  double temp;   // Temperature.
  double mass;   // Species mass.
  double B0;     // Magnetic field.
  double diffD0; // Diffusion amplitude.

  double x_min;            // Minimum x of the grid.
  double x_max;            // Maximum x of the grid.
  double vpar_min;         // Minimum vpar of the grid.
  double vpar_max;         // Maximum vpar of the grid.
  int cells[GKYL_MAX_DIM]; // Number of cells.
  int poly_order;          // Polynomial order of the basis.

  double t_end;   // Final simulation time.
  int num_frames; // Number of output frames.
  int int_diag_calc_num; // Number of integrated diagnostics computations (=INT_MAX for every step).
  double dt_failure_tol; // Minimum allowable fraction of initial time-step.
  int num_failures_max; // Maximum allowable number of consecutive small time-steps.
};

struct diffusion_ctx create_diffusion_ctx(void)
{
  // Create the context with all the inputs for this simulation.
  struct diffusion_ctx ctx = {
    .name = "gk_diffusion_1x1v_p1", // App name.
    .cdim = 1,                      // Number of configuration space dimensions.
    .vdim = 1,                      // Number of velocity space dimensions.

    .n0     = 1.0,  // Density.
    .upar   = 0.0,  // Parallel flow speed.
    .temp   = 2.75, // Temperature.
    .mass   = 1.0,  // Species mass.
    .B0     = 1.0,  // Magnetic field.
    .diffD0 = 0.1,  // Diffusion amplitude.

    .x_min      = -PI,     // Minimum x of the grid.
    .x_max      = PI,      // Maximum x of the grid.
    .vpar_min   = -6.0,      // Minimum vpar of the grid.
    .vpar_max   = 6.0,       // Maximum vpar of the grid.
    .poly_order = 1,         // Polynomial order of the DG basis.
    .cells      = {120, 20}, // Number of cells in each direction.

    .t_end      = 1.0,         // Final simulation time.
    .num_frames = 10,          // Number of output frames.
    .int_diag_calc_num = 1000, // Number of times to compute integrated diagnostics.
    .dt_failure_tol = 1.0e-4, // Minimum allowable fraction of initial time-step.
    .num_failures_max = 20, // Maximum allowable number of consecutive small time-steps.
  };
  return ctx;
}

// Struct with inputs to our app.
struct gkyl_diffusion_app_inp
{
  char name[128]; // Name of the app.

  int cdim, vdim; // Conf- and vel-space dimensions.
  double lower[GKYL_MAX_DIM], upper[GKYL_MAX_DIM]; // Grid extents.
  int cells[GKYL_MAX_DIM];                         // Number of cells.
  int poly_order; // Polynomial order of the basis.

  bool use_gpu;   // Whether to run on GPU.
  int cuts[3]; // Number of subdomain in each dimension.
  struct gkyl_comm *comm; // Communicator to use.

  double cfl_frac; // Factor on RHS of the CFL constraint.

  // Mapping from computational to physical space.
  void (*mapc2p_func)(double t, const double* xn, double* fout, void* ctx);
  void* mapc2p_ctx; // Context.

  // Magnetic field amplitude.
  void (*bmag_func)(double t, const double* xn, double* fout, void* ctx);
  void* bmag_ctx; // Context.

  // Diffusion coefficient.
  void (*diffusion_coefficient_func)(double t, const double* xn, double* fout,
                                     void* ctx);
  void* diffusion_coefficient_ctx; // Context.

  // Initial condition.
  void (*initial_f_func)(double t, const double* xn, double* fout, void* ctx);
  void* initial_f_ctx; // Context.
};

void mapc2p(double t, const double* xc, double* GKYL_RESTRICT xp, void* ctx)
{
  // Mapping from computational to physical space.
  xp[0] = xc[0];
  xp[1] = xc[1];
  xp[2] = xc[2];
}

void bmag_1x(double t, const double* xn, double* restrict fout, void* ctx)
{
  // Magnetic field magnitude.
  double x = xn[0];

  struct diffusion_ctx* dctx = ctx;
  double B0                  = dctx->B0;

  fout[0] = B0;
}

void diffusion_coeff_1x(double t, const double* xn, double* restrict fout,
                        void* ctx)
{
  // Diffusion coefficient profile.
  double x = xn[0];

  struct diffusion_ctx* dctx = ctx;
  double diffD0              = dctx->diffD0;

  fout[0] = diffD0 * (1.0 + 0.99 * sin(x));
}

void init_distf_1x1v(double t, const double* xn, double* restrict fout, void* ctx)
{
  // Initial condition.
  double x = xn[0], vpar = xn[1];

  struct diffusion_ctx* dctx = ctx;
  double n0                  = dctx->n0;
  double upar                = dctx->upar;
  double vtsq                = dctx->temp / dctx->mass;
  int vdim                   = dctx->vdim;
  double Lx                  = dctx->x_max - dctx->x_min;

  // change the initial condition to see if the error localizes somewhere else
  double den = n0 * (1.0 + 0.3 * sin(2 * (2.0 * PI / Lx) * x));

  fout[0] = (den / pow(2.0 * PI * vtsq, vdim / 2.0)) *
            exp(-(pow(vpar - upar, 2)) / (2.0 * vtsq));
}

// Main struct containing all our objects.
struct gkyl_diffusion_app
{
  char name[128]; // Name of the app.
  bool use_gpu;   // Whether to run on the GPU.

  int cdim, vdim; // Conf- and vel-space dimensions.

  struct gkyl_rect_grid grid, grid_conf, grid_vel; // Phase-, conf- and vel-space grids.

  struct gkyl_basis basis, basis_conf; // Phase- and conf-space bases.

  struct gkyl_range global_conf, global_ext_conf; // Local conf-space ranges.
  struct gkyl_range global_vel, global_ext_vel;   // Local vel-space ranges.
  struct gkyl_range global, global_ext;           // Local phase-space ranges.

  struct gkyl_range local_conf, local_ext_conf; // Local conf-space ranges.
  struct gkyl_range local_vel, local_ext_vel;   // Local vel-space ranges.
  struct gkyl_range local, local_ext;           // Local phase-space ranges.

  struct gkyl_comm *comm; // Phase-space communicator.
  struct gkyl_comm *comm_conf; // Conf-space communicator.
  struct gkyl_rect_decomp* decomp; // Phase-space decomposition object.
  struct gkyl_rect_decomp* decomp_conf; // Conf-space decomposition object.

  struct gkyl_array* bmag;     // Magnetic field magnitude.
  struct gk_geometry* gk_geom; // Gyrokinetic geometry.

  struct gkyl_velocity_map* gvm; // Gyrokinetic velocity map.

  struct gkyl_array *f, *f1, *fnew; // Distribution functions (3 for ssp-rk3).
  struct gkyl_array* f_ho; // Host distribution functions for ICs and I/O.

  double cfl;                 // CFL factor (default: 1.0).
  struct gkyl_array* cflrate; // CFL rate in phase-space.
  double* omega_cfl;          // Reduced CFL frequency.

  struct gkyl_array* diffD; // Diffusion coefficient field.
  struct gkyl_dg_updater_diffusion_gyrokinetic* diff_slvr; // Diffusion solver.

  int num_periodic_dir;            // Number of periodic directions.
  int periodic_dirs[GKYL_MAX_DIM]; // List of periodic directions.

  struct gkyl_array* L2_f; // L2 norm f^2.
  double* red_L2_f;        // For reduction of integrated L^2 norm on GPU.
  gkyl_dynvec integ_L2_f;  // integrated L^2 norm reduced across grid.
  bool is_first_integ_L2_write_call; // Flag for integrated L^2 norm dynvec written first time.

  double tcurr; // Current simulation time.
};

static void apply_bc(struct gkyl_diffusion_app* app, double tcurr,
                     struct gkyl_array* distf)
{
  // Apply boundary conditions.
  int num_periodic_dir = app->num_periodic_dir, cdim = app->cdim;
  gkyl_comm_array_per_sync(app->comm, &app->local, &app->local_ext,
                           num_periodic_dir, app->periodic_dirs, distf);
}

struct gkyl_diffusion_app* gkyl_diffusion_app_new(struct gkyl_diffusion_app_inp* inp)
{
  // Create the diffusion app.
  struct gkyl_diffusion_app* app = gkyl_malloc(sizeof(struct gkyl_diffusion_app));

  strcpy(app->name, inp->name);

  app->cdim    = inp->cdim;
  app->vdim    = inp->vdim;
  app->use_gpu = inp->use_gpu;

  // Aliases for simplicity.
  int cdim = app->cdim, vdim = app->vdim;
  bool use_gpu = app->use_gpu;

  double lower_conf[cdim], upper_conf[cdim];
  int cells_conf[cdim];
  for (int d = 0; d < cdim; d++)
  {
    lower_conf[d] = inp->lower[d];
    upper_conf[d] = inp->upper[d];
    cells_conf[d] = inp->cells[d];
  }
  double lower_vel[vdim], upper_vel[vdim];
  int cells_vel[vdim];
  for (int d = 0; d < vdim; d++)
  {
    lower_vel[d] = inp->lower[cdim + d];
    upper_vel[d] = inp->upper[cdim + d];
    cells_vel[d] = inp->cells[cdim + d];
  }

  // Conf-space grid.
  gkyl_rect_grid_init(&app->grid_conf, cdim, lower_conf, upper_conf, cells_conf);

  // Conf-space basis.
  gkyl_cart_modal_serendip(&app->basis_conf, cdim, inp->poly_order);

  // Global conf-space range.
  int ghost_conf[GKYL_MAX_CDIM]; // Number of ghost cells in conf-space.
  for (int d = 0; d < cdim; d++) ghost_conf[d] = 1;
  gkyl_create_grid_ranges(&app->grid_conf, ghost_conf, &app->global_ext_conf, &app->global_conf);

  // Conf-space communicator object.
  int cuts_conf[GKYL_MAX_DIM];
  for (int d = 0; d < cdim; d++) cuts_conf[d] = inp->cuts[d];
  app->decomp_conf = gkyl_rect_decomp_new_from_cuts(cdim, cuts_conf, &app->global_conf);
  app->comm_conf = gkyl_comm_split_comm(inp->comm, 0, app->decomp_conf);

  // Local conf-space range.
  int rank;
  gkyl_comm_get_rank(app->comm_conf, &rank);
  gkyl_create_ranges(&app->decomp_conf->ranges[rank], ghost_conf, &app->local_ext_conf, &app->local_conf);

  // Phase-space grid.
  gkyl_rect_grid_init(&app->grid_vel, vdim, lower_vel, upper_vel, cells_vel);
  gkyl_rect_grid_init(&app->grid, cdim+vdim, inp->lower, inp->upper, inp->cells);

  // Basis functions.
  if (inp->poly_order == 1)
    gkyl_cart_modal_gkhybrid(&app->basis, cdim, vdim);
  else
    gkyl_cart_modal_serendip(&app->basis, cdim + vdim, inp->poly_order);

  // Global phase-space range.
  int ghost_vel[3]        = {0}; // Number of ghost cells in vel-space.
  int ghost[GKYL_MAX_DIM] = {0}; // Number of ghost cells in phase-space.
  for (int d = 0; d < cdim; d++) ghost[d] = ghost_conf[d];
  gkyl_create_grid_ranges(&app->grid_vel, ghost_vel, &app->global_ext_vel, &app->global_vel);
  gkyl_create_grid_ranges(&app->grid, ghost, &app->global_ext, &app->global);

  // Phase-space communicator object.
  app->comm = gkyl_comm_extend_comm(app->comm_conf, &app->local_vel);

  // Create local and local_ext.
  struct gkyl_range local;
  // Local = conf-local X local_vel.
  gkyl_range_ten_prod(&local, &app->local_conf, &app->local_vel);
  gkyl_create_ranges(&local, ghost, &app->local_ext, &app->local);

  // Create bmag arrays.
  app->bmag = mkarr(use_gpu, app->basis_conf.num_basis, app->local_ext_conf.volume);
  struct gkyl_array* bmag_ho = use_gpu? mkarr(false, app->bmag->ncomp, app->bmag->size)
                                      : gkyl_array_acquire(app->bmag);
  gkyl_proj_on_basis* proj_bmag =
    gkyl_proj_on_basis_new(&app->grid_conf, &app->basis_conf,
                           inp->poly_order + 1, 1, inp->bmag_func, inp->bmag_ctx);
  gkyl_proj_on_basis_advance(proj_bmag, 0.0, &app->local_conf, bmag_ho);
  gkyl_array_copy(app->bmag, bmag_ho);
  gkyl_proj_on_basis_release(proj_bmag);
  gkyl_array_release(bmag_ho);

  // Initialize geometry
  struct gkyl_gk_geometry_inp geometry_input = {
    .geometry_id = GKYL_MAPC2P,
    .world       = {0.0},
    .mapc2p      = inp->mapc2p_func,
    .c2p_ctx     = inp->mapc2p_ctx,
    .bmag_func   = inp->bmag_func,
    .bmag_ctx    = inp->bmag_ctx,
    .basis       = app->basis_conf,
    .grid        = app->grid_conf,
    .local       = app->local_conf,
    .local_ext   = app->local_ext_conf,
    .global      = app->global_conf,
    .global_ext  = app->global_ext_conf,
  };
  int geo_ghost[3]        = {1, 1, 1};
  geometry_input.geo_grid = gkyl_gk_geometry_augment_grid(app->grid_conf,
                                                          geometry_input);
  gkyl_cart_modal_serendip(&geometry_input.geo_basis, 3, inp->poly_order);
  gkyl_create_grid_ranges(&geometry_input.geo_grid, geo_ghost,
                          &geometry_input.geo_global_ext,
                          &geometry_input.geo_global);
  memcpy(&geometry_input.geo_local, &geometry_input.geo_global,
         sizeof(struct gkyl_range));
  memcpy(&geometry_input.geo_local_ext, &geometry_input.geo_global_ext,
         sizeof(struct gkyl_range));
  // Deflate geometry.
  struct gk_geometry* gk_geom_3d = gkyl_gk_geometry_mapc2p_new(&geometry_input);
  app->gk_geom = gkyl_gk_geometry_deflate(gk_geom_3d, &geometry_input);
  gkyl_gk_geometry_release(gk_geom_3d);
  if (use_gpu) {
    // Copy geometry from host to device.
    struct gk_geometry* gk_geom_dev =
      gkyl_gk_geometry_new(app->gk_geom, &geometry_input, use_gpu);
    gkyl_gk_geometry_release(app->gk_geom);
    app->gk_geom = gkyl_gk_geometry_acquire(gk_geom_dev);
    gkyl_gk_geometry_release(gk_geom_dev);
  }

  // Velocity space mapping.
  struct gkyl_mapc2p_inp c2p_in = {};
  app->gvm = gkyl_velocity_map_new(c2p_in, app->grid, app->grid_vel, app->local,
    app->local_ext, app->local_vel, app->local_ext_vel, use_gpu);

  // Create distribution function arrays (3 for SSP-RK3).
  app->f    = mkarr(use_gpu, app->basis.num_basis, app->local_ext.volume);
  app->f1   = mkarr(use_gpu, app->basis.num_basis, app->local_ext.volume);
  app->fnew = mkarr(use_gpu, app->basis.num_basis, app->local_ext.volume);
  app->f_ho = use_gpu? mkarr(false, app->f->ncomp, app->f->size)
                     : gkyl_array_acquire(app->f);
  gkyl_proj_on_basis* proj_distf =
    gkyl_proj_on_basis_new(&app->grid, &app->basis, inp->poly_order + 1, 1,
                           inp->initial_f_func, inp->initial_f_ctx);
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

  app->cfl = inp->cfl_frac == 0 ? 1.0 : inp->cfl_frac; // CFL factor.

  // CFL frequency in phase-space.
  app->cflrate = mkarr(use_gpu, 1, app->local_ext.volume);

  if (use_gpu)
    app->omega_cfl = gkyl_cu_malloc(sizeof(double));
  else
    app->omega_cfl = gkyl_malloc(sizeof(double));

  // Create the diffusion coefficient array.
  // For now assume 2nd order diffusion in x only.
  int diffusion_order          = 2;
  bool diff_dir[GKYL_MAX_CDIM] = {false};
  int num_diff_dir             = 1;    //number of diffusion directions
  diff_dir[0]                  = true; //direction of the diffusion
  bool is_zero_flux[2 * GKYL_MAX_DIM] = {false}; // Whether to use zero-flux BCs.

  int szD                     = cdim * app->basis_conf.num_basis;
  app->diffD                  = mkarr(use_gpu, szD, app->local_ext_conf.volume);
  struct gkyl_array* diffD_ho = use_gpu ? mkarr(false, app->diffD->ncomp,
                                                app->diffD->size)
                                        : gkyl_array_acquire(app->diffD);
  // Project the diffusion coefficient.
  gkyl_eval_on_nodes* proj_diffD =
    gkyl_eval_on_nodes_new(&app->grid_conf, &app->basis_conf, 1,
                           inp->diffusion_coefficient_func,
                           inp->diffusion_coefficient_ctx);
  gkyl_eval_on_nodes_advance(proj_diffD, 0.0, &app->local_conf, diffD_ho);
  gkyl_eval_on_nodes_release(proj_diffD);
  gkyl_array_copy(app->diffD, diffD_ho);
  gkyl_array_release(diffD_ho);

  // Diffusion solver.
  app->diff_slvr =
    gkyl_dg_updater_diffusion_gyrokinetic_new(&app->grid, &app->basis,
                                              &app->basis_conf, false, diff_dir,
                                              diffusion_order, &app->local_conf,
                                              is_zero_flux, use_gpu);

  // Assume only periodic dir is x.
  app->num_periodic_dir = 1;
  app->periodic_dirs[0] = 0;

  // Things needed for L2 norm diagnostic.
  app->L2_f = mkarr(use_gpu, 1, app->local_ext.volume);
  if (use_gpu) { app->red_L2_f = gkyl_cu_malloc(sizeof(double)); }
  app->integ_L2_f =
    gkyl_dynvec_new(GKYL_DOUBLE, 1); // Dynamic vector to store L2 norm in time.
  app->is_first_integ_L2_write_call = true;

  // Apply BC to the IC.
  apply_bc(app, 0.0, app->f);

  return app;
}

// Compute out = c1*arr1 + c2*arr2
static inline struct gkyl_array* array_combine(struct gkyl_array* out, double c1, const struct gkyl_array* arr1,
                                               double c2, const struct gkyl_array* arr2, const struct gkyl_range* rng)
{
  return gkyl_array_accumulate_range(gkyl_array_set_range(out, c1, arr1, rng),
                                     c2, arr2, rng);
}

void gkyl_diffusion_app_calc_integrated_L2_f(struct gkyl_diffusion_app* app, double tm)
{
  // Calculate the L2 norm of f.
  gkyl_dg_calc_l2_range(app->basis, 0, app->L2_f, 0, app->f, app->local);
  gkyl_array_scale_range(app->L2_f, app->grid.cellVolume, &app->local);

  double L2[1] = {0.0};
  if (app->use_gpu) {
    gkyl_array_reduce_range(app->red_L2_f, app->L2_f, GKYL_SUM, &app->local);
    gkyl_cu_memcpy(L2, app->red_L2_f, sizeof(double), GKYL_CU_MEMCPY_D2H);
  }
  else
    gkyl_array_reduce_range(L2, app->L2_f, GKYL_SUM, &app->local);

  double L2_global[1] = {0.0};
  gkyl_comm_allreduce_host(app->comm, GKYL_DOUBLE, GKYL_SUM, 1, L2, L2_global);

  gkyl_dynvec_append(app->integ_L2_f, tm, L2_global);
}

void gkyl_diffusion_app_write_integrated_L2_f(struct gkyl_diffusion_app* app)
{
  // Write the dynamic vector with the L2 norm of f.
  int rank;
  gkyl_comm_get_rank(app->comm, &rank);
  if (rank == 0)
  {
    // write out integrated L^2
    const char* fmt = "%s-f_%s.gkyl";
    int sz          = gkyl_calc_strlen(fmt, app->name, "L2");
    char fileNm[sz + 1]; // ensures no buffer overflow
    snprintf(fileNm, sizeof fileNm, fmt, app->name, "L2");

    if (app->is_first_integ_L2_write_call) {
      // Write to a new file (this ensure previous output is removed).
      gkyl_dynvec_write(app->integ_L2_f, fileNm);
      app->is_first_integ_L2_write_call = false;
    }
    else
      // Append to existing file.
      gkyl_dynvec_awrite(app->integ_L2_f, fileNm);
  }
  gkyl_dynvec_clear(app->integ_L2_f);
}

// Meta-data for IO.
struct diffusion_output_meta
{
  int frame;              // frame number
  double stime;           // output time
  int poly_order;         // polynomial order
  const char* basis_type; // name of basis functions
  char basis_type_nm[64]; // used during read
};

static struct gkyl_msgpack_data* diffusion_array_meta_new(
  struct diffusion_output_meta meta)
{
  // Allocate new metadata to include in file.
  // Returned gkyl_msgpack_data must be freed using duffusion_array_meta_release.
  struct gkyl_msgpack_data* mt = gkyl_malloc(sizeof(*mt));

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

  if (status != mpack_ok)
  {
    free(mt->meta); // we need to use free here as mpack does its own malloc
    gkyl_free(mt);
    mt = 0;
  }

  return mt;
}

static void diffusion_array_meta_release(struct gkyl_msgpack_data* mt)
{
  // Release array meta data.
  if (!mt) return;
  MPACK_FREE(mt->meta);
  gkyl_free(mt);
}

void gkyl_diffusion_app_write(struct gkyl_diffusion_app* app, double tm, int frame)
{
  // Write grid diagnostics for this app.
  struct gkyl_msgpack_data* mt = diffusion_array_meta_new(
    (struct diffusion_output_meta){.frame      = frame,
                                   .stime      = tm,
                                   .poly_order = app->basis.poly_order,
                                   .basis_type = app->basis.id});

  const char* fmt = "%s-f_%d.gkyl";
  int sz          = gkyl_calc_strlen(fmt, app->name, frame);
  char fileNm[sz + 1]; // ensures no buffer overflow
  snprintf(fileNm, sizeof fileNm, fmt, app->name, frame);

  // copy data from device to host before writing it out
  gkyl_array_copy(app->f_ho, app->f);

  gkyl_comm_array_write(app->comm, &app->grid, &app->local, mt, app->f_ho,
                        fileNm);

  diffusion_array_meta_release(mt);
}

void gkyl_diffusion_app_release(struct gkyl_diffusion_app* app)
{
  // Free memory associated with the app.
  gkyl_array_release(app->L2_f);
  gkyl_dynvec_release(app->integ_L2_f);
  if (app->use_gpu)
    gkyl_cu_free(app->red_L2_f);

  gkyl_dg_updater_diffusion_gyrokinetic_release(app->diff_slvr);
  gkyl_array_release(app->diffD);
  if (app->use_gpu)
    gkyl_cu_free(app->omega_cfl);
  else
    gkyl_free(app->omega_cfl);

  gkyl_array_release(app->cflrate);
  gkyl_array_release(app->f);
  gkyl_array_release(app->f1);
  gkyl_array_release(app->fnew);
  gkyl_array_release(app->f_ho);
  gkyl_array_release(app->bmag);
  gkyl_gk_geometry_release(app->gk_geom);
  gkyl_velocity_map_release(app->gvm);
  gkyl_comm_release(app->comm);
  gkyl_comm_release(app->comm_conf);
  gkyl_rect_decomp_release(app->decomp_conf);
  gkyl_free(app);
}

void calc_integrated_diagnostics(struct gkyl_tm_trigger* iot,
                                 struct gkyl_diffusion_app* app, double t_curr,
                                 bool force_calc)
{
  // Calculate diagnostics integrated over space.
  if (gkyl_tm_trigger_check_and_bump(iot, t_curr) || force_calc)
  {
    gkyl_diffusion_app_calc_integrated_L2_f(app, t_curr);
  }
}

void write_data(struct gkyl_tm_trigger* iot, struct gkyl_diffusion_app* app,
                double t_curr, bool force_write)
{
  // Write grid and integrated diagnostics.
  bool trig_now = gkyl_tm_trigger_check_and_bump(iot, t_curr);
  if (trig_now || force_write)
  {
    int frame = (!trig_now) && force_write ? iot->curr : iot->curr - 1;

    gkyl_diffusion_app_write(app, t_curr, frame);

    gkyl_diffusion_app_calc_integrated_L2_f(app, t_curr);
    gkyl_diffusion_app_write_integrated_L2_f(app);
  }
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

  /* Check if flag < 0 */
  else if (opt == 1)
  {
    errflag = (int*)flagvalue;
    if (*errflag < 0)
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

sunbooleantype first_RHS_call = SUNTRUE;

/* f routine to compute the ODE RHS function f(t,y). */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
  struct gkyl_diffusion_app* app = (struct gkyl_diffusion_app*)user_data;

  struct gkyl_array* fin  = N_VGetVector_Gkylzero(y);
  struct gkyl_array* fout = N_VGetVector_Gkylzero(ydot);

  if (first_RHS_call) first_RHS_call = SUNFALSE;

  gkyl_array_clear(app->cflrate, 0.0);
  gkyl_array_clear(fout, 0.0);

  apply_bc(app, t, fin); //apply_bc before computing the RHS

  gkyl_dg_updater_diffusion_gyrokinetic_advance(app->diff_slvr, &app->local,
                                                app->diffD,
                                                app->gk_geom->jacobgeo_inv, fin,
                                                app->cflrate, fout);

  return 0; /* return with success */
}

/* dom_eig routine to estimate the dominated eigenvalue */
static int dom_eig(sunrealtype t, N_Vector y, N_Vector fn, sunrealtype* lambdaR,
                   sunrealtype* lambdaI, void* user_data, N_Vector temp1,
                   N_Vector temp2, N_Vector temp3)
{
  if (first_RHS_call) f(t, y, fn, user_data);

  struct gkyl_diffusion_app* app = (struct gkyl_diffusion_app*)user_data;

  gkyl_array_reduce_range(app->omega_cfl, app->cflrate, GKYL_MAX, &app->local);

  double omega_cfl_ho;
  if (app->use_gpu)
    gkyl_cu_memcpy(&omega_cfl_ho, app->omega_cfl, sizeof(double),
                   GKYL_CU_MEMCPY_D2H);
  else
    omega_cfl_ho = app->omega_cfl[0];

  double omega_cfl_global_ho;
  gkyl_comm_allreduce_host(app->comm_conf, GKYL_DOUBLE, GKYL_MAX, 1, &omega_cfl_ho, &omega_cfl_global_ho);

  *lambdaR = -omega_cfl_global_ho;
  *lambdaI = SUN_RCONST(0.0);
  return 0; /* return with success */
}

static int apply_bc_in_LSRK(sunrealtype t, N_Vector y, void* user_data)
{
  struct gkyl_diffusion_app* app = (struct gkyl_diffusion_app*)user_data;
  struct gkyl_array* f           = N_VGetVector_Gkylzero(y);

  apply_bc(app, t, f);

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

int STS_init(struct gkyl_diffusion_app* app, UserData* udata, N_Vector* y, void** arkode_mem)
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
    //TODO: find a better way to set the initial random eigenvector

    if(udata->dee_id == 0)
    {
      DEE = SUNDomEigEst_Power(*y, udata->dee_max_iters, udata->dee_reltol, sunctx);
      if (check_flag(DEE, "SUNDomEigEst_Power", 0)) { return 1; }
    }
    else if(udata->dee_id == 1)
    {
      DEE = SUNDomEigEst_Arnoldi(*y, udata->dee_krylov_dim, sunctx);
      if (check_flag(DEE, "SUNDomEigEst_Arnoldi", 0)) { return 1; }
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

int SSP_init(struct gkyl_diffusion_app* app, UserData* udata, N_Vector* y, void** arkode_mem)
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

int gkyl_diffusion_update(struct gkyl_diffusion_app* app, void* arkode_mem, double tout, N_Vector y, sunrealtype* tcurr)
{
  // Call integrator to evolve the solution to time tout
  int flag = ARKodeEvolve(arkode_mem, tout, y, tcurr, ARK_NORMAL);
  if (check_flag(&flag, "ARKodeEvolve", 1)) { return 1; }

  // Check for any CUDA errors during time step
  if (app->use_gpu) checkCuda(cudaGetLastError());
  return 0;
}

double compute_max_error(N_Vector u, N_Vector v, sunrealtype  t_curr, struct gkyl_diffusion_app* app)
{
  struct gkyl_array* udptr = NV_CONTENT_GKZ(u)->dataptr;
  struct gkyl_array* vdptr = NV_CONTENT_GKZ(v)->dataptr;
  double error             = -DBL_MAX;

  // TODO: change code so these allocations only happen once.
  int ncomp = udptr->ncomp;
  struct gkyl_array* wdptr; // Temporary buffer. Should change code to avoid this.
  double* red_ho = gkyl_malloc(ncomp * sizeof(double));
  double *red_local, *red_global;
  if (app->use_gpu) {
    red_local  = gkyl_cu_malloc(ncomp * sizeof(double));
    red_global = gkyl_cu_malloc(ncomp * sizeof(double));
    wdptr = mkarr(true, ncomp, udptr->size);
  }
  else {
    red_local  = gkyl_malloc(ncomp * sizeof(double));
    red_global = gkyl_malloc(ncomp * sizeof(double));
    wdptr = mkarr(false, ncomp, udptr->size);
  }

  gkyl_array_set(wdptr, 1.0, udptr);
  gkyl_array_accumulate(wdptr, -1.0, vdptr);

  gkyl_array_reduce(red_local, wdptr, GKYL_ABS_MAX);
  gkyl_comm_allreduce(app->comm, GKYL_DOUBLE, GKYL_MAX, 1, red_local, red_global);

  if (app->use_gpu)
    gkyl_cu_memcpy(red_ho, red_global, ncomp * sizeof(double), GKYL_CU_MEMCPY_D2H);
  else
    memcpy(red_ho, red_global, ncomp * sizeof(double));

  // Reduce over components.
  for (int i = 0; i < ncomp; i++) error = fmax(error, fabs(red_ho[i]));

  gkyl_free(red_ho);
  if (app->use_gpu) {
    gkyl_cu_free(red_local );
    gkyl_cu_free(red_global);
  }
  else {
    gkyl_free(red_local );
    gkyl_free(red_global);
  }

  gkyl_array_release(wdptr);

  printf("\nmax error is %e at t = %g over the whole domain\n", error, t_curr);

  return error;
}

static inline struct gkyl_comm*
gkyl_diffusion_create_comm(struct gkyl_app_args app_args)
{
  // Construct communicator for use in app.
  struct gkyl_comm *comm;
#ifdef GKYL_HAVE_MPI
  if (app_args.use_gpu && app_args.use_mpi) {
#ifdef GKYL_HAVE_NCCL
    comm = gkyl_nccl_comm_new( &(struct gkyl_nccl_comm_inp) {
        .mpi_comm = MPI_COMM_WORLD,
      }
    );
#else
    printf(" Using -g and -M together requires NCCL.\n");
    assert(0 == 1);
#endif
  }
  else if (app_args.use_mpi) {
    comm = gkyl_mpi_comm_new( &(struct gkyl_mpi_comm_inp) {
        .mpi_comm = MPI_COMM_WORLD,
      }
    );
  }
  else {
    comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
        .use_gpu = app_args.use_gpu
      }
    );
  }
#else
  comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = app_args.use_gpu
    }
  );
#endif
  return comm;
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

#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi) {
    MPI_Init(&argc, &argv);
  }
#endif

  // Create the context struct.
  struct diffusion_ctx ctx = create_diffusion_ctx();

  // Construct communicator for use in app.
  struct gkyl_comm *comm = gkyl_diffusion_create_comm(app_args);

  // Update the context with user inputs.
  ctx.diffD0 = udata->k;
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
  flag = PrintUserData(udata);
  if (check_flag(&flag, "PrintUserData", 1)) { return 1; }

  // Create the struct of app inputs.
  struct gkyl_diffusion_app_inp app_inp = {
    .cdim       = ctx.cdim,
    .vdim       = ctx.vdim,                     // Conf- and vel-space basis.
    .lower      = {ctx.x_min, ctx.vpar_min},    // Lower grid extents.
    .upper      = {ctx.x_max, ctx.vpar_max},    // Upper grid extents.
    .cells      = {ctx.cells[0], ctx.cells[1]}, // Number of cells.
    .poly_order = ctx.poly_order,               // Polynomial order of DG basis.

    .cfl_frac = 0.5, // CFL factor.

    // Mapping from computational to physical space.
    .mapc2p_func = mapc2p,
    .mapc2p_ctx  = &ctx,

    // Magnetic field amplitude.
    .bmag_func = bmag_1x,
    .bmag_ctx  = &ctx,

    // Diffusion coefficient.
    .diffusion_coefficient_func = diffusion_coeff_1x,
    .diffusion_coefficient_ctx  = &ctx,

    // Initial condition.
    .initial_f_func = init_distf_1x1v,
    .initial_f_ctx  = &ctx,

    .use_gpu = app_args.use_gpu, // Whether to run on GPU.
    .cuts = { app_args.cuts[0] },
    .comm = comm,
  };
  strcpy(app_inp.name, ctx.name);

  // Create app object.
  struct gkyl_diffusion_app* app = gkyl_diffusion_app_new(&app_inp);

  // Initial and final simulation times.
  int frame_curr = 0;
  double t_curr = 0.0, t_end = ctx.t_end;

  // Create triggers for IO.
  int num_frames = ctx.num_frames, num_int_diag_calc = ctx.int_diag_calc_num;
  struct gkyl_tm_trigger trig_write        = {.dt    = t_end / num_frames,
                                              .tcurr = t_curr,
                                              .curr  = frame_curr};
  struct gkyl_tm_trigger trig_calc_intdiag = {.dt = t_end /
                                                    GKYL_MAX2(num_frames,
                                                              num_int_diag_calc),
                                              .tcurr = t_curr,
                                              .curr  = frame_curr};

  // Write out ICs (if restart, it overwrites the restart frame).
  calc_integrated_diagnostics(&trig_calc_intdiag, app, t_curr, false);
  write_data(&trig_write, app, t_curr, false);

  double dt = t_end - t_curr; // Initial time step.
  // Initialize small time-step check.
  double dt_init = -1.0, dt_failure_tol = ctx.dt_failure_tol;
  int num_failures = 0, num_failures_max = ctx.num_failures_max;

  void* arkode_mem = NULL; /* empty ARKode memory structure */
  N_Vector y       = NULL; /* empty vector for storing solution */
  N_Vector yref    = NULL; /* empty vector for storing solution */

  struct gkyl_array* fref = mkarr(app->use_gpu, app->basis.num_basis,
                                  app->local_ext.volume);
  gkyl_array_set(fref, 1.0, app->f);

  // compute the reference solution and the error
  SUNContext sunctx;
  flag = SUNContext_Create(SUN_COMM_NULL, &sunctx);
  if (check_flag(&flag, "SUNContext_Create", 1)) { return 1; }

  y    = N_VMake_Gkylzero(app->f, app->use_gpu, sunctx);
  yref = N_VMake_Gkylzero(fref, app->use_gpu, sunctx);

  /* Create the reference solution memory*/
  void* arkode_mem_ref = NULL;
  flag                 = STS_init(app, udata, &yref, &arkode_mem_ref);
  if (check_flag(&flag, "SSP_init", 1)) { return 1; }

  /* Specify the Runge--Kutta--Legendre LSRK method */
  flag = LSRKStepSetSTSMethod(arkode_mem_ref, ARKODE_LSRK_RKL_2);
  if (check_flag(&flag, "LSRKStepSetSTSMethod", 1)) { return 1; }

  /* Specify the fixed step size for the reference STS solution */
  flag = ARKodeSetFixedStep(arkode_mem_ref, 1.0e-5);
  if (check_flag(&flag, "ARKodeSetFixedStep", 1)) { return 1; }

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
    flag      = gkyl_diffusion_update(app, arkode_mem_ref, tout, yref, &t_curr);
    if (check_flag(&flag, "gkyl_diffusion_update", 1)) 
    {
      fprintf(stdout, "** Update method failed! Aborting reference simulation ....\n");
      break;
    }
    ref_end   = clock();
    ref_time += ((double)(ref_end - ref_start)) / CLOCKS_PER_SEC;

    // Update the computed solution
    start = clock();
    flag  = gkyl_diffusion_update(app, arkode_mem, tout, y, &t_curr);
    if (check_flag(&flag, "gkyl_diffusion_update", 1)) 
    {
      fprintf(stdout, "** Update method failed! Aborting simulation ....\n");
      break;
    }
    end = clock();
    sol_time += ((double)(end - start)) / CLOCKS_PER_SEC;

    // Compute the error between the reference and computed solutions
    max_error = fmax(compute_max_error(y, yref, t_curr, app), max_error);

    calc_integrated_diagnostics(&trig_calc_intdiag, app, t_curr, t_curr > t_end);
    write_data(&trig_write, app, t_curr, t_curr > t_end);

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
  gkyl_diffusion_app_release(app);

  return 0;
}
