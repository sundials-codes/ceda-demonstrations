/* -----------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds and Mustafa Aggul @ UMBC
 * -----------------------------------------------------------------
 * This is the implementation file for an NVECTOR wrapper of the
 * Gkylzero data structure.
 * -----------------------------------------------------------------*/

#include "nvector_gkylzero.h"
#include <gkyl_alloc.h>
#include <string.h> // for memcpy.

/* -----------------------------------------------------------------
 * Simplifying macro: NV_CONTENT_GKZ
 *
 * This gives access to the Gkylzero vector from within the NVECTOR.
 * -----------------------------------------------------------------*/

/* -----------------------------------------------------------------
 * exported functions
 * -----------------------------------------------------------------*/

struct gkyl_array* mkarr(bool on_gpu, long nc, long size)
{
  // Allocate array (filled with zeros).
  struct gkyl_array* a;
  if (on_gpu) a = gkyl_array_cu_dev_new(GKYL_DOUBLE, nc, size);
  else a = gkyl_array_new(GKYL_DOUBLE, nc, size);
  return a;
}

/* Construct a new Gkylzero vector without the underlying Gkylzero vector */
N_Vector N_VNewEmpty_Gkylzero(SUNContext sunctx)
{
  N_Vector v;
  N_VectorContent_Gkylzero content;

  /* Create an empty vector object */
  v = NULL;
  v = N_VNewEmpty(sunctx);
  if (v == NULL) { return (NULL); }

  /* Attach operations -- DELETE ANY THAT WERE DELETED FROM nvector_gkylzero.h */

  /* constructors, destructors, and utility operations */
  v->ops->nvclone      = N_VClone_Gkylzero;
  v->ops->nvcloneempty = N_VCloneEmpty_Gkylzero;
  v->ops->nvdestroy    = N_VDestroy_Gkylzero;

  /* vector operations */
  v->ops->nvlinearsum = N_VLinearSum_Gkylzero;
  v->ops->nvconst     = N_VConst_Gkylzero;
  v->ops->nvscale     = N_VScale_Gkylzero;
  v->ops->nvwrmsnorm  = N_VWrmsNorm_abs_comp_Gkylzero;
  v->ops->nvdotprod   = N_VDotProd_Gkylzero;
  v->ops->nvspace     = N_VSpace_Gkylzero;
  v->ops->nvdiv       = N_VDiv_Gkylzero;
  v->ops->nvabs       = N_VAbs_Gkylzero;
  v->ops->nvinv       = N_VInv_Gkylzero;
  v->ops->nvmaxnorm   = N_VMaxnorm_Gkylzero;
  v->ops->nvaddconst  = N_VAddconst_Gkylzero;

  /* Create content */
  content = NULL;
  content = (N_VectorContent_Gkylzero)malloc(sizeof *content);
  if (content == NULL)
  {
    N_VDestroy(v);
    return (NULL);
  }

  /* Attach content */
  v->content = content;

  /* Attach gkyl_array */
  content->own_vector = SUNFALSE;
  content->use_gpu    = SUNFALSE;
  content->dataptr    = NULL;

  return (v);
}

/* Create a Gkylzero N_Vector wrapper around user supplied gkl_array. */
N_Vector N_VMake_Gkylzero(struct gkyl_array* x, sunbooleantype use_gpu,
                          SUNContext sunctx)
{
  N_Vector v;
  v = NULL;
  v = N_VNewEmpty_Gkylzero(sunctx);
  if (v == NULL) { return (NULL); }
  NV_CONTENT_GKZ(v)->own_vector = SUNFALSE;
  NV_CONTENT_GKZ(v)->use_gpu    = use_gpu;
  NV_CONTENT_GKZ(v)->dataptr    = x;
  return (v);
}

/* Extract Gkylzero vector */
struct gkyl_array* N_VGetVector_Gkylzero(N_Vector v)
{
  return NV_CONTENT_GKZ(v)->dataptr;
}

/* -----------------------------------------------------------------
 * implementation of vector operations -- DELETE ALL THAT WILL BE UNUSED BY LSRKSTEP
 * -----------------------------------------------------------------*/

N_Vector N_VCloneEmpty_Gkylzero(N_Vector w)
{
  N_Vector v;
  N_VectorContent_Gkylzero content;
  if (w == NULL) { return (NULL); }

  /* Create vector */
  v = NULL;
  v = N_VNewEmpty(w->sunctx);
  if (v == NULL) { return (NULL); }

  /* Attach operations */
  if (N_VCopyOps(w, v))
  {
    N_VDestroy(v);
    return (NULL);
  }

  /* Create content */
  content = NULL;
  content = (N_VectorContent_Gkylzero)malloc(sizeof *content);
  if (content == NULL)
  {
    N_VDestroy(v);
    return (NULL);
  }

  /* Attach content */
  v->content = content;

  /* Initialize content */
  content->own_vector = SUNFALSE;
  /* N_VCloneEmpty_Gkylzero must be called only in
     N_VClone_Gkylzero to ensure use_gpu has the correct flag.
     Otherwise, use_gpu flag will be false even if it must be true*/

  //TO DO: Check to verify if this function is called separately
  content->use_gpu = SUNFALSE;
  content->dataptr = NULL;

  return (v);
}

N_Vector N_VClone_Gkylzero(N_Vector w)
{
  N_Vector v;
  struct gkyl_array* vdptr;
  struct gkyl_array* wdptr = NV_CONTENT_GKZ(w)->dataptr;

  v = NULL;
  v = N_VCloneEmpty_Gkylzero(w);
  if (v == NULL) { return (NULL); }

  vdptr = mkarr(NV_CONTENT_GKZ(w)->use_gpu, wdptr->ncomp, wdptr->size);

  NV_CONTENT_GKZ(v)->dataptr    = vdptr;
  NV_CONTENT_GKZ(v)->use_gpu    = NV_CONTENT_GKZ(w)->use_gpu;
  NV_CONTENT_GKZ(v)->own_vector = SUNTRUE;
  return (v);
}

void N_VDestroy_Gkylzero(N_Vector v)
{
  if (v == NULL) { return; }

  /* free content */
  if (v->content != NULL)
  {
    /* free the Gkylzero parvector if it's owned by the vector wrapper */
    if (NV_CONTENT_GKZ(v)->own_vector && NV_CONTENT_GKZ(v)->dataptr != NULL)
    {
      gkyl_array_release(NV_CONTENT_GKZ(v)->dataptr);
      NV_CONTENT_GKZ(v)->dataptr = NULL;
    }
    free(v->content);
    v->content = NULL;
  }

  /* free ops and vector */
  if (v->ops != NULL)
  {
    free(v->ops);
    v->ops = NULL;
  }
  free(v);
  v = NULL;

  return;
}

/* returns z = a*x + b*y */
void N_VLinearSum_Gkylzero(sunrealtype a, N_Vector x, sunrealtype b, N_Vector y,
                           N_Vector z)
{
  struct gkyl_array* xdptr = NV_CONTENT_GKZ(x)->dataptr;
  struct gkyl_array* ydptr = NV_CONTENT_GKZ(y)->dataptr;
  struct gkyl_array* zdptr = NV_CONTENT_GKZ(z)->dataptr;

  gkyl_array_comp_op(zdptr, GKYL_AXPBY, a, xdptr, b, ydptr);
}

void N_VConst_Gkylzero(sunrealtype c, N_Vector z)
{
  struct gkyl_array* zdptr = NV_CONTENT_GKZ(z)->dataptr;

  gkyl_array_clear(zdptr, c);
}

void N_VScale_Gkylzero(sunrealtype c, N_Vector x, N_Vector z)
{
  struct gkyl_array* xdptr = NV_CONTENT_GKZ(x)->dataptr;
  struct gkyl_array* zdptr = NV_CONTENT_GKZ(z)->dataptr;

  gkyl_array_set(zdptr, c, xdptr);
}

sunrealtype N_VWrmsNorm_abs_comp_Gkylzero(N_Vector x, N_Vector w)
{
  struct gkyl_array* xdptr = NV_CONTENT_GKZ(x)->dataptr;
  struct gkyl_array* wdptr = NV_CONTENT_GKZ(w)->dataptr;

  // TODO: change code so these allocations only happen once.
  int ncomp = xdptr->ncomp;
  struct gkyl_array* zdptr; // Temporary buffer. Should change code to avoid this.
  double* asum_ho = gkyl_malloc(ncomp * sizeof(double));
  double* asum;
  if (gkyl_array_is_cu_dev(xdptr))
  {
    asum  = gkyl_cu_malloc(ncomp * sizeof(double));
    zdptr = mkarr(true, ncomp, xdptr->size);
  }
  else
  {
    asum  = gkyl_malloc(ncomp * sizeof(double));
    zdptr = mkarr(false, ncomp, xdptr->size);
  }

  gkyl_array_comp_op(zdptr, GKYL_PROD, 1.0, xdptr, 0.0, wdptr);
  gkyl_array_reduce(asum, zdptr, GKYL_SQ_SUM);

  if (gkyl_array_is_cu_dev(xdptr))
    gkyl_cu_memcpy(asum_ho, asum, ncomp * sizeof(double), GKYL_CU_MEMCPY_D2H);
  else memcpy(asum_ho, asum, ncomp * sizeof(double));

  // Sum over compontents, divide by number degrees of freedom and take sqrt.
  sunrealtype asum_out = 0.0;
  for (int i = 0; i < ncomp; i++) asum_out += asum_ho[i];

  asum_out = SUNRsqrt(asum_out / (xdptr->size * ncomp));

  gkyl_free(asum_ho);
  if (gkyl_array_is_cu_dev(xdptr)) gkyl_cu_free(asum);
  else gkyl_free(asum);

  gkyl_array_release(zdptr);

  return asum_out;
}

sunrealtype N_VWrmsNorm_cell_norm_Gkylzero(N_Vector x, N_Vector w)
{
  struct gkyl_array* xdptr = NV_CONTENT_GKZ(x)->dataptr;
  struct gkyl_array* wdptr = NV_CONTENT_GKZ(w)->dataptr;

  // TODO: change code so these allocations only happen once.
  int ncomp      = xdptr->ncomp;
  double* red_ho = gkyl_malloc(ncomp * sizeof(double));
  double* red;
  if (gkyl_array_is_cu_dev(xdptr)) red = gkyl_cu_malloc(ncomp * sizeof(double));
  else red = gkyl_malloc(ncomp * sizeof(double));

  // Reduce over cells.
  gkyl_array_reduce_weighted(red, xdptr, wdptr, GKYL_SQ_SUM);

  if (gkyl_array_is_cu_dev(xdptr))
    gkyl_cu_memcpy(red_ho, red, ncomp * sizeof(double), GKYL_CU_MEMCPY_D2H);
  else memcpy(red_ho, red, ncomp * sizeof(double));

  // Reduce over components.
  sunrealtype red_out = 0.0;
  for (sunindextype i = 0; i < ncomp; ++i) { red_out += red_ho[i]; }
  red_out = SUNRsqrt(red_out / xdptr->size);

  gkyl_free(red_ho);
  if (gkyl_array_is_cu_dev(xdptr)) gkyl_cu_free(red);
  else gkyl_free(red);

  return red_out;
}

//TODO: update this function to its GPU version
sunrealtype N_VDotProd_Gkylzero(N_Vector x, N_Vector y)
{
  struct gkyl_array* xdptr = NV_CONTENT_GKZ(x)->dataptr;
  struct gkyl_array* ydptr = NV_CONTENT_GKZ(y)->dataptr;

  sunrealtype* x_data = xdptr->data;
  sunrealtype* y_data = ydptr->data;

  sunindextype N       = (xdptr->size * xdptr->ncomp);
  sunrealtype dot_prod = 0.0;

  for (sunindextype i = 0; i < N; ++i) { dot_prod += x_data[i] * y_data[i]; }

  return dot_prod;
}

//Will be removed soon. No need to create a new function.
void N_VSpace_Gkylzero(N_Vector v, sunindextype* x, sunindextype* y)
{
  *x = 0;
  *y = 0;
}

void N_VDiv_Gkylzero(N_Vector u, N_Vector v, N_Vector w)
{
  struct gkyl_array* udptr = NV_CONTENT_GKZ(u)->dataptr;
  struct gkyl_array* vdptr = NV_CONTENT_GKZ(v)->dataptr;
  struct gkyl_array* wdptr = NV_CONTENT_GKZ(w)->dataptr;

  /* SUN_RCONST(1.0) values are unused dummy variables */
  gkyl_array_comp_op(wdptr, GKYL_DIV, SUN_RCONST(1.0), udptr, SUN_RCONST(0.0),
                     vdptr);

  return;
}

void N_VAbs_Gkylzero(N_Vector u, N_Vector v)
{
  struct gkyl_array* udptr = NV_CONTENT_GKZ(u)->dataptr;
  struct gkyl_array* vdptr = NV_CONTENT_GKZ(v)->dataptr;

  /* SUN_RCONST(1.0) values and the last vdptr pointer are unused dummy variables */
  gkyl_array_comp_op(vdptr, GKYL_ABS, SUN_RCONST(1.0), udptr, SUN_RCONST(1.0),
                     vdptr);

  return;
}

void N_VInv_Gkylzero(N_Vector u, N_Vector v)
{
  struct gkyl_array* udptr = NV_CONTENT_GKZ(u)->dataptr;
  struct gkyl_array* vdptr = NV_CONTENT_GKZ(v)->dataptr;

  /* SUN_RCONST(1.0) values and the last vdptr pointer are unused dummy variables */
  gkyl_array_comp_op(vdptr, GKYL_INV, SUN_RCONST(1.0), udptr, SUN_RCONST(1.0),
                     vdptr);

  return;
}

//use gkyl_array_comp_op!
sunrealtype N_VMaxnorm_Gkylzero(N_Vector u)
{
  struct gkyl_array* udptr = NV_CONTENT_GKZ(u)->dataptr;

  sunrealtype red_ho[udptr->ncomp];
  double* red;
  if (gkyl_array_is_cu_dev(udptr))
    red = gkyl_cu_malloc(udptr->ncomp * sizeof(double));
  else red = gkyl_malloc(udptr->ncomp * sizeof(double));

  gkyl_array_reduce(red, udptr, GKYL_ABS_MAX);

  if (gkyl_array_is_cu_dev(udptr))
    gkyl_cu_memcpy(red_ho, red, udptr->ncomp * sizeof(double),
                   GKYL_CU_MEMCPY_D2H);
  else memcpy(red_ho, red, udptr->ncomp * sizeof(double));

  sunrealtype max = 0.0;
  for (sunindextype i = 0; i < udptr->ncomp; ++i)
  {
    max = fmax(max, red_ho[i]);
  }

  if (gkyl_array_is_cu_dev(udptr)) gkyl_cu_free(red);
  else gkyl_free(red);

  return (max);
}

void N_VAddconst_Gkylzero(N_Vector u, sunrealtype x, N_Vector v)
{
  struct gkyl_array* udptr = NV_CONTENT_GKZ(u)->dataptr;
  struct gkyl_array* vdptr = NV_CONTENT_GKZ(v)->dataptr;

  gkyl_array_copy(vdptr, udptr);

  sunindextype N = udptr->ncomp;

  for (sunindextype i = 0; i < N; ++i) { gkyl_array_shiftc(vdptr, x, i); }

  return;
}
