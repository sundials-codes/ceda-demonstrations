/* -----------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds and Mustafa Aggul @ SMU
 * -----------------------------------------------------------------
 * This is the implementation file for an NVECTOR wrapper of the
 * Gkylzero data structure.
 * -----------------------------------------------------------------*/


#include "nvector_gkylzero.h"

/* -----------------------------------------------------------------
 * Simplifying macro: NV_CONTENT_GKZ
 *
 * This gives access to the Gkylzero vector from within the NVECTOR.
 * -----------------------------------------------------------------*/

#define NV_CONTENT_GKZ(v) ((N_VectorContent_Gkylzero)(v->content))

/* -----------------------------------------------------------------
 * exported functions
 * -----------------------------------------------------------------*/

struct gkyl_array*
mkarr(bool on_gpu, long nc, long size)
{
  // Allocate array (filled with zeros).
  struct gkyl_array* a;
  if (on_gpu)
    a = gkyl_array_cu_dev_new(GKYL_DOUBLE, nc, size);
  else
    a = gkyl_array_new(GKYL_DOUBLE, nc, size);
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
  v->ops->nvclone           = N_VClone_Gkylzero;
  v->ops->nvcloneempty      = N_VCloneEmpty_Gkylzero;
  v->ops->nvdestroy         = N_VDestroy_Gkylzero;

  /* vector operations */
  v->ops->nvlinearsum    = N_VLinearSum_Gkylzero;
  v->ops->nvconst        = N_VConst_Gkylzero;
  v->ops->nvscale        = N_VScale_Gkylzero;
  v->ops->nvwrmsnorm     = N_VWrmsNorm_Gkylzero;
  v->ops->nvspace        = N_VSpace_Gkylzero;
  v->ops->nvdiv          = N_VDiv_Gkylzero;
  v->ops->nvabs          = N_VAbs_Gkylzero;
  v->ops->nvinv          = N_VInv_Gkylzero;
  v->ops->nvmaxnorm      = N_VMaxnorm_Gkylzero;
  v->ops->nvaddconst     = N_VAddconst_Gkylzero;



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
  content->own_vector    = SUNFALSE;
  content->use_gpu       = SUNFALSE;
  content->dataptr       = NULL;

  return (v);
}

/* Create a Gkylzero N_Vector wrapper around user supplied gkl_array. */
N_Vector N_VMake_Gkylzero(struct gkyl_array* x, sunbooleantype use_gpu, SUNContext sunctx)
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
  content->dataptr    = NULL;

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

  NV_CONTENT_GKZ(v)->dataptr = vdptr;
  NV_CONTENT_GKZ(v)->use_gpu = NV_CONTENT_GKZ(w)->use_gpu;
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

  gkyl_array_set(zdptr, a, xdptr);
  gkyl_array_accumulate(zdptr, b, ydptr);
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

sunrealtype N_VWrmsNorm_Gkylzero(N_Vector x, N_Vector w)
{
  sunrealtype asum, prodi;

  struct gkyl_array* xdptr = NV_CONTENT_GKZ(x)->dataptr;
  struct gkyl_array* wdptr = NV_CONTENT_GKZ(w)->dataptr;

  sunrealtype *x_data = xdptr->data;
  sunrealtype *w_data = wdptr->data;

  sunindextype N = (xdptr->size*xdptr->ncomp);
  asum = 0.0;

  for (sunindextype i=0; i<N; ++i) {
    prodi = x_data[i] * w_data[i];
    asum += SUNSQR(prodi);
  }
  asum = SUNRsqrt(asum/N);
  return asum;
}

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

  sunrealtype *u_data = udptr->data;
  sunrealtype *v_data = vdptr->data;
  sunrealtype *w_data = wdptr->data;

  sunindextype N = (udptr->size*udptr->ncomp);

  for (sunindextype i=0; i<N; ++i) {
    w_data[i] = u_data[i] / v_data[i];
  }

  return;
}

void N_VAbs_Gkylzero(N_Vector u, N_Vector v)
{
  struct gkyl_array* udptr = NV_CONTENT_GKZ(u)->dataptr;
  struct gkyl_array* vdptr = NV_CONTENT_GKZ(v)->dataptr;

  sunrealtype *u_data = udptr->data;
  sunrealtype *v_data = vdptr->data;

  sunindextype N = (udptr->size*udptr->ncomp);

  for (sunindextype i=0; i<N; ++i) {
    v_data[i] = SUNRabs(u_data[i]);
  }

  return;
}

void N_VInv_Gkylzero(N_Vector u, N_Vector v)
{
  struct gkyl_array* udptr = NV_CONTENT_GKZ(u)->dataptr;
  struct gkyl_array* vdptr = NV_CONTENT_GKZ(v)->dataptr;

  sunrealtype *u_data = udptr->data;
  sunrealtype *v_data = vdptr->data;

  sunindextype N = (udptr->size*udptr->ncomp);

  for (sunindextype i=0; i<N; ++i) {
    v_data[i] = 1.0 / u_data[i];
  }

  return;
}

sunrealtype N_VMaxnorm_Gkylzero(N_Vector u)
{
  struct gkyl_array* udptr = NV_CONTENT_GKZ(u)->dataptr;

  sunrealtype *u_data = udptr->data;

  sunindextype N = (udptr->size*udptr->ncomp);

  sunrealtype max = 0.0;

  for (sunindextype i=0; i<N; ++i) {
    if (SUNRabs(u_data[i]) > max) { max = SUNRabs(u_data[i]); }
  }

  return (max);
}

void N_VAddconst_Gkylzero(N_Vector u, sunrealtype x ,N_Vector v)
{
  struct gkyl_array* udptr = NV_CONTENT_GKZ(u)->dataptr;
  struct gkyl_array* vdptr = NV_CONTENT_GKZ(v)->dataptr;

  sunrealtype *u_data = udptr->data;
  sunrealtype *v_data = vdptr->data;

  sunindextype N = (udptr->size*udptr->ncomp);

  for (sunindextype i=0; i<N; ++i) {
    v_data[i] = u_data[i] + x;
  }

  return;
}