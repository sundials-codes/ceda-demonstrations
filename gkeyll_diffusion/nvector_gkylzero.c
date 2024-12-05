/* -----------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds and Mustafa Aggul @ SMU
 * -----------------------------------------------------------------
 * This is the implementation file for an NVECTOR wrapper of the
 * Gkylzero data structure.
 * -----------------------------------------------------------------*/


#include <nvector/nvector_gkylzero.h>

/* -----------------------------------------------------------------
 * Simplifying macro: NV_CONTENT_GKZ
 *
 * This gives access to the Gkylzero vector from within the NVECTOR.
 * -----------------------------------------------------------------*/

#define NV_CONTENT_GKZ(v) ((N_VectorContent_Gkylzero)(v->content))

/* -----------------------------------------------------------------
 * exported functions
 * -----------------------------------------------------------------*/

/* Returns vector type ID. */
N_Vector_ID N_VGetVectorID_Gkylzero(SUNDIALS_MAYBE_UNUSED N_Vector v)
{
  return SUNDIALS_NVEC_CUSTOM;
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
  v->ops->nvgetvectorid     = N_VGetVectorID_Gkylzero;
  v->ops->nvclone           = N_VClone_Gkylzero;
  v->ops->nvcloneempty      = N_VCloneEmpty_Gkylzero;
  v->ops->nvdestroy         = N_VDestroy_Gkylzero;
  v->ops->nvgetcommunicator = N_VGetCommunicator_Gkylzero;
  v->ops->nvgetlength       = N_VGetLength_Gkylzero;

  /* vector operations */
  v->ops->nvlinearsum    = N_VLinearSum_Gkylzero;
  v->ops->nvconst        = N_VConst_Gkylzero;
  v->ops->nvprod         = N_VProd_Gkylzero;
  v->ops->nvdiv          = N_VDiv_Gkylzero;
  v->ops->nvscale        = N_VScale_Gkylzero;
  v->ops->nvabs          = N_VAbs_Gkylzero;
  v->ops->nvinv          = N_VInv_Gkylzero;
  v->ops->nvaddconst     = N_VAddConst_Gkylzero;
  v->ops->nvdotprod      = N_VDotProd_Gkylzero;
  v->ops->nvmaxnorm      = N_VMaxNorm_Gkylzero;
  v->ops->nvwrmsnormmask = N_VWrmsNormMask_Gkylzero;
  v->ops->nvwrmsnorm     = N_VWrmsNorm_Gkylzero;
  v->ops->nvmin          = N_VMin_Gkylzero;
  v->ops->nvwl2norm      = N_VWL2Norm_Gkylzero;
  v->ops->nvl1norm       = N_VL1Norm_Gkylzero;
  v->ops->nvcompare      = N_VCompare_Gkylzero;
  v->ops->nvinvtest      = N_VInvTest_Gkylzero;
  v->ops->nvconstrmask   = N_VConstrMask_Gkylzero;
  v->ops->nvminquotient  = N_VMinQuotient_Gkylzero;
  v->ops->nvlinearcombination = N_VLinearCombination_Gkylzero;
  v->ops->nvscaleaddmulti     = N_VScaleAddMulti_Gkylzero;
  v->ops->nvdotprodmulti      = N_VDotProdMulti_Gkylzero;
  v->ops->nvdotprodlocal     = N_VDotProdLocal_Gkylzero;
  v->ops->nvmaxnormlocal     = N_VMaxNormLocal_Gkylzero;
  v->ops->nvminlocal         = N_VMinLocal_Gkylzero;
  v->ops->nvl1normlocal      = N_VL1NormLocal_Gkylzero;
  v->ops->nvinvtestlocal     = N_VInvTestLocal_Gkylzero;
  v->ops->nvconstrmasklocal  = N_VConstrMaskLocal_Gkylzero;
  v->ops->nvminquotientlocal = N_VMinQuotientLocal_Gkylzero;
  v->ops->nvwsqrsumlocal     = N_VWSqrSumLocal_Gkylzero;
  v->ops->nvwsqrsummasklocal = N_VWSqrSumMaskLocal_Gkylzero;
  v->ops->nvdotprodmultilocal     = N_VDotProdMultiLocal_Gkylzero;
  v->ops->nvdotprodmultiallreduce = N_VDotProdMultiAllReduce_Gkylzero;

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

  /* Attach lengths and communicator */
  content->local_length  = local_length;
  content->global_length = global_length;
  content->own_vector    = SUNFALSE;
  content->x             = NULL;

  return (v);
}

/* Create a Gkylzero N_Vector wrapper around user supplied gkl_array. */
N_Vector N_VMake_Gkylzero(struct gkyl_array* x, SUNContext sunctx)
{
  N_Vector v;
  v = NULL;
  v = N_VNewEmpty_Gkylzero(sunctx);
  if (v == NULL) { return (NULL); }
  NV_CONTENT_GKZ(v)->own_vector = SUNFALSE;
  NV_CONTENT_GKZ(v)->x = x;
  return (v);
}

/* Extract Gkylzero vector */
struct gkyl_array* N_VGetVector_Gkylzero(N_Vector v)
{
  return NV_CONTENT_GKZ(v)->x;
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
  content->x          = NULL;

  return (v);
}

N_Vector N_VClone_Gkylzero(N_Vector w)
{
  N_Vector v;
  struct gkyl_array* vx;
  struct gkyl_array* wx = NV_CONTENT_GKZ(v)->x;

  v = NULL;
  v = N_VCloneEmpty_Gkylzero(w);
  if (v == NULL) { return (NULL); }

  vx = ; /* CALL GKYLZERO VECTOR "CLONE" ROUTINE TO ALLOCATE MEMORY FOR vx */

  NV_CONTENT_GKZ(v)->x = vx;
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
    if (NV_CONTENT_GKZ(v)->own_vector && NV_CONTENT_GKZ(v)->x != NULL)
    {
      /* CALL GKYLZERO "FREE" ROUTINE TO DEALLOCATE MEMORY FROM INSIDE NV_CONTENT_GKZ(v) */
      NV_CONTENT_GKZ(v)->x = NULL;
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

MPI_Comm N_VGetCommunicator_Gkylzero(N_Vector v)
{
  /* extract MPI communicator from Gkylzero vector and return */
}

sunindextype N_VGetLength_Gkylzero(N_Vector v)
{
  /* extract global vector length from Gkylzero vector and return */
}

void N_VLinearSum_Gkylzero(sunrealtype a, N_Vector x, sunrealtype b, N_Vector y,
                           N_Vector z)
{
  /* call linear sum operation from Gkylzero vector and return */
}

void N_VConst_Gkylzero(sunrealtype c, N_Vector z)
{
  /* set all entries in Gkylzero vector to c and return */
}

void N_VProd_Gkylzero(N_Vector x, N_Vector y, N_Vector z)
{
  /* fill this in */
}

void N_VDiv_Gkylzero(N_Vector x, N_Vector y, N_Vector z)
{
  /* fill this in */
}

void N_VScale_Gkylzero(sunrealtype c, N_Vector x, N_Vector z)
{
  /* fill this in */
}

void N_VAbs_Gkylzero(N_Vector x, N_Vector z)
{
  /* fill this in */
}

void N_VInv_Gkylzero(N_Vector x, N_Vector z)
{
  /* fill this in */
}

void N_VAddConst_Gkylzero(N_Vector x, sunrealtype b, N_Vector z)
{
  /* fill this in */
}

sunrealtype N_VDotProdLocal_Gkylzero(N_Vector x, N_Vector y)
{
  /* fill this in */
}

sunrealtype N_VDotProd_Gkylzero(N_Vector x, N_Vector y)
{
  /* fill this in */
}

sunrealtype N_VMaxNormLocal_Gkylzero(N_Vector x)
{
  /* fill this in */
}

sunrealtype N_VMaxNorm_Gkylzero(N_Vector x)
{
  /* fill this in */
}

sunrealtype N_VWSqrSumLocal_Gkylzero(N_Vector x, N_Vector w)
{
  /* fill this in */
}

sunrealtype N_VWrmsNorm_Gkylzero(N_Vector x, N_Vector w)
{
  /* fill this in */
}

sunrealtype N_VWSqrSumMaskLocal_Gkylzero(N_Vector x, N_Vector w, N_Vector id)
{
  /* fill this in */
}

sunrealtype N_VWrmsNormMask_Gkylzero(N_Vector x, N_Vector w, N_Vector id)
{
  /* fill this in */
}

sunrealtype N_VMinLocal_Gkylzero(N_Vector x)
{
  /* fill this in */
}

sunrealtype N_VMin_Gkylzero(N_Vector x)
{
  /* fill this in */
}

sunrealtype N_VWL2Norm_Gkylzero(N_Vector x, N_Vector w)
{
  /* fill this in */
}

sunrealtype N_VL1NormLocal_Gkylzero(N_Vector x)
{
  /* fill this in */
}

sunrealtype N_VL1Norm_Gkylzero(N_Vector x)
{
  /* fill this in */
}

void N_VCompare_Gkylzero(sunrealtype c, N_Vector x, N_Vector z)
{
  /* fill this in */
}

sunbooleantype N_VInvTestLocal_Gkylzero(N_Vector x, N_Vector z)
{
  /* fill this in */
}

sunbooleantype N_VInvTest_Gkylzero(N_Vector x, N_Vector z)
{
  /* fill this in */
}

sunbooleantype N_VConstrMaskLocal_Gkylzero(N_Vector c, N_Vector x, N_Vector m)
{
  /* fill this in */
}

sunbooleantype N_VConstrMask_Gkylzero(N_Vector c, N_Vector x, N_Vector m)
{
  /* fill this in */
}

sunrealtype N_VMinQuotientLocal_Gkylzero(N_Vector num, N_Vector denom)
{
  /* fill this in */
}

sunrealtype N_VMinQuotient_Gkylzero(N_Vector num, N_Vector denom)
{
  /* fill this in */
}

SUNErrCode N_VLinearCombination_Gkylzero(int nvec, sunrealtype* c, N_Vector* X,
                                         N_Vector z)
{
  /* fill this in */
}

SUNErrCode N_VScaleAddMulti_Gkylzero(int nvec, sunrealtype* a, N_Vector x,
                                     N_Vector* Y, N_Vector* Z)
{
  /* fill this in */
}

SUNErrCode N_VDotProdMulti_Gkylzero(int nvec, N_Vector x, N_Vector* Y,
                                    sunrealtype* dotprods)
{
  /* fill this in */
}

SUNErrCode N_VDotProdMultiLocal_Gkylzero(int nvec, N_Vector x, N_Vector* Y,
                                         sunrealtype* dotprods)
{
  /* fill this in */
}

SUNErrCode N_VDotProdMultiAllReduce_Gkylzero(int nvec, N_Vector x, sunrealtype* sum)
{
  /* fill this in */
}
