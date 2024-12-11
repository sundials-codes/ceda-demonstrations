/* -----------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds and Mustafa Aggul @ SMU
 * -----------------------------------------------------------------
 * This is the main header file for an NVECTOR wrapper of the
 * Gkylzero data structure.
 * -----------------------------------------------------------------*/

#ifndef _NVECTOR_GKYLZERO_H
#define _NVECTOR_GKYLZERO_H

#include <stdio.h>
#include <stdlib.h>
#include <sundials/sundials_nvector.h>

/* gkylzero header files -- ADD MORE AS NECESSARY */
#include <gkyl_array.h>
#include <gkyl_array_ops.h>

#ifdef __cplusplus /* wrapper to enable C++ usage */
extern "C" {
#endif

/* -----------------------------------------------------------------
 * Gkylzero implementation of N_Vector
 * -----------------------------------------------------------------*/

struct _N_VectorContent_Gkylzero
{
  sunbooleantype own_vector; /* ownership Gkylzero vector */
  struct gkyl_array* x;      /* the actual Gkylzero object pointer */
  /* ADD ANY GKEYLLZERO OBJECTS HERE THAT WILL BE NEEDED, E.G., MPI COMMUNICATOR */
};

typedef struct _N_VectorContent_Gkylzero* N_VectorContent_Gkylzero;

/* -----------------------------------------------------------------
 * Functions exported by nvector_gkylzero
 * -----------------------------------------------------------------*/

N_Vector N_VNewEmpty_Gkylzero(SUNContext sunctx);
N_Vector N_VMake_Gkylzero(struct gkyl_array* x, SUNContext sunctx);
struct gkyl_array* N_VGetVector_Gkylzero(N_Vector v);

N_Vector_ID N_VGetVectorID_Gkylzero(N_Vector v);
N_Vector N_VCloneEmpty_Gkylzero(N_Vector w);
N_Vector N_VClone_Gkylzero(N_Vector w);
void N_VDestroy_Gkylzero(N_Vector v);
MPI_Comm N_VGetCommunicator_Gkylzero(N_Vector v);
sunindextype N_VGetLength_Gkylzero(N_Vector v);

/* vector operations -- DELETE ALL THAT WILL BE UNUSED BY LSRKSTEP */
void N_VLinearSum_Gkylzero(sunrealtype a, N_Vector x, sunrealtype b, N_Vector y, N_Vector z);
void N_VConst_Gkylzero(sunrealtype c, N_Vector z);
void N_VProd_Gkylzero(N_Vector x, N_Vector y, N_Vector z);
void N_VDiv_Gkylzero(N_Vector x, N_Vector y, N_Vector z);
void N_VScale_Gkylzero(sunrealtype c, N_Vector x, N_Vector z);
void N_VAbs_Gkylzero(N_Vector x, N_Vector z);
void N_VInv_Gkylzero(N_Vector x, N_Vector z);
void N_VAddConst_Gkylzero(N_Vector x, sunrealtype b, N_Vector z);
sunrealtype N_VDotProd_Gkylzero(N_Vector x, N_Vector y);
sunrealtype N_VMaxNorm_Gkylzero(N_Vector x);
sunrealtype N_VWrmsNorm_Gkylzero(N_Vector x, N_Vector w);
sunrealtype N_VWrmsNormMask_Gkylzero(N_Vector x, N_Vector w, N_Vector id);
sunrealtype N_VMin_Gkylzero(N_Vector x);
sunrealtype N_VWL2Norm_Gkylzero(N_Vector x, N_Vector w);
sunrealtype N_VL1Norm_Gkylzero(N_Vector x);
void N_VCompare_Gkylzero(sunrealtype c, N_Vector x, N_Vector z);
sunbooleantype N_VInvTest_Gkylzero(N_Vector x, N_Vector z);
sunbooleantype N_VConstrMask_Gkylzero(N_Vector c, N_Vector x, N_Vector m);
sunrealtype N_VMinQuotient_Gkylzero(N_Vector num, N_Vector denom);
SUNErrCode N_VLinearCombination_Gkylzero(int nvec, sunrealtype* c, N_Vector* X, N_Vector z);
SUNErrCode N_VScaleAddMulti_Gkylzero(int nvec, sunrealtype* a, N_Vector x, N_Vector* Y, N_Vector* Z);
SUNErrCode N_VDotProdMulti_Gkylzero(int nvec, N_Vector x, N_Vector* Y, sunrealtype* dotprods);
sunrealtype N_VDotProdLocal_Gkylzero(N_Vector x, N_Vector y);
sunrealtype N_VMaxNormLocal_Gkylzero(N_Vector x);
sunrealtype N_VMinLocal_Gkylzero(N_Vector x);
sunrealtype N_VL1NormLocal_Gkylzero(N_Vector x);
sunrealtype N_VWSqrSumLocal_Gkylzero(N_Vector x, N_Vector w);
sunrealtype N_VWSqrSumMaskLocal_Gkylzero(N_Vector x, N_Vector w, N_Vector id);
sunbooleantype N_VInvTestLocal_Gkylzero(N_Vector x, N_Vector z);
SUNErrCode N_VDotProdMultiLocal_Gkylzero(int nvec, N_Vector x, N_Vector* Y, sunrealtype* dotprods);
SUNErrCode N_VDotProdMultiAllReduce_Gkylzero(int nvec, N_Vector x, sunrealtype* sum);

#ifdef __cplusplus
}
#endif

#endif