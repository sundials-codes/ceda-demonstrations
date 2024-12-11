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
#include <sundials/sundials_context.h>
#include <sundials/sundials_types.h> /* definition of type sunrealtype          */

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
  sunbooleantype own_vector;  /* ownership Gkylzero vector */
  struct gkyl_array* dataptr; /* the actual Gkylzero object pointer */
  /* ADD ANY GKEYLLZERO OBJECTS HERE THAT WILL BE NEEDED, E.G., MPI COMMUNICATOR */
};

typedef struct _N_VectorContent_Gkylzero* N_VectorContent_Gkylzero;

/* -----------------------------------------------------------------
 * Functions exported by nvector_gkylzero
 * -----------------------------------------------------------------*/
struct gkyl_array*
mkarr(bool on_gpu, long nc, long size);

N_Vector N_VNewEmpty_Gkylzero(SUNContext sunctx);
N_Vector N_VMake_Gkylzero(struct gkyl_array* x, SUNContext sunctx);
struct gkyl_array* N_VGetVector_Gkylzero(N_Vector v);

N_Vector N_VCloneEmpty_Gkylzero(N_Vector w);
N_Vector N_VClone_Gkylzero(N_Vector w, sunbooleantype use_gpu);
void N_VDestroy_Gkylzero(N_Vector v);

/* vector operations -- DELETE ALL THAT WILL BE UNUSED BY LSRKSTEP */
void N_VLinearSum_Gkylzero(sunrealtype a, N_Vector x, sunrealtype b, N_Vector y, N_Vector z);
void N_VConst_Gkylzero(sunrealtype c, N_Vector z);
void N_VScale_Gkylzero(sunrealtype c, N_Vector x, N_Vector z);
sunrealtype N_VWrmsNorm_Gkylzero(N_Vector x, N_Vector w);

#ifdef __cplusplus
}
#endif

#endif
