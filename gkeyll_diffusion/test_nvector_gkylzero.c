#include "src/nvector_gkylzero.h"
#include <gkyl_null_comm.h>

void test_NVector(bool use_gpu)
{
  int num_of_failures = 0;
  long int num_basis  = 100;
  long int size       = 100;
  double eq_check_tol = 1e-10;

  struct gkyl_range local;
  int lower[] = {1}, upper[] = {size};
  gkyl_range_init(&local, 1, lower, upper);

  // Construct communicator for use in app.
  struct gkyl_comm *comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = use_gpu
    }
  );

  printf("\nTESTING nvector_gkylzero:\n");

  /* Create the SUNDIALS context object for this simulation */
  SUNContext sunctx;
  SUNContext_Create(SUN_COMM_NULL, &sunctx);

  struct gkyl_array* testarray;

  testarray = mkarr(use_gpu, num_basis, local.volume);

  double* ta_data = testarray->data;

  for (unsigned int i = 0; i < (testarray->size * testarray->ncomp); ++i)
  {
    ta_data[i] = (double)i;
  }

  N_Vector NV_test = N_VMake_Gkylzero(testarray, use_gpu, comm, &local, sunctx);

  struct gkyl_array* testarrayreturn;

  testarrayreturn = N_VGetVector_Gkylzero(NV_test);

  double* tar_data = testarrayreturn->data;

  bool failure = ((testarray->size != testarrayreturn->size) ||
                  (testarray->ncomp != testarrayreturn->ncomp));

  for (unsigned int i = 0; i < (testarrayreturn->size * testarrayreturn->ncomp);
       ++i)
  {
    failure = (fabs(tar_data[i] - ta_data[i]) > eq_check_tol) || failure;
  }

  if (failure)
  {
    printf("\n      FAILED in N_VMake_Gkylzero or N_VGetVector_Gkylzero");
    num_of_failures++;
  }
  else
  {
    printf(
      "\n      N_VMake_Gkylzero and N_VGetVector_Gkylzero PASSED the test");
  }

  /* ----------------------------------------------------------------------
  * N_VCloneEmpty_Gkylzero Test
  * --------------------------------------------------------------------*/

  N_Vector NV_test_clone = N_VClone_Gkylzero(NV_test);

  struct gkyl_array* testarrayclone;

  testarrayclone = N_VGetVector_Gkylzero(NV_test_clone);

  double* tac_data = testarrayclone->data;

  failure = ((testarray->size != testarrayclone->size) ||
             (testarray->ncomp != testarrayclone->ncomp) ||
             (tac_data == ta_data));

  if (failure)
  {
    printf("\n      FAILED in N_VCloneEmpty_Gkylzero or N_VClone_Gkylzero");
    num_of_failures++;
  }
  else
  {
    printf(
      "\n      N_VCloneEmpty_Gkylzero and N_VClone_Gkylzero PASSED the test");
  }

  /* ----------------------------------------------------------------------
  * N_VScale_Gkylzero Test
  * --------------------------------------------------------------------*/

  N_Vector NV_test_return = N_VClone_Gkylzero(NV_test);

  N_VScale_Gkylzero(2.0, NV_test, NV_test_return);

  testarrayreturn = N_VGetVector_Gkylzero(NV_test_return);

  double* tas_data = testarrayreturn->data;

  failure = ((testarray->size != testarrayreturn->size) ||
             (testarray->ncomp != testarrayreturn->ncomp));

  for (unsigned int i = 0; i < (testarray->size * testarray->ncomp); ++i)
  {
    failure = (fabs(tas_data[i] - 2.0 * ta_data[i]) > eq_check_tol) || failure;
  }

  if (failure)
  {
    printf("\n      FAILED in N_VScale_Gkylzero");
    num_of_failures++;
  }
  else { printf("\n      N_VScale_Gkylzero PASSED the test"); }

  /* ----------------------------------------------------------------------
  * N_VConst_Gkylzero Test
  * --------------------------------------------------------------------*/

  N_VConst_Gkylzero(173.0, NV_test_return);

  testarrayreturn = N_VGetVector_Gkylzero(NV_test_return);

  double* tacon_data = testarrayreturn->data;

  failure = false;
  for (unsigned int i = 0; i < (testarray->size * testarray->ncomp); ++i)
  {
    failure = (fabs(tacon_data[i] - 173.0) > eq_check_tol) || failure;
  }

  if (failure)
  {
    printf("\n            FAILED in N_VConst_Gkylzero");
    num_of_failures++;
  }
  else { printf("\n      N_VConst_Gkylzero PASSED the test"); }

  /* ----------------------------------------------------------------------
  * N_VLinearSum_Gkylzero Test
  * --------------------------------------------------------------------*/

  double a = 2.0;
  double b = 3.0;
  double c = 1.75;
  double d = 2.89;

  struct gkyl_array* v1      = mkarr(use_gpu, num_basis, local.volume);
  struct gkyl_array* v2      = mkarr(use_gpu, num_basis, local.volume);
  struct gkyl_array* lin_sum = mkarr(use_gpu, num_basis, local.volume);

  N_Vector Nv1      = N_VMake_Gkylzero(v1     , use_gpu, comm, &local, sunctx);
  N_Vector Nv2      = N_VMake_Gkylzero(v2     , use_gpu, comm, &local, sunctx);
  N_Vector Nlin_sum = N_VMake_Gkylzero(lin_sum, use_gpu, comm, &local, sunctx);

  N_VConst_Gkylzero(c, Nv1);
  N_VConst_Gkylzero(d, Nv2);

  N_VLinearSum_Gkylzero(a, Nv1, b, Nv2, Nlin_sum);

  lin_sum = N_VGetVector_Gkylzero(Nlin_sum);

  double* lin_sum_data = lin_sum->data;

  failure = false;
  for (unsigned int i = 0; i < (testarrayreturn->size * testarrayreturn->ncomp);
       ++i)
  {
    failure = (fabs(lin_sum_data[i] - (a * c + b * d)) > eq_check_tol) || failure;
  }

  if (failure)
  {
    printf("\n      FAILED in N_VLinearSum_Gkylzero");
    num_of_failures++;
  }
  else { printf("\n      N_VLinearSum_Gkylzero PASSED the test"); }

  /* ----------------------------------------------------------------------
  * N_VWrmsNorm_abs_comp_Gkylzero Test
  * --------------------------------------------------------------------*/

  N_VConst_Gkylzero(-0.5, Nv1);
  N_VConst_Gkylzero(0.5, Nv2);

  double wrmsnorm = N_VWrmsNorm_abs_comp_Gkylzero(Nv1, Nv2);

  /* ans should equal 1/4 */
  failure = (wrmsnorm < 0.0) ? 1 : (fabs(wrmsnorm - 1.0 / 4.0) > eq_check_tol);

  if (failure)
  {
    printf("\n      FAILED in N_VWrmsNorm_abs_comp_Gkylzero");
    num_of_failures++;
  }
  else { printf("\n      N_VWrmsNorm_abs_comp_Gkylzero PASSED the test"); }

  /* ----------------------------------------------------------------------
  * N_VDiv_Gkylzero Test
  * --------------------------------------------------------------------*/

  struct gkyl_array* nvdiv = mkarr(use_gpu, num_basis, local.volume);

  N_Vector Nvdiv = N_VMake_Gkylzero(nvdiv, use_gpu, comm, &local, sunctx);

  N_VConst_Gkylzero(c, Nv1);
  N_VConst_Gkylzero(d, Nv2);

  N_VDiv_Gkylzero(Nv1, Nv2, Nvdiv);

  nvdiv = N_VGetVector_Gkylzero(Nvdiv);

  double* nvdiv_data = nvdiv->data;

  failure = false;
  for (unsigned int i = 0; i < (nvdiv->size * nvdiv->ncomp); ++i)
  {
    failure = (fabs(nvdiv_data[i] - c / d) > eq_check_tol) || failure;
  }

  if (failure)
  {
    printf("\n      FAILED in N_VDiv_Gkylzero");
    num_of_failures++;
  }
  else { printf("\n      N_VDiv_Gkylzero PASSED the test"); }

  /* ----------------------------------------------------------------------
  * N_VAbs_Gkylzero Test
  * --------------------------------------------------------------------*/

  struct gkyl_array* nvabs = mkarr(use_gpu, num_basis, local.volume);

  N_Vector Nvabs = N_VMake_Gkylzero(nvabs, use_gpu, comm, &local, sunctx);

  N_VConst_Gkylzero(-1.0, Nv1);

  N_VAbs_Gkylzero(Nv1, Nvabs);

  nvabs = N_VGetVector_Gkylzero(Nvabs);

  double* nvabs_data = nvabs->data;

  failure = false;
  for (unsigned int i = 0; i < (nvdiv->size * nvdiv->ncomp); ++i)
  {
    failure = (fabs(nvabs_data[i] - 1.0) > eq_check_tol) || failure;
  }

  if (failure)
  {
    printf("\n      FAILED in N_VAbs_Gkylzero");
    num_of_failures++;
  }
  else { printf("\n      N_VAbs_Gkylzero PASSED the test"); }

  /* ----------------------------------------------------------------------
  * N_VInv_Gkylzero Test
  * --------------------------------------------------------------------*/

  struct gkyl_array* nvinv = mkarr(use_gpu, num_basis, local.volume);

  N_Vector Nvinv = N_VMake_Gkylzero(nvinv, use_gpu, comm, &local, sunctx);

  N_VConst_Gkylzero(c, Nv1);

  N_VInv_Gkylzero(Nv1, Nvinv);

  nvinv = N_VGetVector_Gkylzero(Nvinv);

  double* nvinv_data = nvinv->data;

  failure = false;
  for (unsigned int i = 0; i < (nvdiv->size * nvdiv->ncomp); ++i)
  {
    failure = (fabs(nvinv_data[i] - 1.0 / c) > eq_check_tol) || failure;
  }

  if (failure)
  {
    printf("\n      FAILED in N_VInv_Gkylzero");
    num_of_failures++;
  }
  else { printf("\n      N_VInv_Gkylzero PASSED the test"); }

  /* ----------------------------------------------------------------------
  * N_VMaxnorm_Gkylzero Test
  * --------------------------------------------------------------------*/

  N_VConst_Gkylzero(0.0, Nv1);

  sunrealtype nvmaxnorm = N_VMaxnorm_Gkylzero(Nv1);

  if (nvmaxnorm < 0.0 || nvmaxnorm >= eq_check_tol) { failure = 1; }

  if (failure)
  {
    printf("\n      FAILED in N_VMaxnorm_Gkylzero");
    num_of_failures++;
  }
  else { printf("\n      N_VMaxnorm_Gkylzero PASSED the test"); }

  /* ----------------------------------------------------------------------
  * N_VAddconst_Gkylzero Test
  * --------------------------------------------------------------------*/

  struct gkyl_array* nvadd = mkarr(use_gpu, num_basis, local.volume);

  N_Vector Nvadd = N_VMake_Gkylzero(nvadd, use_gpu, comm, &local, sunctx);

  N_VConst_Gkylzero(c, Nv1);

  N_VAddconst_Gkylzero(Nv1, d, Nvadd);

  nvadd = N_VGetVector_Gkylzero(Nvadd);

  double* nvadd_data = nvadd->data;

  failure = false;
  for (unsigned int i = 0; i < (nvdiv->size * nvdiv->ncomp); ++i)
  {
    failure = (fabs(nvadd_data[i] - (c + d)) > eq_check_tol) || failure;
  }

  if (failure)
  {
    printf("\n      FAILED in N_VAddconst_Gkylzero");
    num_of_failures++;
  }
  else { printf("\n      N_VAddconst_Gkylzero PASSED the test"); }

  if (num_of_failures == 0)
  {
    printf("\n\n nvector_gkylzero PASSED all tests!\n\n");
  }
  else
  {
    printf("\n\n nvector_gkylzero failed in %d test(s)!\n\n", num_of_failures);
  }

  N_VDestroy_Gkylzero(NV_test);
  N_VDestroy_Gkylzero(NV_test_clone);
  N_VDestroy_Gkylzero(NV_test_return);
  N_VDestroy_Gkylzero(Nv1);
  N_VDestroy_Gkylzero(Nv2);
  N_VDestroy_Gkylzero(Nlin_sum);
  N_VDestroy_Gkylzero(Nvdiv);
  N_VDestroy_Gkylzero(Nvabs);
  N_VDestroy_Gkylzero(Nvinv);
  N_VDestroy_Gkylzero(Nvadd);

  gkyl_array_release(testarray);
  gkyl_array_release(v1);
  gkyl_array_release(v2);
  gkyl_array_release(lin_sum);
  gkyl_array_release(nvdiv);
  gkyl_array_release(nvabs);
  gkyl_array_release(nvinv);
  gkyl_array_release(nvadd);
  gkyl_comm_release(comm);
}

int main(int argc, char** argv)
{
  test_NVector(false);

  return 0;
}
