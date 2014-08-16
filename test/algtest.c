// algtest.c
//
// https://github.com/tylerneylon/calgebra
//

#include "calgebra.h"
#include "test/ctest.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

int test_basic_ops() {
  // We start with
  //  A = ( 1 )    B = ( -2 )
  //      ( 3 )        (  0 )
  alg__Mat A = alg__alloc_matrix(2, 1);
  // TODO Enable a better initialization construct.
  memcpy(A->data, ((float[]){ 1, 3 }), 2 * sizeof(float));

  alg__Mat B = alg__alloc_matrix(2, 1);
  memcpy(B->data, ((float[]){ -2, 0}), 2 * sizeof(float));

  // Test alg__elt.
  test_that(alg__elt(A, 0, 0) == 1);
  test_that(alg__elt(B, 1, 0) == 0);

  // Test alg__dot_prod.
  test_that(alg__dot_prod(A, 0, B, 0) == -2);

  // Test alg__norm.
  // This uses that sqrt(10) is in (3.162, 3.163).
  test_that(3.162 <= alg__norm(A, 0) &&
                     alg__norm(A, 0) <= 3.163);
  test_that(alg__norm(B, 0) == 2);

  // Test alg__mul_and_add.
  alg__mul_and_add(2, A, 0, B, 0);
  test_that(alg__elt(B, 0, 0) == 0);
  test_that(alg__elt(B, 1, 0) == 6);

  // Test alg__scale.
  alg__scale(0.5, B, 0);
  test_that(alg__elt(B, 1, 0) == 3);

  // Test some of the above with tranposed matrices.
  // As a reminder, we now have B = ( 0 )
  //                                ( 3 ).
  // After the transpose, we'll have
  //   A = ( 1 3 )   B = ( 0 3 ).
  A->is_transposed = 1;
  B->is_transposed = 1;

  test_that(alg__elt(A, 0, 1) == 3);
  test_that(alg__elt(B, 0, 1) == 3);

  test_that(alg__dot_prod(A, 1, B, 1) == 9);

  test_that(alg__norm(A, 1) == 3);

  alg__mul_and_add(-2, A, 0, B, 1);
  test_that(alg__elt(B, 0, 1) == 1);

  alg__free_matrix(A);
  alg__free_matrix(B);

  return test_success;
}

// TODO Also test R.

int test_QR() {
  alg__Mat A = alg__alloc_matrix(2, 2);
  memcpy(A->data, ((float[])
         { -5,  2,
            0,  6  }), 4 * sizeof(float));
  alg__QR(A, NULL);

  test_that(fabs(alg__elt(A, 0, 0)) == 1);
  test_that(     alg__elt(A, 1, 0)  == 0);
  test_that(fabs(alg__elt(A, 1, 1)) == 1);

  memcpy(A->data, ((float[])
         {  1,  9,
            1,  7  }), 4 * sizeof(float));
  alg__QR(A, NULL);

  float x = alg__elt(A, 0, 0);
  test_that(fabs(x - 1.0 / sqrtf(2))        < 0.001);
  test_that(fabs(alg__dot_prod(A, 0, A, 1)) < 0.001);
  test_that(fabs(alg__norm(A, 0) - 1)       < 0.001);
  test_that(fabs(alg__norm(A, 1) - 1)       < 0.001);

  alg__free_matrix(A);

  return test_success;
}

int main(int argc, char **argv) {
  set_verbose(0);  // Set this to 1 while debugging a test.
  start_all_tests(argv[0]);
  run_tests(test_basic_ops, test_QR);
  return end_all_tests();
}
