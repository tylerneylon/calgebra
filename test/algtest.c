// algtest.c
//
// https://github.com/tylerneylon/calgebra
//

#include "calgebra.h"
#include "test/ctest.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int test_basic_ops() {
  // We start with
  //  A = ( 1 )    B = ( -2 )
  //      ( 3 )        (  0 )
  alg__Mat A = alg__alloc_matrix(2, 1);
  alg__set_matrix(A, 1, 3);

  alg__Mat B = alg__alloc_matrix(2, 1);
  alg__set_matrix(B, -2, 0);

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
  alg__set_matrix(A, -5,  2,
                      0,  6 );
  alg__QR(A, NULL);

  test_that(fabs(alg__elt(A, 0, 0)) == 1);
  test_that(     alg__elt(A, 1, 0)  == 0);
  test_that(fabs(alg__elt(A, 1, 1)) == 1);

  alg__set_matrix(A,  1,  9,
                      1,  7 );
  alg__QR(A, NULL);

  float x = fabs(alg__elt(A, 0, 0));
  test_that(fabs(x - 1.0 / sqrtf(2))        < 0.001);
  test_that(fabs(alg__dot_prod(A, 0, A, 1)) < 0.001);
  test_that(fabs(alg__norm(A, 0) - 1)       < 0.001);
  test_that(fabs(alg__norm(A, 1) - 1)       < 0.001);

  alg__set_matrix(A,  3, -6,
                      4, 17 );
  alg__Mat R = alg__alloc_matrix(2, 2);
  alg__QR(A, R);

  // These test magnitude only since the column signs
  // are not specified by the decomposition.
  test_that(fabs(fabs(alg__elt(A, 0, 0)) - 0.6) < 0.001);
  test_that(fabs(fabs(alg__elt(A, 1, 0)) - 0.8) < 0.001);
  test_that(fabs(fabs(alg__elt(A, 0, 1)) - 0.8) < 0.001);
  test_that(fabs(fabs(alg__elt(A, 1, 1)) - 0.6) < 0.001);

  test_that(fabs(fabs(alg__elt(R, 0, 0)) -  5) < 0.001);
  test_that(fabs(fabs(alg__elt(R, 1, 0)) -  0) < 0.001);
  test_that(fabs(fabs(alg__elt(R, 0, 1)) - 10) < 0.001);
  test_that(fabs(fabs(alg__elt(R, 1, 1)) - 15) < 0.001);

  test_that(fabs(alg__dot_prod(A, 0, A, 1)) < 0.001);
  test_that(fabs(alg__norm(A, 0) - 1)       < 0.001);
  test_that(fabs(alg__norm(A, 1) - 1)       < 0.001);

  alg__free_matrix(R);
  alg__free_matrix(A);

  return test_success;
}

int test_lp_pt1() {
  alg__Mat A = alg__alloc_matrix(1, 2);
  alg__set_matrix(A, 1, 5);

  alg__Mat b = alg__alloc_matrix(1, 1);
  alg__set_matrix(b, 5);

  alg__Mat c = alg__alloc_matrix(2, 1);
  alg__set_matrix(c, 1, 1);

  alg__Mat x = alg__alloc_matrix(2, 1);

  alg__run_lp(A, b, x, c);

  // We expect the answer x = (0 1)^T.

  test_that(fabs(alg__elt(x, 0, 0) - 0) < 0.001);
  test_that(fabs(alg__elt(x, 1, 0) - 1) < 0.001);

  return test_success;
}

int test_lp_pt2() {
  alg__Mat A = alg__alloc_matrix(3, 5);
  alg__set_matrix(A,  1,  0,  0,  0,  1,
                      0,  1,  0,  4, -5,
                      0,  0,  1, -4,  1 );

  alg__Mat b = alg__alloc_matrix(3, 1);
  alg__set_matrix(b, 7, -7, -5);

  alg__Mat c = alg__alloc_matrix(5, 1);
  alg__set_matrix(c, 0, 0, 0, 3, 2);

  alg__Mat x = alg__alloc_matrix(5, 1);

  alg__run_lp(A, b, x, c);

  // We expect the answer x = (4 0 0 2 3)^T.

  float ans[] = { 4, 0, 0, 2, 3 };

  for (int i = 0; i < 5; ++i) {
    test_that(fabs(alg__elt(x, i, 0) - ans[i]) < 0.001);
  }

  return test_success;
}

int test_l2_min() {
  // Set a matrix with rows orthogonal to (1 -1 -1).
  alg__Mat A = alg__alloc_matrix(2, 3);
  alg__set_matrix(A,  5,  2,  3,
                      1,  4, -3 );
  alg__Mat b = alg__alloc_matrix(2, 1);
  alg__set_matrix(b, 7, 5);
  alg__Mat x = alg__alloc_matrix(3, 1);

  alg__l2_min(A, b, x);

  // We expect the answer x = (1 1 0)^T.

  test_that(fabs(alg__elt(x, 0, 0) - 1) < 0.001);
  test_that(fabs(alg__elt(x, 1, 0) - 1) < 0.001);
  test_that(fabs(alg__elt(x, 2, 0) - 0) < 0.001);

  return test_success;
}

int test_l2_error_cases() {
  alg__Mat A = alg__alloc_matrix(1, 1);
  alg__Mat b = alg__alloc_matrix(2, 1);
  alg__Mat x = alg__alloc_matrix(2, 1);

  alg__Status status;

  status = alg__l2_min(A, b, x);
  test_that(status == alg__status_input_error);

  alg__free_matrix(A);
  A = alg__alloc_matrix(2, 2);

  status = alg__l2_min(A, b, NULL);
  test_that(status == alg__status_input_error);

  status = alg__l2_min(A, b, x);
  test_that(status == alg__status_ok);

  return test_success;
}

int test_no_soln_cases() {
  // We'll set up Ax=b to have so solutions.
  // Specifically, A=0 and b=1.

  alg__Mat A = alg__alloc_matrix(1, 1);
  *A->data = 0;
  alg__Mat b = alg__alloc_matrix(1, 1);
  *b->data = 1;
  alg__Mat x = alg__alloc_matrix(1, 1);

  alg__Status status;

  status = alg__l1_min(A, b, x);
  test_that(status == alg__status_no_soln);

  status = alg__l2_min(A, b, x);
  test_that(status == alg__status_no_soln);

  alg__Mat c = alg__alloc_matrix(1, 1);
  status = alg__run_lp(A, b, x, c);
  test_that(status == alg__status_no_soln);

  alg__free_matrix(c);
  alg__free_matrix(x);
  alg__free_matrix(b);
  alg__free_matrix(A);

  return test_success;
}

int test_lp_errors() {
  // The no solution case is tested in
  // test_no_soln_cases.

  // Test a problem instance with an unbounded
  // solution set.
  // Ax=b, x>=0, minimize -(x_1 + x_2). A=(1 0) b=1.
  // Then x=(1 y)^T is a solution for any y,
  // and the minimized value gets smallyer as y increases.

  alg__Mat A = alg__alloc_matrix(1, 2);
  A->data[0] = 1;
  A->data[1] = 0;

  alg__Mat b = alg__alloc_matrix(1, 1);
  *b->data = 1;

  alg__Mat c = alg__alloc_matrix(2, 1);
  c->data[0] = -1;
  c->data[1] = -1;

  alg__Mat x = alg__alloc_matrix(2, 1);

  alg__Status status = alg__run_lp(A, b, x, c);

  test_that(status == alg__status_unbdd_soln);

  alg__free_matrix(x);
  alg__free_matrix(c);
  alg__free_matrix(b);
  alg__free_matrix(A);

  return test_success;
}

int test_l1_min() {
  // The conceptual problem has the feasible solution set
  // (1 1 0) + t(-1 -1 8), specified with
  // A = (4 4 1)  b = (8)
  //     (8 0 1)      (8)
  // For this set, x = (1 1 0)^T minimizes ||x||_1.
  alg__Mat A = alg__alloc_matrix(2, 3);
  alg__set_matrix(A,  4,  4,  1,
                      8,  0,  1 );
  alg__Mat b = alg__alloc_matrix(2, 1);
  alg__set_matrix(b, 8, 8);
  alg__Mat x = alg__alloc_matrix(3, 1);

  alg__Status status = alg__l1_min(A, b, x);

  test_that(status == alg__status_ok);

  test_that(fabs(alg__elt(x, 0, 0) - 1) < 0.001);
  test_that(fabs(alg__elt(x, 1, 0) - 1) < 0.001);
  test_that(fabs(alg__elt(x, 2, 0) - 0) < 0.001);

  return test_success;
}

int test_linf_min() {
  // The conceptual problem has the feasible solution set
  // (-1 1) + t(2 1), specified with
  // A = (1 -2)  b = (-3)
  // For this set, x = (-1 1)^T minimizes ||x||_inf.

  alg__Mat A = alg__alloc_matrix(1, 2);
  alg__set_matrix(A, 1, -2);

  alg__Mat b = alg__alloc_matrix(1, 1);
  alg__set_matrix(b, -3);

  alg__Mat x = alg__alloc_matrix(2, 1);

  alg__Status status = alg__linf_min(A, b, x);

  test_that(status == alg__status_ok);
  
  test_that(fabs(alg__elt(x, 0, 0) - -1) < 0.001);
  test_that(fabs(alg__elt(x, 1, 0) -  0) < 0.001);

  return test_success;
}

int main(int argc, char **argv) {
  set_verbose(0);  // Set this to 1 while debugging a test.
  start_all_tests(argv[0]);
  run_tests(test_basic_ops, test_QR,
            test_lp_pt1, test_lp_pt2, test_l2_min,
            test_l2_error_cases, test_no_soln_cases,
            test_lp_errors, test_l1_min,
            test_linf_min);
  return end_all_tests();
}
