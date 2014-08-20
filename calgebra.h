// calgebra.h
//
// https://github.com/tylerneylon/calgebra
//
// A C library to solve some linear algebraic problems.
//

// TODO Add a way to get degeneracy or instability warnings.

#pragma once

// The element in the ith row and jth column (0-indexed), denoted
// m_ij, is at data[i * ncols + j]; this is row-major order.
typedef struct {
  float *data;
  int nrows, ncols;
  int is_transposed;
} alg__MatStruct, *alg__Mat;

typedef enum {
  alg__status_ok,
  alg__status_no_soln,
  alg__status_unbdd_soln,
  alg__status_input_error
} alg__Status;

// 1. Create. copy, destroy, or print a matrix.

alg__Mat alg__alloc_matrix  (int nrows, int ncols);
alg__Mat alg__copy_matrix   (alg__Mat orig);
void     alg__free_matrix   (alg__Mat M);

      // Returns a newly-allocated string; caller must free it.
char *   alg__matrix_as_str (alg__Mat M);

// 2. Basic matrix operations.
      // In the comments, we use A[i] to mean the ith column of A.

 
      // Use alg__elt(A, i, j) as either a value or variable (r- or l-value).
#define  alg__elt(A, i, j) \
  (A->data[(i) * (A->is_transposed ? 1 : A->ncols) + \
           (j) * (A->is_transposed ? A->ncols : 1)])

      // Returns < A[i], B[j] >.
float    alg__dot_prod             (alg__Mat A, int i, alg__Mat B, int j);

      // B[j] += c * A[i].
void     alg__mul_and_add (float c, alg__Mat A, int i, alg__Mat B, int j);

      // A[i] *= c.
void     alg__scale       (float c, alg__Mat A, int i);

      // Returns ||A[i]||_2.
float    alg__norm                 (alg__Mat A, int i);


// 3. Decompositions.

      // Provide a reduced QR decomposition; R may be NULL to ignore it.
void     alg__QR (alg__Mat A_to_Q, alg__Mat R);

// 4. Optimizations.

// The next two functions solve this problem for p=1 or p=2:
//    Find x which gives    min ||x||_p
//    with the constraint   Ax = b.
// The output x should be pre-allocated, including memory for the data.

alg__Status alg__l1_min (alg__Mat A, alg__Mat b, alg__Mat x);
alg__Status alg__l2_min (alg__Mat A, alg__Mat b, alg__Mat x);

// Solve a general linear programming problem.
// Specifically, find x that minimizes (c^T * x) with Ax=b, x >= 0.
// The l1_min function above is a wrapper around this.
alg__Status alg__run_lp (alg__Mat A, alg__Mat b, alg__Mat x, alg__Mat c);
