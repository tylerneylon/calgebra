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

// 1. Matrix allocation and copying.

alg__Mat alg__alloc_matrix (int nrows, int ncols);
alg__Mat alg__copy_matrix  (alg__Mat orig);
void     alg__free_matrix  (alg__Mat m);

// 2. Basic matrix operations.
//    In the comments, we use A[i] to mean the ith column of A.

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
void     alg__QR(alg__Mat A_to_Q, alg__Mat R);

// 4. Optimizations.

// Solve the problem:
//    Find x which gives    min ||x||_2
//    with the constraint   Ax = b.
// The output x should be pre-allocated, including memory for the data.
void alg__l2_min(alg__Mat A, alg__Mat b, alg__Mat x);
