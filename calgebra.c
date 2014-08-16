// calgebra.c
//
// https://github.com/tylerneylon/calgebra
//

#include "calgebra.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define true  1
#define false 0

// Internal functions.

#define data_size(nrows, ncols) (sizeof(float) * nrows * ncols)
#define num_cols(A) (A->is_transposed ? A->nrows : A->ncols)
#define num_rows(A) (A->is_transposed ? A->ncols : A->nrows)
#define elt(A, i, j) \
  A->data[i * (A->is_transposed ? 1 : A->ncols) + \
          j * (A->is_transposed ? A->ncols : 1)]
#define col_elt(A, i) elt(A, i, 0)

// Public functions.

// 1. Matrix allocation and copying.

alg__Mat alg__alloc_matrix(int nrows, int ncols) {
  alg__Mat M = malloc(sizeof(alg__MatStruct));
  M->data    = malloc(data_size(nrows, ncols));
  return M;
}

alg__Mat alg__copy_matrix(alg__Mat orig) {
  alg__Mat M = alg__alloc_matrix(orig->nrows, orig->ncols);
  memcpy(M->data, orig->data, data_size(M->nrows, M->ncols));
  return M;
}

void alg__free_matrix(alg__Mat M) {
  free(M->data);
  free(M);
}

// 2. Basic matrix operations.

float alg__dot_prod(alg__Mat A, int i, alg__Mat B, int j) {
  if (num_rows(A) != num_rows(B)) {
    fprintf(stderr, "Error: expected A,B to have equal num_rows.\n");
    return 0.0;
  }
  float sum = 0;
  for (int k = 0; k < num_rows(A); ++k) {
    sum += elt(A, k, i) * elt(B, k, j);
  }
  return sum;
}

void alg__mul_and_add(float c, alg__Mat A, int i, alg__Mat B, int j) {
  for (int k = 0; k < num_rows(A); ++k) {
    elt(B, k, j) += c * elt(A, k, i);
  }
}

void alg__scale(float c, alg__Mat A, int i) {
  for (int k = 0; k < num_rows(A); ++k) {
    elt(A, k, i) *= c;
  }
}

float alg__norm(alg__Mat A, int i) {
  float norm_squared = alg__dot_prod(A, i, A, i);
  return sqrtf(norm_squared);
}

// 3. Decompositions.

void alg__QR(alg__Mat Q, alg__Mat R) {
  if (num_rows(Q) < num_cols(Q)) {
    fprintf(stderr, "Error: expected a tall or square matrix for QR decomp.\n");
    return;
  }
  // Q starts off as an arbitrary tall-or-square matrix A.
  for (int i = 0; i < num_cols(Q); ++i) {
    float norm = alg__norm(Q, i);
    alg__scale(1.0 / norm, Q, i);
    for (int j = i + 1; j < num_cols(Q); ++j) {
      float dot_prod = alg__dot_prod(Q, i, Q, j);
      alg__mul_and_add(-dot_prod, Q, i, Q, j);
    }
  }
  // TODO Save the R values. This is not needed for L2-min, though.
}

// 4. Optimizations.

void alg__l2_min(alg__Mat A, alg__Mat b, alg__Mat x) {
  if (num_cols(A) != num_cols(b)) {
    fprintf(stderr, "Error: A and b must have the same number of columns.\n");
    return;
  }
  if (x == NULL) {
    fprintf(stderr, "Error: expected output matrix x to be pre-allocated.\n");
    return;
  }
  alg__Mat Q = alg__copy_matrix(A);

  // We want to work with rows of A (and Q).
  A->is_transposed = !A->is_transposed;

  // We want to orthonormalize the rows of Q.
  Q->is_transposed = true;
  alg__QR(Q, NULL);

  for (int i = 0; i < A->nrows; ++i) {
    float a_i_q_i = alg__dot_prod(A, i, Q, i);
    float a_i_x   = alg__dot_prod(A, i, x, 0);
    float alpha = (col_elt(b, i) - a_i_x) / a_i_q_i;
    alg__mul_and_add(alpha, Q, i, x, 0);
  }

  // Clean up.
  A->is_transposed = !A->is_transposed;
  alg__free_matrix(Q);
}

