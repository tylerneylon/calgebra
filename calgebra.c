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
#define elt(A, i, j) alg__elt(A, i, j)
#define col_elt(A, i) elt(A, i, 0)


// Public functions.

// 1. Create. copy, destroy, or print a matrix.

alg__Mat alg__alloc_matrix(int nrows, int ncols) {
  alg__Mat M = malloc(sizeof(alg__MatStruct));
  *M = (alg__MatStruct) {
         .data  = malloc(data_size(nrows, ncols)),
         .nrows = nrows,
         .ncols = ncols,
         .is_transposed = false };
  return M;
}

alg__Mat alg__copy_matrix(alg__Mat orig) {
  alg__Mat M = alg__alloc_matrix(orig->nrows, orig->ncols);
  M->is_transposed = orig->is_transposed;
  memcpy(M->data, orig->data, data_size(M->nrows, M->ncols));
  return M;
}

void alg__free_matrix(alg__Mat M) {
  free(M->data);
  free(M);
}

char *alg__matrix_as_str(alg__Mat M) {
  char **row_strs = alloca(num_rows(M) * sizeof(char *));
  size_t sum_bytes = 1;  // Start at 1 for the null terminator.
  for (int row = 0; row < num_rows(M); ++row) {
    size_t buf_len = 16 * (num_cols(M) + 1);
    char *s = row_strs[row] = alloca(buf_len);
    char *s_end = s + buf_len;
    s += snprintf(s, s_end - s, "( ");
    for (int col = 0; col < num_cols(M); ++col) {
      s += snprintf(s, s_end - s, "%5.2g ", elt(M, row, col));
    }
    s += snprintf(s, s_end - s, ")\n");
    sum_bytes += (s - row_strs[row]);
  }
  char *sum_s = malloc(sum_bytes);
  char *s     = sum_s;
  for (int r = 0; r < num_rows(M); ++r) s = stpcpy(s, row_strs[r]);
  return sum_s;
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
  if (num_rows(A) != num_rows(b)) {
    fprintf(stderr, "Error: A and b must have the same number of rows.\n");
    return;
  }
  if (num_cols(A) != num_rows(x) || num_cols(x) != 1) {
    fprintf(stderr, "Error: expected x to have size #cols(A) x 1.\n");
    return;
  }
  if (x == NULL) {
    fprintf(stderr, "Error: expected output matrix x to be pre-allocated.\n");
    return;
  }

  // We want to work with rows of A (and Q).
  A->is_transposed = !A->is_transposed;
  alg__Mat Q = alg__copy_matrix(A);
  alg__QR(Q, NULL);

  for (int i = 0; i < num_cols(A); ++i) {
    float a_i_q_i = alg__dot_prod(A, i, Q, i);
    float a_i_x   = alg__dot_prod(A, i, x, 0);
    float alpha = (col_elt(b, i) - a_i_x) / a_i_q_i;
    alg__mul_and_add(alpha, Q, i, x, 0);
  }

  // Clean up.
  A->is_transposed = !A->is_transposed;
  alg__free_matrix(Q);
}

