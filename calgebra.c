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

#define tol 1e-6


// Internal types.

typedef enum {
  phase1,
  phase2
} Phase;


// Internal functions.

#define data_size(nrows, ncols) (sizeof(float) * nrows * ncols)
#define num_cols(A) (A->is_transposed ? A->nrows : A->ncols)
#define num_rows(A) (A->is_transposed ? A->ncols : A->nrows)
#define elt(A, i, j) alg__elt(A, i, j)
#define col_elt(A, i) elt(A, i, 0)

// This sets the given entry to 1 by scaling its row
// and clears the rest of its column with row operations.
// The given entry is expected to be nonzero.
void make_col_a_01_col(alg__Mat A, int row, int col) {
  // Normalize the row so that A_{row,col} = 1.
  float scale = 1.0 / elt(A, row, col);
  A->is_transposed = true;
  alg__scale(scale, A, row);
  A->is_transposed = false;
  elt(A, row, col) = 1;  // Avoid precision errors.

  // Zero out all other entries in the col.
  int n = num_rows(A);
  A->is_transposed = true;
  for (int r = 0; r < n; ++r) {
    if (r == row) continue;
    alg__mul_and_add(-elt(A, col, r), A, row, A, r);
    // Set the same-column element explicitly to 0 to avoid precision errors.
    elt(A, col, r) = 0;
  }
  A->is_transposed = false;
}

alg__Status apply_lp(alg__Mat tab, Phase phase) {

  //printf("At start of apply_lp, tab:\n%s", alg__matrix_as_str(tab));

  int pivot_row_start = (phase == phase1 ? 2 : 1);
  int last_col = num_cols(tab) - 1;

  // In phase 1, we start by clearing the columns of the artificial variables.
  if (phase == phase1) {
    for (int c = 2; c < num_rows(tab); ++c) {
      make_col_a_01_col(tab, c, c);
      //printf("After clearing column %d, tableau is:\n%s", c, alg__matrix_as_str(tab));
    }
  }

  while (true) {
    int pivot_col = 1;
    for (; pivot_col < last_col && elt(tab, 0, pivot_col) <= 0; ++pivot_col);
    if (pivot_col == last_col) return alg__status_ok;

    // Now tab_{0, pivot_col} is the 1st positive entry, ignoring tab_0,0, in the top row.
    int   pivot_row = -1;
    float pivot_ratio;
    for (int r = pivot_row_start; r < num_rows(tab); ++r) {
      if (elt(tab, r, pivot_col) <= 0) continue;
      float r_ratio = elt(tab, r, last_col) / elt(tab, r, pivot_col);
      if (pivot_row == -1 || r_ratio < pivot_ratio) {
        pivot_row   = r;
        pivot_ratio = r_ratio;
      }
    }
    if (pivot_row == -1) return alg__status_unbdd_soln;

    //printf("pivot is (0-indexed) row=%d, col=%d\n", pivot_row, pivot_col);
    make_col_a_01_col(tab, pivot_row, pivot_col);
    //printf("After one iteration, the tableau is:\n%s", alg__matrix_as_str(tab));
  }
}


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
      s += snprintf(s, s_end - s, "%8.2g ", elt(M, row, col));
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

alg__Status alg__l1_min(alg__Mat A, alg__Mat b, alg__Mat x) {
  alg__Mat A2 = alg__alloc_matrix(num_rows(A), 2 * num_cols(A));
  for (int r = 0; r < num_rows(A); ++r) {
    for (int c = 0; c < num_cols(A); ++c) {
      float src_val = elt(A, r, c);
      elt(A2, r, 2 * c + 0) =  src_val;
      elt(A2, r, 2 * c + 1) = -src_val;
    }
  }

  alg__Mat c = alg__alloc_matrix(num_cols(A2), 1);
  for (int r = 0; r < num_rows(c); ++r) col_elt(c, r) = (r % 2 ? -1 : 1);

  alg__Mat x2 = alg__alloc_matrix(num_cols(A2), 1);

  alg__Status status = alg__run_lp(A2, b, x2, c);

  if (status == alg__status_ok) {
    // Copy x2 over to x.
    for (int r = 0; r < num_rows(x); ++r) {
      col_elt(x, r) = col_elt(x2, 2 * r) - col_elt(x2, 2 * r + 1);
    }
  }

  // Clean up.
  alg__free_matrix(x2);
  alg__free_matrix(c);
  alg__free_matrix(A2);

  return status;
}

alg__Status alg__l2_min(alg__Mat A, alg__Mat b, alg__Mat x) {
  if (num_rows(A) != num_rows(b)) {
    fprintf(stderr, "Error: A and b must have the same number of rows.\n");
    return alg__status_input_error;
  }
  if (num_cols(A) != num_rows(x) || num_cols(x) != 1) {
    fprintf(stderr, "Error: expected x to have size #cols(A) x 1.\n");
    return alg__status_input_error;
  }
  if (x == NULL) {
    fprintf(stderr, "Error: expected output matrix x to be pre-allocated.\n");
    return alg__status_input_error;
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

  return alg__status_ok;
}

alg__Status alg__run_lp(alg__Mat A, alg__Mat b, alg__Mat x, alg__Mat c) {

  //printf("\n");
  //printf("A is %dx%d, b is %dx%d, x is %dx%d, c is %dx%d\n",
  //       num_rows(A), num_cols(A),
  //       num_rows(b), num_cols(b),
  //       num_rows(x), num_cols(x),
  //       num_rows(c), num_cols(c));

  // Check that the size of A matches b, x, and c.
  if (num_rows(A) != num_rows(b) ||
      num_cols(A) != num_rows(x) ||
      num_cols(A) != num_rows(c)) {
    fprintf(stderr, "Error: input size mismatch seen in alg__run_lp.\n");
    return alg__status_input_error;
  }
  // Check that x, b, and c are single-column matrices.
  if (num_cols(x) != 1 || num_cols(b) != 1 || num_cols(c) != 1) {
    fprintf(stderr, "Error: expected x, b, and c to be single-column matrices in alg__run_lp.\n");
    return alg__status_input_error;
  }

  // TEMP
  //printf("A:\n%s", alg__matrix_as_str(A));
  //printf("b:\n%s", alg__matrix_as_str(b));
  //printf("x:\n%s", alg__matrix_as_str(x));
  //printf("c:\n%s", alg__matrix_as_str(c));


  // Phase 1.
  alg__Mat tab1 = alg__alloc_matrix(num_rows(A) + 2, num_cols(A) + num_rows(A) + 3);
  // Set up the artificial variable cost row = the top row in tab1.
  for (int col = 0; col < num_cols(tab1); ++col) {
    float val = (col == 0 ? 1 : 0);
    // The artifical variables each have cost 1, the negative of which is in the top row.
    if (2 <= col && col <= num_rows(A) + 1) val = -1;
    elt(tab1, 0, col) = val;
  }
  // Set up the cost row to be used in phase 2 = the second row in tab1.
  for (int col = 0; col < num_cols(tab1); ++col) {
    float val = (col == 1 ? 1 : 0);
    int c_idx = col - (num_rows(A) + 2);
    if (0 <= c_idx && c_idx < num_rows(c)) {
      val = -col_elt(c, c_idx);
    }
    elt(tab1, 1, col) = val;
  }
  // Set up the remaining rows.
  for (int row = 2; row < num_rows(tab1); ++row) {
    for (int col = 0; col < num_cols(tab1); ++col) {
      float val = 0;
      if (col == num_cols(tab1) - 1) {
        val = col_elt(b, row - 2);
      }
      int artif_col_idx = col - 2;
      if (artif_col_idx >= 0 && artif_col_idx < num_rows(A)) {
        int artif_row_idx = row - 2;
        val = (artif_col_idx == artif_row_idx ? 1 : 0);
        // Rows with b_i < 0 will be negated.
        if (col_elt(b, artif_row_idx) < 0 && val) val *= -1;
      }
      int A_col_idx = col - num_rows(A) - 2;
      if (A_col_idx >= 0 && A_col_idx < num_cols(A)) {
        val = elt(A, row - 2, A_col_idx);
      }
      // Negate each row with b_i < 0 to keep the final column nonnegative.
      if (col_elt(b, row - 2) < 0 && val) val *= -1;
      //printf("Setting tab1_{%d, %d} to %g\n", row, col, val);
      elt(tab1, row, col) = val;
    }
  }

  // TEMP
  //printf("tableau for phase 1:\n%s", alg__matrix_as_str(tab1));

  alg__Status status = apply_lp(tab1, phase1);
  // We might have an unbounded solution set; return early if so.
  if (status) return status;

  //printf("After phase 1, tableau is:\n%s", alg__matrix_as_str(tab1));

  // Check if an initial feasible solution was found.
  if (fabs(elt(tab1, 0, num_cols(tab1) - 1)) > tol) return alg__status_no_soln;

  // Phase 2.
  alg__Mat tab2 = alg__alloc_matrix(num_rows(A) + 1, num_cols(A) + 2);
  // Copy over the submatrix of tab1 that we will continue working with.
  for (int row = 0; row < num_rows(tab2); ++row) {
    for (int col = 0; col < num_cols(tab2); ++col) {
      // The src_col is either the 2nd col of tab1 or
      // the columns after the artificial variables.
      int src_col = (col == 0 ? 1 : col + 1 + num_rows(A));
      elt(tab2, row, col) = elt(tab1, row + 1, src_col);
    }
  }

  // TEMP
  //printf("phase 2 tableau is starting as:\n%s", alg__matrix_as_str(tab2));

  apply_lp(tab2, phase2);

  //printf("After phase 2, tableau is:\n%s", alg__matrix_as_str(tab2));

  // Copy the result out to x and clean up.
  for (int c = 0; c < num_cols(x); ++c) col_elt(x, c) = 0;
  int last_col = num_cols(tab2) - 1;
  for (int c = 1; c < last_col; ++c) {
    // Check to see if c is a 01 column; if yes, it gives us a value in x.
    int set_row = -1;
    for (int r = 0; r < num_rows(tab2); ++r) {
      float val = elt(tab2, r, c);
      if (val == 0) continue;
      if (set_row != -1 || val != 1) {
        // This means it's not a 01 column.
        set_row = -2;
        break;
      }
      set_row = r;  // This is the first val=1 we've seen in the column.
    }
    if (set_row > 0) {
      col_elt(x, c - 1) = elt(tab2, set_row, last_col);
    }
  }

  // TEMP
  //printf("x:\n%s", alg__matrix_as_str(x));

  alg__free_matrix(tab2);
  alg__free_matrix(tab1);

  return alg__status_ok;
}

