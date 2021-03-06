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

#define tol 1e-4

#define unbdd_soln_str "The solution set is unbounded."


// Public globals.

const char *alg__err_str = NULL;


// Internal types and globals.

typedef enum {
  phase1,
  phase2
} Phase;

static int dbg_verbosity = 0;

// Internal functions.

#define dbg_printf(...) if (dbg_verbosity > 0) printf(__VA_ARGS__)
#define dbg_print_matrix(m) \
  if (dbg_verbosity > 0) { char *s = alg__matrix_as_str(m); printf("%s", s); free(s); }

#define data_size(nrows, ncols) (sizeof(float) * nrows * ncols)
#define num_cols(A) (A->is_transposed ? A->nrows : A->ncols)
#define num_rows(A) (A->is_transposed ? A->ncols : A->nrows)
#define elt(A, i, j) alg__elt(A, i, j)
#define col_elt(A, i) elt(A, i, 0)

// This sets the given entry to 1 by scaling its row
// and clears the rest of its column with row operations.
// The given entry is expected to be nonzero.
static void make_col_a_01_col(alg__Mat A, int row, int col) {
  // Normalize the row so that A_{row,col} = 1.
  float scale = 1.0 / elt(A, row, col);
  A->is_transposed = !A->is_transposed;
  alg__scale(scale, A, row);
  A->is_transposed = !A->is_transposed;
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

static alg__Status apply_lp(alg__Mat tab, Phase phase) {

  dbg_printf("At start of apply_lp (phase %d), tableau is:\n",
             phase == phase1 ? 1 : 2);
  dbg_print_matrix(tab);


  int pivot_row_start = (phase == phase1 ? 2 : 1);
  int last_col = num_cols(tab) - 1;

  // In phase 1, we start by clearing the columns of the artificial variables.
  if (phase == phase1) {
    for (int c = 2; c < num_rows(tab); ++c) {
      make_col_a_01_col(tab, c, c);
      dbg_printf("After clearing column %d, tableau is:\n", c);
      dbg_print_matrix(tab);
    }
  }

  while (true) {
    int pivot_col = 1;
    for (; pivot_col < last_col && elt(tab, 0, pivot_col) < tol; ++pivot_col);
    if (pivot_col == last_col) return alg__status_ok;

    // Now tab_{0, pivot_col} is the 1st positive entry, ignoring tab_0,0, in the top row.
    int   pivot_row = -1;
    float pivot_ratio;
    for (int r = pivot_row_start; r < num_rows(tab); ++r) {
      if (elt(tab, r, pivot_col) < tol) continue;
      float r_ratio = elt(tab, r, last_col) / elt(tab, r, pivot_col);
      if (pivot_row == -1 || r_ratio < pivot_ratio) {
        pivot_row   = r;
        pivot_ratio = r_ratio;
      }
    }
    if (pivot_row == -1) {
      alg__err_str = unbdd_soln_str;
      return alg__status_unbdd_soln;
    }

    dbg_printf("pivot is (0-indexed) row=%d, col=%d\n", pivot_row, pivot_col);
    make_col_a_01_col(tab, pivot_row, pivot_col);
    dbg_printf("After an iteration, the tableau is:\n");
    dbg_print_matrix(tab);
  }
}

// Accepts a matrix A and allocates a new one, A2, that's caller-owned.
// Each implied variable x_i of A, corresponding to each column of A,
// is split into a positive and negative part: x_i = x^+_i - x^-_i in A2.
// So A2 is twice is wide as A. A2 is the return value.
static alg__Mat convert_to_restricted_vars(alg__Mat A) {
  alg__Mat A2 = alg__alloc_matrix(num_rows(A), 2 * num_cols(A));
  for (int r = 0; r < num_rows(A); ++r) {
    for (int c = 0; c < num_cols(A); ++c) {
      float src_val = elt(A, r, c);
      elt(A2, r, 2 * c + 0) =  src_val;
      elt(A2, r, 2 * c + 1) = -src_val;
    }
  }
  return A2;
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
    alg__err_str = "Expected A, B to have the same #rows.";
    return 0.0 / 0.0;  // If NaN is supported, it is returned here.
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

alg__Status alg__QR(alg__Mat Q, alg__Mat R) {
  if (num_rows(Q) < num_cols(Q)) {
    alg__err_str = "Expected alg__QR input to be a tall or square matrix.";
    return alg__status_input_error;
  }

  if (R) memset(R->data, 0, sizeof(float) * num_rows(R) * num_cols(R));

  alg__Status status = alg__status_ok;
  // Q starts off as an arbitrary tall-or-square matrix A.
  for (int i = 0; i < num_cols(Q); ++i) {
    float norm = alg__norm(Q, i);
    if (norm == 0) {
      status = alg__status_lin_dep;
      continue;
    }
    alg__scale(1.0 / norm, Q, i);
    if (R) elt(R, i, i) = norm;
    for (int j = i + 1; j < num_cols(Q); ++j) {
      float dot_prod = alg__dot_prod(Q, i, Q, j);
      if (R) elt(R, i, j) = dot_prod;
      alg__mul_and_add(-dot_prod, Q, i, Q, j);
    }
  }
  return status;
}

// 4. Optimizations.

alg__Status alg__l1_min(alg__Mat A, alg__Mat b, alg__Mat x) {
  alg__Mat A2 = convert_to_restricted_vars(A);

  alg__Mat c = alg__alloc_matrix(num_cols(A2), 1);
  for (int r = 0; r < num_rows(c); ++r) col_elt(c, r) = 1;

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
  if (x == NULL) {
    alg__err_str = "The matrix x is expected to be pre-allocated.";
    return alg__status_input_error;
  }
  if (num_rows(A) != num_rows(b)) {
    alg__err_str = "A and b must have the same number of rows.";
    return alg__status_input_error;
  }
  if (num_cols(A) != num_rows(x) || num_cols(x) != 1) {
    alg__err_str = "x is expected to have size #cols(A) x 1.";
    return alg__status_input_error;
  }

  // We want to work with rows of A (and Q).
  A->is_transposed = !A->is_transposed;
  alg__Mat Q = alg__copy_matrix(A);
  alg__QR(Q, NULL);

  for (int i = 0; i < num_cols(A); ++i) {
    float a_i_q_i = alg__dot_prod(A, i, Q, i);
    float a_i_x   = alg__dot_prod(A, i, x, 0);
    float diff    = col_elt(b, i) - a_i_x;
    if (diff == 0) continue;  // Don't worry about a_i_q_i = 0 in this case.
    if (a_i_q_i == 0) {
      alg__err_str = "The solution set is empty.";
      return alg__status_no_soln;
    }
    float alpha = diff / a_i_q_i;
    alg__mul_and_add(alpha, Q, i, x, 0);
  }

  // Clean up.
  A->is_transposed = !A->is_transposed;
  alg__free_matrix(Q);

  return alg__status_ok;
}

alg__Status alg__linf_min (alg__Mat A, alg__Mat b, alg__Mat x) {

  if (x == NULL) {
    alg__err_str = "The matrix x is expected to be pre-allocated.";
    return alg__status_input_error;
  }
  if (num_rows(A) != num_rows(b)) {
    alg__err_str = "A and b must have the same number of rows.";
    return alg__status_input_error;
  }
  if (num_cols(A) != num_rows(x) || num_cols(x) != 1) {
    alg__err_str = "x is expected to have size #cols(A) x 1.";
    return alg__status_input_error;
  }

  // We convert A -> A2 -> A3. The first conversion is only
  // to convert unrestricted to restricted variables.
  // That is, the user didn't say x >= 0 but LP uses that constraint,
  // so we wrap our application of LP to meet the user's expectations.
  alg__Mat A2 = convert_to_restricted_vars(A);

  // Our approach is to add one slack variable s_i for each
  // variable x_i in A2, and another variable t to measure ||x||_inf.
  // The added inequalities correspond to x_i <= t, captured
  // by the slack variables as s_i = t - x_i >= 0, and sent in
  // as x_i + s_i - t = 0.
  // This is captured in the augmented matrices A3 and b3:
  //
  //        <x> <s> <t>    b3:
  //  A3 = ( A2  0   0 )  ( b )
  //       ( I   I  -1 )  ( 0 )
  //
  // (There is no b2; I use the name b3 for consistency with A3.)

  // Set up the augmented parameter matrix A3.
  int nr = num_rows(A2);
  int nc = num_cols(A2);
  alg__Mat A3 = alg__alloc_matrix(nr + nc, 2 * nc + 1);
  memset(A3->data, 0, num_rows(A3) * num_cols(A3) * sizeof(float));

  // Set A2 as a submatrix of A3.
  for (int r = 0; r < nr; ++r) {
    for (int c = 0; c < nc; ++c) {
      elt(A3, r, c) = elt(A2, r, c);
    }
  }
  // Set up the two identity (I) submatrices of A3.
  for (int r = 0; r < nc; ++r) {
    elt(A3, r + nr, r +  0) = 1;
    elt(A3, r + nr, r + nc) = 1;
  }
  // Set up the -1 column of A3.
  for (int r = nr; r < num_rows(A3); ++r) {
    elt(A3, r, num_cols(A3) - 1) = -1;
  }

  // Set up b3, the augmented version of b.
  alg__Mat b3 = alg__alloc_matrix(nr + nc, 1);
  for (int r = 0; r < num_rows(b3); ++r) {
    col_elt(b3, r) = (r < nr ? col_elt(b, r) : 0);
  }

  // Set up c3, the cost matrix. The cost is simply the value of t.
  alg__Mat c3 = alg__alloc_matrix(num_cols(A3), 1);
  memset(c3->data, 0, num_rows(c3) * sizeof(float));
  col_elt(c3, num_rows(c3) - 1) = 1;

  // Set up x3.
  alg__Mat x3 = alg__alloc_matrix(num_cols(A3), 1);

  alg__Status status = alg__run_lp(A3, b3, x3, c3);
  if (status != alg__status_ok) goto end_linf;

  // Copy out the user-facing values from x3.
  for (int r = 0; r < num_rows(x); ++r) {
    col_elt(x, r) = col_elt(x3, 2 * r) - col_elt(x3, 2 * r + 1);
  }

  dbg_printf("x:\n");
  dbg_print_matrix(x);

end_linf:

  alg__free_matrix(x3);
  alg__free_matrix(c3);
  alg__free_matrix(b3);
  alg__free_matrix(A3);
  alg__free_matrix(A2);

  return status;
}

alg__Status alg__run_lp(alg__Mat A, alg__Mat b, alg__Mat x, alg__Mat c) {

  dbg_printf("\n");
  dbg_printf("A is %dx%d, b is %dx%d, x is %dx%d, c is %dx%d\n",
             num_rows(A), num_cols(A),
             num_rows(b), num_cols(b),
             num_rows(x), num_cols(x),
             num_rows(c), num_cols(c));

  // Check that the size of A matches b, x, and c.
  if (num_rows(A) != num_rows(b) ||
      num_cols(A) != num_rows(x) ||
      num_cols(A) != num_rows(c)) {
    alg__err_str = "The input sizes of A, b, x, c do not all match.";
    return alg__status_input_error;
  }
  // Check that x, b, and c are single-column matrices.
  if (num_cols(x) != 1 || num_cols(b) != 1 || num_cols(c) != 1) {
    alg__err_str = "x, b, and c are all expected to be single-column matrices.";
    return alg__status_input_error;
  }

  dbg_printf("A:\n"); dbg_print_matrix(A);
  dbg_printf("b:\n"); dbg_print_matrix(b);
  dbg_printf("x:\n"); dbg_print_matrix(x);
  dbg_printf("c:\n"); dbg_print_matrix(c);


  // Phase 1.
  alg__Mat tab1 = alg__alloc_matrix(num_rows(A) + 2, num_cols(A) + num_rows(A) + 3);
  alg__Mat tab2 = alg__alloc_matrix(num_rows(A) + 1, num_cols(A) + 2);
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
      elt(tab1, row, col) = val;
    }
  }

  dbg_printf("tableau for phase 1:\n");
  dbg_print_matrix(tab1);

  alg__Status status = apply_lp(tab1, phase1);
  if (status != alg__status_ok) goto end_lp;  // It may be an unbounded solution set.

  dbg_printf("After phase 1, tableau is:\n");
  dbg_print_matrix(tab1);

  // Check if an initial feasible solution was found.
  if (fabs(elt(tab1, 0, num_cols(tab1) - 1)) > tol) {
    alg__err_str = "There are no solutions x with Ax=b and x>=0.";
    return alg__status_no_soln;
  }

  // Phase 2.
  // Copy over the submatrix of tab1 that we will continue working with.
  for (int row = 0; row < num_rows(tab2); ++row) {
    for (int col = 0; col < num_cols(tab2); ++col) {
      // The src_col is either the 2nd col of tab1 or
      // the columns after the artificial variables.
      int src_col = (col == 0 ? 1 : col + 1 + num_rows(A));
      elt(tab2, row, col) = elt(tab1, row + 1, src_col);
    }
  }

  dbg_printf("phase 2 tableau is starting as:\n");
  dbg_print_matrix(tab2);

  status = apply_lp(tab2, phase2);
  if (status != alg__status_ok) goto end_lp;  // It may be an unbounded solution set.

  dbg_printf("After phase 2, tableau is:\n");
  dbg_print_matrix(tab2);

  // Copy the result out to x and clean up.
  memset(x->data, 0, num_rows(x) * sizeof(float));
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

  dbg_printf("x:\n");
  dbg_print_matrix(x);

end_lp:
  alg__free_matrix(tab2);
  alg__free_matrix(tab1);

  return status;
}

