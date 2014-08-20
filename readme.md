# calgebra

*A C library for linear algebraic optimization problems.*

This is a small, fast library focused on solving the
following linear problems:

* L2-minimization, aka *least squares*,
* L1-minimization, and
* general linear programming problems.

These problems and the algorithms used
are described in more detail below.

## Problem types and algorithms used

### L2-minimization

Given input matrix *A* and column vector *b*, find
column vector *x* so that:

* *Ax=b*, and
* ||*x*||<sub>2</sub> is minimized.

You can solve this in `calgebra` by using the
`alg__l2_min` function; a usage example is given below.

The algorithm used by `calgebra` is essentially to project any
feasible solution onto the row span of *A*. Some texts use
the term
[*least squares*](http://en.wikipedia.org/wiki/Least_squares)
to indicate a slightly different
problem, where *b* may not be in the column span of *A*, so that
the value minimized is ||*Ax-b*||<sub>2</sub>. This is
an equivalent problem that can also be solved with
`alg__l2_min` by using input matrix *C* and column matrix *d* chosen
so that {*x*: *Cx=d*} = {*Ax-b*: *x* is free}.
Internally, `calgebra` uses a modified Gram-Schmidt
[QR decomposition](http://en.wikipedia.org/wiki/QR_decomposition)
to perform the projection onto the row span of *A*.

### L1-minimization

This is the same as L2-minimization, except that we
minimize ||*x*||<sub>1</sub> instead of ||*x*||<sub>2</sub>.

Specifically, given input matrix *A* and column vector *b*, find
column vector *x* so that:

* *Ax=b*, and
* ||*x*||<sub>1</sub> is minimized.

You can solve this in `calgebra` by using the
`alg__l1_min` function; a usage example is given below.

Internally, this problem is reduced to a general
linear program, described next.

### General linear programming

Given input matrix *A* and column vectors *b* and *c*, find
column vector *x* so that:

* *Ax=b*,
* *x*≥0, and
* *c*<sup>T</sup>*x* is minimized.

The algorithm used by `calgebra` is
[the simplex method](http://en.wikipedia.org/wiki/Simplex_algorithm),
along with Bland's rule to avoid cycling. Although this algorithm
is not guaranteed to complete in polynomial time, it is widely
believed to be the fastest for most practical applications.

## Examples

### L1- and L2-minimization example

In the example below, we call several key functions:

* `alg__alloc_matrix(nrows, ncols)` allocates and returns a new matrix.
* `alg__l{1,2}_min(A, b, x)` solves an L{1,2}-minimization problem.
* `alg__free_matrix(M)` frees matrix `M`.

A matrix is stored as type `alg__Mat`, which is a typedef
for `alg__MatStruct *`, defined transparently in the header.
Matrices are stored in row-major order, meaning that
entries in a single row are stored contiguously in memory.

Use `alg__elt(M, i, j)` to access the element of matrix `M`
in row `i` and column `j`; this is aware of the transposed
state of `M` (that is, the value of `M->is_transposed` is
taken into consideration).

```
// This problem has the feasible solution set
// (1 1 0) + t(-1 -1 8), where t is free, specified with
// A = (4 4 1)  b = (8)
//     (8 0 1)      (8)
// For this set, x = (1 1 0)^T minimizes ||x||_1.

alg__Mat A = alg__alloc_matrix(2, 3);
memcpy(A->data, ((float[])
       {  4,  4,  1,
          8,  0,  1 }), 6 * sizeof(float));
alg__Mat b = alg__alloc_matrix(2, 1);
memcpy(b->data, ((float[]){  8,  8 }), 2 * sizeof(float));
alg__Mat x = alg__alloc_matrix(3, 1);

alg__Status status;

status = alg__l1_min(A, b, x);
if (status == alg__status_ok) {
  // Now x is the solution of the L1-min problem.
}

status = alg__l2_min(A, b, x);
if (status == alg__status_ok) {
  // Now x is the solution of the L2-min problem.
}

alg__free_matrix(x);
alg__free_matrix(b);
alg__free_matrix(A);
```


### Linear programming example

This illustrates use of the `alg__run_lp` function
to solve a linear program.

```
alg__Mat A = alg__alloc_matrix(3, 5);
memcpy(A->data, ((float[])
      {  1,  0,  0,  0,  1,
         0,  1,  0,  4, -5,
         0,  0,  1, -4,  1}), 3 * 5 * sizeof(float));

alg__Mat b = alg__alloc_matrix(3, 1);
memcpy(b->data, ((float[]){ 7, -7, -5 }), 3 * sizeof(float));

alg__Mat c = alg__alloc_matrix(5, 1);
memcpy(c->data, ((float[]){ 0, 0, 0, 3, 2 }), 5 * sizeof(float));

alg__Mat x = alg__alloc_matrix(5, 1);

alg__Status status = alg__run_lp(A, b, x, c);

if (status == alg__status_ok) {
  // Now x is the solution of the linear program.
}

alg__free_matrix(x);
alg__free_matrix(c);
alg__free_matrix(b);
alg__free_matrix(A);
```

## Values of the `alg__Status` enum

The following status values are possible:

status value       | meaning
--------------------------|--------------------
`alg__status_ok`          | It's all good.
`alg__status_no_soln`     | *Ax=b* has no solutions.
`alg__status_unbdd_soln`  | The value of *c*<sup>T</sup>*x* can be made arbitrarily low.
`alg__status_input_error` | The input matrix dimensions are not as expected, or an input was unexpectedly `NULL`.
`alg__status_lin_dep`     | (Only from `alg__QR`) The input had linearly dependent columns; the output is still valid.
