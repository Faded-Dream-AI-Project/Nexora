/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software.
 *
 * Tensors: dtype + shape + strides (in elements) + a data pointer. Tensors
 * are views: creating one bumps an arena for its buffer and its nx_tensor_t
 * struct; view helpers (alias/reshape/permute/narrow/wrap) allocate only a new
 * struct and share or wrap existing storage. Nothing is freed individually:
 * tensor lifetime is the arena's lifetime (marks/rewinds), except wrapped
 * external memory which the caller owns.
 *
 * Bulk arithmetic is NOT here — kernels live in src/cpu/ and src/vk/. This
 * module provides layout, indexing and debugging. */

#ifndef NX_TENSOR_H
#define NX_TENSOR_H

#include "nx_arena.h"
#include "nx_common.h"

#include <stdio.h>

#define NX_TENSOR_NAME_MAX 24

typedef struct nx_tensor {
    char name[NX_TENSOR_NAME_MAX]; /* "" when unnamed; debug aid */
    nx_dtype_t dtype;
    int32_t ndim;                  /* 0 = scalar; <= NX_MAX_DIMS */
    int32_t shape[NX_MAX_DIMS];
    size_t strides[NX_MAX_DIMS];   /* element strides; row-major when contiguous */
    size_t elems;                  /* number of elements in the view, >= 1 */
    void *data;                    /* first element of this view */
    nx_arena_t *storage;           /* arena owning the backing buffer; NULL = borrowed */
} nx_tensor_t;

/* Creates a new uninitialized tensor with row-major layout. The backing
 * buffer and the struct both come from `a` (buffer NX_ALIGNMENT-aligned).
 * dims must all be >= 1; ndim == 0 is a scalar (shape may be NULL).
 * Returns NULL with an error recorded on bad shape or arena failure. */
nx_tensor_t *nx_tensor_create(nx_arena_t *a, nx_dtype_t dtype, int32_t ndim,
                              const int32_t *shape);

/* Wraps external memory as a tensor (no copy; caller owns `mem`). `strides`
 * NULL means row-major contiguous over `shape`. The nx_tensor_t struct itself
 * still lives in `a`. */
nx_tensor_t *nx_tensor_wrap(nx_arena_t *a, void *mem, nx_dtype_t dtype, int32_t ndim,
                            const int32_t *shape, const size_t *strides);

/* New view of the exact same data (same layout, fresh struct). */
nx_tensor_t *nx_tensor_alias(nx_arena_t *a, const nx_tensor_t *t);

/* Row-major reinterpretation of the same elements. `shape` may contain one
 * -1, inferred from the element count. Requires a contiguous tensor. */
nx_tensor_t *nx_tensor_reshape(nx_arena_t *a, const nx_tensor_t *t, int32_t ndim,
                               const int32_t *shape);

/* Dimension reordering: new dim i is old dim perm[i] (PyTorch convention).
 * ndim must stay the same; perm must be a permutation of 0..ndim-1. */
nx_tensor_t *nx_tensor_permute(nx_arena_t *a, const nx_tensor_t *t, const int32_t *perm);

/* Slice along one dimension: dims [start, start+len). */
nx_tensor_t *nx_tensor_narrow(nx_arena_t *a, const nx_tensor_t *t, int32_t dim,
                              int32_t start, int32_t len);

/* Sets the debug name (truncated to fit, always NUL-terminated). */
void nx_tensor_set_name(nx_tensor_t *t, const char *name);

/* True when the stride pattern is row-major over `shape` (size-1 dims are
 * stride-agnostic, as usual). */
bool nx_tensor_is_contiguous(const nx_tensor_t *t);

/* Element count (== t->elems). */
size_t nx_tensor_numel(const nx_tensor_t *t);

/* Logical element access by multi-index (`idx` has t->ndim entries; for a
 * scalar pass anything, idx is not read). Bootstrap/test helpers: they assert
 * dtype and bounds and are not meant for hot loops. */
float nx_tensor_get_f32_at(const nx_tensor_t *t, const int32_t *idx);
void nx_tensor_set_f32_at(nx_tensor_t *t, const int32_t *idx, float value);
int32_t nx_tensor_get_i32_at(const nx_tensor_t *t, const int32_t *idx);
void nx_tensor_set_i32_at(nx_tensor_t *t, const int32_t *idx, int32_t value);

/* Fills every element of the view in logical order (any layout). */
void nx_tensor_fill_f32(nx_tensor_t *t, float value);
void nx_tensor_fill_i32(nx_tensor_t *t, int32_t value);

/* Iterates the view in logical (row-major) order, exposing the element
 * offset from t->data. Usage:
 *     nx_tensor_iter_t it; size_t off;
 *     nx_tensor_iter_init(&it, t);
 *     while (nx_tensor_iter_next(&it, &off)) ... ((float *)t->data)[off] ... */
typedef struct nx_tensor_iter {
    const nx_tensor_t *t;
    int32_t idx[NX_MAX_DIMS];
    bool done;
} nx_tensor_iter_t;

void nx_tensor_iter_init(nx_tensor_iter_t *it, const nx_tensor_t *t);
bool nx_tensor_iter_next(nx_tensor_iter_t *it, size_t *offset);

/* Debug print: one header line + up to max_elems element values on one line.
 * f16/bf16 print as raw hex (no decode yet — lands with the cpu/ backend). */
void nx_tensor_print(const nx_tensor_t *t, FILE *f, size_t max_elems);

#endif /* NX_TENSOR_H */
