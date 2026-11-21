/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software. */

#include "nx_tensor.h"

#include "nx_arena.h"
#include "nx_error.h"

#include <assert.h>
#include <inttypes.h>
#include <string.h>

/* ---- shape helpers ---------------------------------------------------- */

/* Validates ndim/shape and computes the element count; records a specific
 * error message on failure. */
static bool shape_check(int32_t ndim, const int32_t *shape, size_t *out_elems)
{
    size_t n = 1;

    if (ndim < 0 || ndim > NX_MAX_DIMS) {
        nx_error_set(NX_E_RANGE, "ndim %d outside [0, %d]", (int)ndim, (int)NX_MAX_DIMS);
        return false;
    }
    for (int32_t d = 0; d < ndim; d++) {
        if (shape[d] < 1) {
            nx_error_set(NX_E_INVAL, "shape[%" PRId32 "] = %" PRId32 ", must be >= 1", d,
                         shape[d]);
            return false;
        }
        if (nx_size_mul_overflows(n, (size_t)shape[d], &n)) {
            nx_error_set(NX_E_OVERFLOW, "shape[%" PRId32 "] = %" PRId32
                                           " overflows the element count",
                         d, shape[d]);
            return false;
        }
    }
    *out_elems = n;
    return true;
}

static void shape_set_row_major(int32_t ndim, const int32_t *shape, size_t *strides)
{
    size_t acc = 1;

    for (int32_t d = ndim - 1; d >= 0; d--) {
        strides[d] = acc;
        acc *= (size_t)shape[d];
    }
    if (ndim == 0) {
        strides[0] = 0; /* scalar: unused */
    }
}

static bool dtype_valid(nx_dtype_t dtype)
{
    int d = (int)dtype;
    return d >= 0 && d < (int)NX_DTYPE_COUNT && nx_dtype_size(dtype) != 0;
}

/* ---- struct allocation ------------------------------------------------ */

static nx_tensor_t *tensor_new(nx_arena_t *a)
{
    nx_tensor_t *t = nx_arena_alloc(a, sizeof *t, _Alignof(nx_tensor_t));

    if (t == NULL) {
        return NULL; /* arena recorded the error */
    }
    memset(t, 0, sizeof *t);
    return t;
}

/* New view struct sharing src's dtype/name. */
static nx_tensor_t *view_new(nx_arena_t *a, const nx_tensor_t *src)
{
    nx_tensor_t *n = tensor_new(a);

    if (n == NULL) {
        return NULL;
    }
    n->dtype = src->dtype;
    memcpy(n->name, src->name, sizeof n->name);
    return n;
}

static size_t logical_offset(const nx_tensor_t *t, const int32_t *idx)
{
    size_t off = 0;

    assert(idx != NULL || t->ndim == 0);
    for (int32_t d = 0; d < t->ndim; d++) {
        assert(idx[d] >= 0 && idx[d] < t->shape[d]);
        off += (size_t)idx[d] * t->strides[d];
    }
    return off;
}

/* ---- creation --------------------------------------------------------- */

nx_tensor_t *nx_tensor_create(nx_arena_t *a, nx_dtype_t dtype, int32_t ndim,
                              const int32_t *shape)
{
    size_t elems, bytes;
    nx_tensor_t *t;

    if (a == NULL) {
        nx_error_set(NX_E_INVAL, "arena NULL");
        return NULL;
    }
    if (!dtype_valid(dtype)) {
        nx_error_set(NX_E_INVAL, "bad dtype %d", (int)dtype);
        return NULL;
    }
    if (ndim > 0 && shape == NULL) {
        nx_error_set(NX_E_INVAL, "shape NULL");
        return NULL;
    }
    if (!shape_check(ndim, shape, &elems)) {
        return NULL;
    }
    if (nx_size_mul_overflows(elems, nx_dtype_size(dtype), &bytes)) {
        nx_error_set(NX_E_OVERFLOW, "tensor byte size overflows (%zu elements)", elems);
        return NULL;
    }

    t = tensor_new(a);
    if (t == NULL) {
        return NULL;
    }
    t->data = nx_arena_alloc(a, bytes, NX_ALIGNMENT);
    if (t->data == NULL) {
        return NULL; /* arena recorded the error */
    }
    t->dtype = dtype;
    t->ndim = ndim;
    t->elems = elems;
    for (int32_t d = 0; d < ndim; d++) {
        t->shape[d] = shape[d];
    }
    shape_set_row_major(ndim, shape, t->strides);
    t->storage = a;
    return t;
}

nx_tensor_t *nx_tensor_wrap(nx_arena_t *a, void *mem, nx_dtype_t dtype, int32_t ndim,
                            const int32_t *shape, const size_t *strides)
{
    size_t elems;
    nx_tensor_t *t;

    if (a == NULL) {
        nx_error_set(NX_E_INVAL, "arena NULL");
        return NULL;
    }
    if (mem == NULL) {
        nx_error_set(NX_E_INVAL, "mem NULL");
        return NULL;
    }
    if (!dtype_valid(dtype)) {
        nx_error_set(NX_E_INVAL, "bad dtype %d", (int)dtype);
        return NULL;
    }
    if (ndim > 0 && shape == NULL) {
        nx_error_set(NX_E_INVAL, "shape NULL");
        return NULL;
    }
    if (!shape_check(ndim, shape, &elems)) {
        return NULL;
    }

    t = tensor_new(a);
    if (t == NULL) {
        return NULL;
    }
    t->dtype = dtype;
    t->ndim = ndim;
    t->elems = elems;
    for (int32_t d = 0; d < ndim; d++) {
        t->shape[d] = shape[d];
    }
    if (strides != NULL) {
        memcpy(t->strides, strides, (size_t)ndim * sizeof *strides);
        if (ndim == 0) {
            t->strides[0] = 0;
        }
    } else {
        shape_set_row_major(ndim, shape, t->strides);
    }
    t->data = mem;
    t->storage = NULL; /* borrowed */
    return t;
}

/* ---- views ------------------------------------------------------------ */

nx_tensor_t *nx_tensor_alias(nx_arena_t *a, const nx_tensor_t *t)
{
    nx_tensor_t *n;

    if (a == NULL || t == NULL) {
        nx_error_set(NX_E_INVAL, "alias: arena or tensor NULL");
        return NULL;
    }
    n = tensor_new(a);
    if (n == NULL) {
        return NULL;
    }
    *n = *t;
    return n;
}

nx_tensor_t *nx_tensor_reshape(nx_arena_t *a, const nx_tensor_t *t, int32_t ndim,
                               const int32_t *shape)
{
    nx_tensor_t *n;
    int32_t new_shape[NX_MAX_DIMS];
    size_t known = 1;
    int32_t infer = -1;

    if (a == NULL || t == NULL) {
        nx_error_set(NX_E_INVAL, "reshape: arena or tensor NULL");
        return NULL;
    }
    if (ndim < 0 || ndim > NX_MAX_DIMS) {
        nx_error_set(NX_E_RANGE, "reshape: ndim %d outside [0, %d]", (int)ndim,
                     (int)NX_MAX_DIMS);
        return NULL;
    }
    if (ndim > 0 && shape == NULL) {
        nx_error_set(NX_E_INVAL, "reshape: shape NULL");
        return NULL;
    }
    if (!nx_tensor_is_contiguous(t)) {
        nx_error_set(NX_E_INVAL, "reshape: tensor is not contiguous");
        return NULL;
    }

    for (int32_t d = 0; d < ndim; d++) {
        if (shape[d] == -1) {
            if (infer >= 0) {
                nx_error_set(NX_E_INVAL, "reshape: more than one -1");
                return NULL;
            }
            infer = d;
            new_shape[d] = 1; /* placeholder until inferred */
        } else if (shape[d] >= 1) {
            new_shape[d] = shape[d];
            if (nx_size_mul_overflows(known, (size_t)shape[d], &known)) {
                nx_error_set(NX_E_OVERFLOW, "reshape: shape overflows");
                return NULL;
            }
        } else {
            nx_error_set(NX_E_INVAL, "reshape: shape[%" PRId32 "] = %" PRId32
                                         ", must be >= 1 or -1",
                         d, shape[d]);
            return NULL;
        }
    }
    if (infer >= 0) {
        if (t->elems % known != 0) {
            nx_error_set(NX_E_INVAL, "reshape: %" PRIu32
                                         " does not divide the element count",
                         (uint32_t)known);
            return NULL;
        }
        size_t inferred = t->elems / known;
        if (inferred > (size_t)INT32_MAX) {
            nx_error_set(NX_E_RANGE, "reshape: inferred dim %zu exceeds INT32_MAX", inferred);
            return NULL;
        }
        new_shape[infer] = (int32_t)inferred;
    } else if (known != t->elems) {
        nx_error_set(NX_E_INVAL, "reshape: %" PRIu32
                                     " elements given but tensor has %zu",
                     (uint32_t)known, t->elems);
        return NULL;
    }

    n = view_new(a, t);
    if (n == NULL) {
        return NULL;
    }
    n->ndim = ndim;
    n->elems = t->elems;
    for (int32_t d = 0; d < ndim; d++) {
        n->shape[d] = new_shape[d];
    }
    shape_set_row_major(ndim, new_shape, n->strides);
    n->data = t->data;
    n->storage = t->storage;
    return n;
}

nx_tensor_t *nx_tensor_permute(nx_arena_t *a, const nx_tensor_t *t, const int32_t *perm)
{
    bool seen[NX_MAX_DIMS] = { false };
    nx_tensor_t *n;

    if (a == NULL || t == NULL || perm == NULL) {
        nx_error_set(NX_E_INVAL, "permute: arena, tensor or perm NULL");
        return NULL;
    }
    if (t->ndim < 1) {
        nx_error_set(NX_E_INVAL, "permute: cannot permute a scalar");
        return NULL;
    }
    for (int32_t i = 0; i < t->ndim; i++) {
        if (perm[i] < 0 || perm[i] >= t->ndim || seen[perm[i]]) {
            nx_error_set(NX_E_INVAL, "permute: perm is not a permutation of 0..%d",
                         (int)(t->ndim - 1));
            return NULL;
        }
        seen[perm[i]] = true;
    }

    n = view_new(a, t);
    if (n == NULL) {
        return NULL;
    }
    n->ndim = t->ndim;
    n->elems = t->elems;
    for (int32_t i = 0; i < t->ndim; i++) {
        n->shape[i] = t->shape[perm[i]];
        n->strides[i] = t->strides[perm[i]];
    }
    n->data = t->data;
    n->storage = t->storage;
    return n;
}

nx_tensor_t *nx_tensor_narrow(nx_arena_t *a, const nx_tensor_t *t, int32_t dim,
                              int32_t start, int32_t len)
{
    nx_tensor_t *n;

    if (a == NULL || t == NULL) {
        nx_error_set(NX_E_INVAL, "narrow: arena or tensor NULL");
        return NULL;
    }
    if (dim < 0 || dim >= t->ndim) {
        nx_error_set(NX_E_RANGE, "narrow: dim %" PRId32 " outside [0, %d)", dim,
                     (int)t->ndim);
        return NULL;
    }
    if (len < 1 || start < 0 || (int64_t)start + len > t->shape[dim]) {
        nx_error_set(NX_E_RANGE, "narrow: [%d, %d) outside dim %" PRId32
                                     " of size %" PRId32,
                     (int)start, (int)(start + len), dim, t->shape[dim]);
        return NULL;
    }

    n = view_new(a, t);
    if (n == NULL) {
        return NULL;
    }
    n->ndim = t->ndim;
    for (int32_t d = 0; d < t->ndim; d++) {
        n->shape[d] = t->shape[d];
        n->strides[d] = t->strides[d];
    }
    n->shape[dim] = len;
    n->elems = t->elems / (size_t)t->shape[dim] * (size_t)len;
    {
        size_t skip = (size_t)start * t->strides[dim];
        n->data = (char *)t->data + skip * nx_dtype_size(t->dtype);
    }
    n->storage = t->storage;
    return n;
}

/* ---- metadata --------------------------------------------------------- */

void nx_tensor_set_name(nx_tensor_t *t, const char *name)
{
    assert(t != NULL && name != NULL);
    snprintf(t->name, sizeof t->name, "%s", name);
}

bool nx_tensor_is_contiguous(const nx_tensor_t *t)
{
    size_t acc = 1;

    assert(t != NULL);
    for (int32_t d = t->ndim - 1; d >= 0; d--) {
        if (t->shape[d] != 1 && t->strides[d] != acc) {
            return false;
        }
        acc *= (size_t)t->shape[d];
    }
    return true;
}

size_t nx_tensor_numel(const nx_tensor_t *t)
{
    assert(t != NULL);
    return t->elems;
}

/* ---- element access --------------------------------------------------- */

float nx_tensor_get_f32_at(const nx_tensor_t *t, const int32_t *idx)
{
    assert(t != NULL && t->dtype == NX_F32);
    return ((const float *)t->data)[logical_offset(t, idx)];
}

void nx_tensor_set_f32_at(nx_tensor_t *t, const int32_t *idx, float value)
{
    assert(t != NULL && t->dtype == NX_F32);
    ((float *)t->data)[logical_offset(t, idx)] = value;
}

int32_t nx_tensor_get_i32_at(const nx_tensor_t *t, const int32_t *idx)
{
    assert(t != NULL && (t->dtype == NX_I32 || t->dtype == NX_I16 || t->dtype == NX_I8));
    int64_t v;
    switch (t->dtype) {
    case NX_I16:
        v = ((const int16_t *)t->data)[logical_offset(t, idx)];
        break;
    case NX_I8:
        v = ((const int8_t *)t->data)[logical_offset(t, idx)];
        break;
    default:
        v = ((const int32_t *)t->data)[logical_offset(t, idx)];
        break;
    }
    if (v < INT32_MIN) {
        return INT32_MIN;
    }
    if (v > INT32_MAX) {
        return INT32_MAX;
    }
    return (int32_t)v;
}

void nx_tensor_set_i32_at(nx_tensor_t *t, const int32_t *idx, int32_t value)
{
    assert(t != NULL && (t->dtype == NX_I32 || t->dtype == NX_I16 || t->dtype == NX_I8));
    size_t off = logical_offset(t, idx);
    switch (t->dtype) {
    case NX_I16:
        ((int16_t *)t->data)[off] = (int16_t)value;
        break;
    case NX_I8:
        ((int8_t *)t->data)[off] = (int8_t)value;
        break;
    default:
        ((int32_t *)t->data)[off] = value;
        break;
    }
}

void nx_tensor_fill_f32(nx_tensor_t *t, float value)
{
    nx_tensor_iter_t it;
    size_t off;

    assert(t != NULL && t->dtype == NX_F32);
    nx_tensor_iter_init(&it, t);
    while (nx_tensor_iter_next(&it, &off)) {
        ((float *)t->data)[off] = value;
    }
}

void nx_tensor_fill_i32(nx_tensor_t *t, int32_t value)
{
    nx_tensor_iter_t it;
    size_t off;

    assert(t != NULL && (t->dtype == NX_I32 || t->dtype == NX_I16 || t->dtype == NX_I8));
    nx_tensor_iter_init(&it, t);
    while (nx_tensor_iter_next(&it, &off)) {
        nx_tensor_set_i32_at(t, it.idx, value);
    }
}

/* ---- iteration -------------------------------------------------------- */

void nx_tensor_iter_init(nx_tensor_iter_t *it, const nx_tensor_t *t)
{
    assert(it != NULL && t != NULL);
    it->t = t;
    for (int32_t d = 0; d < NX_MAX_DIMS; d++) {
        it->idx[d] = 0;
    }
    it->done = false;
}

bool nx_tensor_iter_next(nx_tensor_iter_t *it, size_t *offset)
{
    const nx_tensor_t *t;
    int32_t d;

    assert(it != NULL && offset != NULL);
    if (it->done) {
        return false;
    }
    t = it->t;
    *offset = logical_offset(t, it->idx);

    d = t->ndim - 1;
    while (d >= 0) {
        it->idx[d]++;
        if (it->idx[d] < t->shape[d]) {
            break;
        }
        it->idx[d] = 0;
        d--;
    }
    if (d < 0) {
        it->done = true;
    }
    return true;
}

/* ---- debug print ------------------------------------------------------ */

static void print_dims(FILE *f, const char *label, const int32_t *v, int32_t ndim)
{
    fprintf(f, "%s=[", label);
    for (int32_t d = 0; d < ndim; d++) {
        fprintf(f, "%s%" PRId32, d != 0 ? "," : "", v[d]);
    }
    fprintf(f, "]");
}

static void print_strides(FILE *f, const size_t *v, int32_t ndim)
{
    fprintf(f, "strides=[");
    for (int32_t d = 0; d < ndim; d++) {
        fprintf(f, "%s%zu", d != 0 ? "," : "", v[d]);
    }
    fprintf(f, "]");
}

static void print_element(FILE *f, const nx_tensor_t *t, size_t off)
{
    switch (t->dtype) {
    case NX_F32:
        fprintf(f, "%g", ((const float *)t->data)[off]);
        break;
    case NX_I32:
        fprintf(f, "%" PRId32, ((const int32_t *)t->data)[off]);
        break;
    case NX_I16:
        fprintf(f, "%" PRId16, ((const int16_t *)t->data)[off]);
        break;
    case NX_I8:
        fprintf(f, "%" PRId8, ((const int8_t *)t->data)[off]);
        break;
    case NX_U8:
        fprintf(f, "%" PRIu8, ((const uint8_t *)t->data)[off]);
        break;
    case NX_F16:
    case NX_BF16:
        fprintf(f, "0x%04" PRIx16, ((const uint16_t *)t->data)[off]);
        break;
    default:
        fprintf(f, "?");
        break;
    }
}

void nx_tensor_print(const nx_tensor_t *t, FILE *f, size_t max_elems)
{
    nx_tensor_iter_t it;
    size_t off, shown;

    assert(t != NULL && f != NULL);
    fprintf(f, "tensor '%s' %s ", t->name[0] != '\0' ? t->name : "(unnamed)",
            nx_dtype_name(t->dtype));
    print_dims(f, "shape", t->shape, t->ndim);
    fprintf(f, " ");
    print_strides(f, t->strides, t->ndim);
    fprintf(f, " elems=%zu contiguous=%s", t->elems,
            nx_tensor_is_contiguous(t) ? "yes" : "no");
    if (t->dtype == NX_F16 || t->dtype == NX_BF16) {
        fprintf(f, " (raw hex; decoder lands with the cpu/ backend)");
    }
    fprintf(f, "\n  ");

    shown = 0;
    nx_tensor_iter_init(&it, t);
    while (nx_tensor_iter_next(&it, &off) && shown < max_elems) {
        if (shown != 0) {
            fprintf(f, " ");
        }
        print_element(f, t, off);
        shown++;
    }
    if (shown < t->elems) {
        fprintf(f, " ... (%zu of %zu shown)", shown, t->elems);
    }
    fprintf(f, "\n");
}
