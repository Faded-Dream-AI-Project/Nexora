/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software. */

#include "tests.h"

#include "core/nx_arena.h"
#include "core/nx_tensor.h"

#include <stdlib.h>

void test_tensor(void)
{
    /* dtype table */
    NX_CHECK_EQ_I64((int64_t)nx_dtype_size(NX_F32), 4);
    NX_CHECK_EQ_I64((int64_t)nx_dtype_size(NX_F16), 2);
    NX_CHECK_EQ_I64((int64_t)nx_dtype_size(NX_BF16), 2);
    NX_CHECK_EQ_I64((int64_t)nx_dtype_size(NX_I32), 4);
    NX_CHECK_EQ_I64((int64_t)nx_dtype_size(NX_I16), 2);
    NX_CHECK_EQ_I64((int64_t)nx_dtype_size(NX_I8), 1);
    NX_CHECK_EQ_I64((int64_t)nx_dtype_size(NX_U8), 1);
    NX_CHECK_EQ_I64((int64_t)nx_dtype_size((nx_dtype_t)99), 0);
    NX_CHECK_EQ_STR(nx_dtype_name(NX_F32), "f32");
    NX_CHECK_EQ_STR(nx_dtype_name(NX_BF16), "bf16");
    NX_CHECK(nx_dtype_from_name("bf16") == NX_BF16);
    NX_CHECK(nx_dtype_from_name("nope") == NX_DTYPE_COUNT);
    NX_CHECK(nx_dtype_from_name(NULL) == NX_DTYPE_COUNT);

    nx_arena_t a;
    NX_CHECK_STATUS(nx_arena_init(&a, 1 << 20), NX_OK);

    /* create 2x3 f32 */
    int32_t shape[2] = { 2, 3 };
    nx_tensor_t *t = nx_tensor_create(&a, NX_F32, 2, shape);
    NX_CHECK(t != NULL);
    NX_CHECK(t->ndim == 2);
    NX_CHECK(t->shape[0] == 2 && t->shape[1] == 3);
    NX_CHECK_EQ_I64((int64_t)t->elems, 6);
    NX_CHECK_EQ_I64((int64_t)t->strides[0], 3);
    NX_CHECK_EQ_I64((int64_t)t->strides[1], 1);
    NX_CHECK(nx_tensor_is_contiguous(t));
    NX_CHECK(t->storage == &a);
    NX_CHECK((uintptr_t)t->data % 64 == 0);
    NX_CHECK_EQ_I64((int64_t)nx_tensor_numel(t), 6);

    /* fill + indexed get/set */
    nx_tensor_fill_f32(t, 3.5f);
    int32_t at00[2] = { 0, 0 };
    int32_t at11[2] = { 1, 1 };
    NX_CHECK_EQ_F64(nx_tensor_get_f32_at(t, at00), 3.5, 1e-6);
    nx_tensor_set_f32_at(t, at11, 9.0);
    NX_CHECK_EQ_F64(nx_tensor_get_f32_at(t, at11), 9.0, 1e-6);

    /* iterator walks in row-major logical order */
    float *raw = t->data;
    for (int i = 0; i < 6; i++) {
        raw[i] = (float)i;
    }
    nx_tensor_iter_t it;
    size_t off;
    int seen = 0;
    float expect = 0.0f;
    nx_tensor_iter_init(&it, t);
    while (nx_tensor_iter_next(&it, &off)) {
        NX_CHECK_EQ_F64(((const float *)t->data)[off], expect, 1e-6);
        seen++;
        expect += 1.0f;
    }
    NX_CHECK_EQ_I64(seen, 6);

    /* scalars */
    nx_tensor_t *s = nx_tensor_create(&a, NX_F32, 0, NULL);
    NX_CHECK(s != NULL && s->ndim == 0);
    NX_CHECK_EQ_I64((int64_t)s->elems, 1);
    nx_tensor_set_f32_at(s, NULL, 2.5f);
    NX_CHECK_EQ_F64(nx_tensor_get_f32_at(s, NULL), 2.5, 1e-6);
    NX_CHECK(nx_tensor_is_contiguous(s));
    int scalar_iters = 0;
    nx_tensor_iter_init(&it, s);
    while (nx_tensor_iter_next(&it, &off)) {
        NX_CHECK_EQ_I64(off, 0);
        scalar_iters++;
    }
    NX_CHECK_EQ_I64(scalar_iters, 1);

    /* rejections: rank, zero dims, byte overflow, bad dtype, NULL arena */
    int32_t shape9[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    NX_CHECK(nx_tensor_create(&a, NX_F32, 9, shape9) == NULL);
    NX_CHECK(nx_error_last_status() == NX_E_RANGE);
    int32_t zshape[2] = { 2, 0 };
    NX_CHECK(nx_tensor_create(&a, NX_F32, 2, zshape) == NULL);
    NX_CHECK(nx_error_last_status() == NX_E_INVAL);
    int32_t ov[3] = { (int32_t)1 << 21, (int32_t)1 << 21, (int32_t)1 << 21 };
    NX_CHECK(nx_tensor_create(&a, NX_F32, 3, ov) == NULL); /* 2^63 elems * 4 bytes */
    NX_CHECK(nx_error_last_status() == NX_E_OVERFLOW);
    NX_CHECK(nx_tensor_create(&a, (nx_dtype_t)77, 1, shape) == NULL);
    NX_CHECK(nx_tensor_create(NULL, NX_F32, 2, shape) == NULL);

    /* i32 tensors: fill/set/get, plus narrowing paths */
    int32_t ishape[2] = { 2, 2 };
    nx_tensor_t *ti = nx_tensor_create(&a, NX_I32, 2, ishape);
    NX_CHECK(ti != NULL);
    nx_tensor_fill_i32(ti, 7);
    int32_t at01[2] = { 0, 1 };
    NX_CHECK_EQ_I64(nx_tensor_get_i32_at(ti, at01), 7);
    nx_tensor_set_i32_at(ti, at01, -3);
    NX_CHECK_EQ_I64(nx_tensor_get_i32_at(ti, at01), -3);

    nx_tensor_t *t16 = nx_tensor_create(&a, NX_I16, 2, ishape);
    NX_CHECK(t16 != NULL);
    nx_tensor_fill_i32(t16, 300);
    NX_CHECK_EQ_I64(nx_tensor_get_i32_at(t16, at00), 300);
    nx_tensor_set_i32_at(t16, at00, 40000); /* out of int16 range: wraps */
    NX_CHECK_EQ_I64(nx_tensor_get_i32_at(t16, at00), 40000 - 65536);

    /* names */
    nx_tensor_set_name(t, "weights");
    NX_CHECK_EQ_STR(t->name, "weights");
    static char longname[64];
    memset(longname, 'n', sizeof longname - 1);
    longname[sizeof longname - 1] = '\0';
    nx_tensor_set_name(t, longname);
    NX_CHECK_EQ_I64((int64_t)strlen(t->name), NX_TENSOR_NAME_MAX - 1);

    /* debug print: shape, dtype and values present */
    nx_tensor_set_name(t, "w");
    char *out = NULL;
    size_t outlen = 0;
    FILE *mf = open_memstream(&out, &outlen);
    NX_CHECK(mf != NULL);
    nx_tensor_print(t, mf, 100);
    fclose(mf);
    NX_CHECK(out != NULL);
    NX_CHECK(strstr(out, "f32") != NULL);
    NX_CHECK(strstr(out, "shape=[2,3]") != NULL);
    NX_CHECK(strstr(out, "'w'") != NULL);
    NX_CHECK(strstr(out, "5") != NULL); /* value from the 0..5 fill above */
    free(out);

    /* truncation marker when max_elems < elems */
    mf = open_memstream(&out, &outlen);
    NX_CHECK(mf != NULL);
    nx_tensor_print(t, mf, 3);
    fclose(mf);
    NX_CHECK(strstr(out, "of 6 shown") != NULL);
    free(out);

    nx_arena_destroy(&a);
}

void test_tensor_views(void)
{
    nx_arena_t a;
    NX_CHECK_STATUS(nx_arena_init(&a, 1 << 20), NX_OK);

    int32_t shape[2] = { 2, 3 };
    nx_tensor_t *t = nx_tensor_create(&a, NX_F32, 2, shape);
    NX_CHECK(t != NULL);
    float *raw = t->data;
    for (int i = 0; i < 6; i++) {
        raw[i] = (float)i;
    }

    /* reshape with -1 inference, data untouched */
    int32_t rshape[3] = { 3, 2, -1 };
    nx_tensor_t *r = nx_tensor_reshape(&a, t, 3, rshape);
    NX_CHECK(r != NULL);
    NX_CHECK(r->shape[0] == 3 && r->shape[1] == 2 && r->shape[2] == 1);
    NX_CHECK(r->data == t->data);
    NX_CHECK(nx_tensor_is_contiguous(r));
    int32_t at110[3] = { 1, 1, 0 };
    NX_CHECK_EQ_F64(nx_tensor_get_f32_at(r, at110), 3.0, 1e-6);

    /* reshape rejections */
    int32_t bad1[1] = { 7 };
    NX_CHECK(nx_tensor_reshape(&a, t, 1, bad1) == NULL); /* wrong count */
    int32_t bad2[2] = { -1, -1 };
    NX_CHECK(nx_tensor_reshape(&a, t, 2, bad2) == NULL); /* two -1 */
    int32_t bad3[2] = { 4, 5 }; /* 20 != 6 */
    NX_CHECK(nx_tensor_reshape(&a, t, 2, bad3) == NULL);
    int32_t bad4[2] = { 4, 0 };
    NX_CHECK(nx_tensor_reshape(&a, t, 2, bad4) == NULL);

    /* permute = 2D transpose */
    int32_t perm[2] = { 1, 0 };
    nx_tensor_t *tr = nx_tensor_permute(&a, t, perm);
    NX_CHECK(tr != NULL);
    NX_CHECK(tr->shape[0] == 3 && tr->shape[1] == 2);
    NX_CHECK(!nx_tensor_is_contiguous(tr));
    int32_t at21[2] = { 2, 1 };
    NX_CHECK_EQ_F64(nx_tensor_get_f32_at(tr, at21), 5.0, 1e-6); /* t[1][2] */
    int32_t badperm[2] = { 0, 0 };
    NX_CHECK(nx_tensor_permute(&a, t, badperm) == NULL);

    /* permute of non-contiguous is fine (views of views) */
    int32_t ident[2] = { 0, 1 };
    nx_tensor_t *tr_back = nx_tensor_permute(&a, tr, ident);
    NX_CHECK(tr_back != NULL);
    int32_t at21b[2] = { 2, 1 };
    NX_CHECK_EQ_F64(nx_tensor_get_f32_at(tr_back, at21b), 5.0, 1e-6);

    /* narrow: rows and columns */
    nx_tensor_t *row = nx_tensor_narrow(&a, t, 0, 1, 1); /* second row */
    NX_CHECK(row != NULL);
    NX_CHECK(row->shape[0] == 1 && row->shape[1] == 3);
    NX_CHECK_EQ_I64((int64_t)row->elems, 3);
    NX_CHECK(nx_tensor_is_contiguous(row)); /* [1,3] with strides [3,1] */
    int32_t at00b[2] = { 0, 0 };
    NX_CHECK_EQ_F64(nx_tensor_get_f32_at(row, at00b), 3.0, 1e-6);

    nx_tensor_t *col = nx_tensor_narrow(&a, t, 1, 1, 1); /* middle column */
    NX_CHECK(col != NULL);
    NX_CHECK(col->shape[0] == 2 && col->shape[1] == 1);
    NX_CHECK(!nx_tensor_is_contiguous(col)); /* strides [3,1] for [2,1] */
    int32_t at10[2] = { 1, 0 };
    NX_CHECK_EQ_F64(nx_tensor_get_f32_at(col, at10), 4.0, 1e-6);

    NX_CHECK(nx_tensor_narrow(&a, t, 0, 1, 2) == NULL); /* out of range */
    NX_CHECK(nx_tensor_narrow(&a, t, 0, 2, 1) == NULL);
    NX_CHECK(nx_tensor_narrow(&a, t, 2, 0, 1) == NULL); /* bad dim */

    /* reshape of a non-contiguous view must be rejected */
    NX_CHECK(nx_tensor_reshape(&a, tr, 1, (int32_t[]){ 6 }) == NULL);

    /* wrap borrowed memory, default and custom strides */
    float mem[4] = { 10, 20, 30, 40 };
    nx_tensor_t *w = nx_tensor_wrap(&a, mem, NX_F32, 2, (int32_t[]){ 2, 2 }, NULL);
    NX_CHECK(w != NULL);
    NX_CHECK(w->storage == NULL);
    int32_t at11b[2] = { 1, 1 };
    NX_CHECK_EQ_F64(nx_tensor_get_f32_at(w, at11b), 40.0, 1e-6);
    size_t colstr[2] = { 1, 2 }; /* column-major [2,2] */
    nx_tensor_t *wc = nx_tensor_wrap(&a, mem, NX_F32, 2, (int32_t[]){ 2, 2 }, colstr);
    int32_t at01c[2] = { 0, 1 };
    NX_CHECK_EQ_F64(nx_tensor_get_f32_at(wc, at01c), 30.0, 1e-6);
    NX_CHECK(nx_tensor_wrap(&a, NULL, NX_F32, 1, (int32_t[]){ 1 }, NULL) == NULL);

    /* alias: full copy of the view, same data */
    nx_tensor_t *al = nx_tensor_alias(&a, w);
    NX_CHECK(al != NULL && al->data == w->data);
    NX_CHECK_EQ_I64((int64_t)al->elems, (int64_t)w->elems);

    /* views into views share the storage pointer chain */
    nx_tensor_t *row_of_tr = nx_tensor_narrow(&a, tr, 1, 0, 1);
    NX_CHECK(row_of_tr != NULL);
    NX_CHECK(row_of_tr->storage == t->storage);

    nx_arena_destroy(&a);
}
