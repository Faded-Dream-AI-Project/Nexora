/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software. */

#include "nx_common.h"

#include <string.h>

size_t nx_dtype_size(nx_dtype_t dtype)
{
    static const size_t sizes[NX_DTYPE_COUNT] = {
        [NX_F32] = 4, [NX_F16] = 2, [NX_BF16] = 2, [NX_I32] = 4,
        [NX_I16] = 2, [NX_I8] = 1,  [NX_U8] = 1,
    };
    if ((int)dtype < 0 || dtype >= NX_DTYPE_COUNT) {
        return 0;
    }
    return sizes[dtype];
}

const char *nx_dtype_name(nx_dtype_t dtype)
{
    static const char *const names[NX_DTYPE_COUNT] = {
        [NX_F32] = "f32", [NX_F16] = "f16", [NX_BF16] = "bf16", [NX_I32] = "i32",
        [NX_I16] = "i16", [NX_I8] = "i8",   [NX_U8] = "u8",
    };
    if ((int)dtype < 0 || dtype >= NX_DTYPE_COUNT) {
        return "?";
    }
    return names[dtype];
}

nx_dtype_t nx_dtype_from_name(const char *name)
{
    if (name == NULL) {
        return NX_DTYPE_COUNT;
    }
    for (int d = 0; d < NX_DTYPE_COUNT; d++) {
        if (strcmp(name, nx_dtype_name((nx_dtype_t)d)) == 0) {
            return (nx_dtype_t)d;
        }
    }
    return NX_DTYPE_COUNT;
}

bool nx_size_mul_overflows(size_t a, size_t b, size_t *out)
{
#if defined(__builtin_mul_overflow)
    size_t r = 0;
    if (__builtin_mul_overflow(a, b, &r)) {
        return true;
    }
    *out = r;
    return false;
#else
    if (a != 0 && b > SIZE_MAX / a) {
        return true;
    }
    *out = a * b;
    return false;
#endif
}

bool nx_size_add_overflows(size_t a, size_t b, size_t *out)
{
#if defined(__builtin_add_overflow)
    size_t r = 0;
    if (__builtin_add_overflow(a, b, &r)) {
        return true;
    }
    *out = r;
    return false;
#else
    if (b > SIZE_MAX - a) {
        return true;
    }
    *out = a + b;
    return false;
#endif
}
