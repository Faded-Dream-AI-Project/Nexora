/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software.
 *
 * Shared definitions for every Nexora module. This header MUST be the first
 * thing included (directly or via another nx_*.h header) in every translation
 * unit: it defines the POSIX feature-test macro required because we compile
 * with strict -std=c11. */

#ifndef NX_COMMON_H
#define NX_COMMON_H

#define _POSIX_C_SOURCE 200809L

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define NEXORA_VERSION_MAJOR 0
#define NEXORA_VERSION_MINOR 1
#define NEXORA_VERSION_PATCH 0
#define NEXORA_VERSION_STRING "0.1.0"

/* Alignment used for the start of every arena-backed allocation region
 * (one x86 cache line; also satisfies every dtype and AVX2 requirement). */
#define NX_ALIGNMENT 64

/* Maximum tensor rank. A [batch, heads, seq, head_dim] attention layout fits
 * comfortably; anything beyond is a bug in caller logic. */
#define NX_MAX_DIMS 8

/* Printf-style argument checking on GCC/Clang; expands to nothing elsewhere. */
#if defined(__GNUC__)
#define NX_PRINTF(fmt_arg, first_arg) __attribute__((format(printf, fmt_arg, first_arg)))
#else
#define NX_PRINTF(fmt_arg, first_arg)
#endif

/* Element types supported by tensors. F16/BF16 are storage formats this early:
 * arithmetic on them is emulated and lands with the cpu/ backend (phase 1)
 * and optim/ (phase 4). */
typedef enum {
    NX_F32 = 0,
    NX_F16 = 1,
    NX_BF16 = 2,
    NX_I32 = 3,
    NX_I16 = 4,
    NX_I8 = 5,
    NX_U8 = 6,
    NX_DTYPE_COUNT = 7
} nx_dtype_t;

/* Size in bytes of one element of `dtype`. */
size_t nx_dtype_size(nx_dtype_t dtype);

/* Short lowercase name ("f32", "i8", ...) valid for config files and logs. */
const char *nx_dtype_name(nx_dtype_t dtype);

/* Parses a nx_dtype_name() string; NX_DTYPE_COUNT if unknown. */
nx_dtype_t nx_dtype_from_name(const char *name);

/* Overflow-checked size_t arithmetic. Each returns true (and leaves *out
 * untouched) when the operation would overflow; false on success. */
bool nx_size_mul_overflows(size_t a, size_t b, size_t *out);
bool nx_size_add_overflows(size_t a, size_t b, size_t *out);

#endif /* NX_COMMON_H */
