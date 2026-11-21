/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software.
 *
 * Heap aligned allocation for long-lived buffers (arenas handle the rest).
 * The POSIX implementation lives here; a Windows variant will slot in behind
 * the same interface later (see docs/ARCHITECTURE.md, platform isolation). */

#ifndef NX_ALLOC_H
#define NX_ALLOC_H

#include "nx_common.h"

/* Allocates `size` bytes aligned to `alignment` bytes. `alignment` must be a
 * power of two >= sizeof(void *). Returns NULL on failure or when size == 0
 * (callers must treat size == 0 as a bug in their code, not an empty buffer).
 * Does NOT record an nx_error_set() message: the caller has the context. */
void *nx_alloc_aligned(size_t size, size_t alignment);

/* Frees a pointer returned by nx_alloc_aligned(); NULL is allowed. */
void nx_free_aligned(void *ptr);

#endif /* NX_ALLOC_H */
