/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software. */

#include "nx_alloc.h"

#include <assert.h>
#include <errno.h>
#include <stdlib.h>

void *nx_alloc_aligned(size_t size, size_t alignment)
{
    void *ptr = NULL;
    int rc;

    assert(alignment != 0 && (alignment & (alignment - 1)) == 0);
    if (size == 0 || alignment < sizeof(void *)) {
        return NULL;
    }

    rc = posix_memalign(&ptr, alignment, size);
    if (rc != 0) {
        return NULL;
    }
    return ptr;
}

void nx_free_aligned(void *ptr)
{
    free(ptr);
}
