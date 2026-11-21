/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software. */

#include "tests.h"

#include "core/nx_alloc.h"

void test_alloc(void)
{
    static const size_t aligns[] = { 16, 32, 64, 4096 };
    static const size_t sizes[] = { 1, 3, 17, 100, 4097 };

    for (size_t i = 0; i < sizeof aligns / sizeof aligns[0]; i++) {
        for (size_t j = 0; j < sizeof sizes / sizeof sizes[0]; j++) {
            unsigned char *p = nx_alloc_aligned(sizes[j], aligns[i]);
            NX_CHECK(p != NULL);
            NX_CHECK((uintptr_t)p % aligns[i] == 0);
            memset(p, 0xA5, sizes[j]);
            NX_CHECK(p[0] == 0xA5 && p[sizes[j] - 1] == 0xA5);
            nx_free_aligned(p);
        }
    }

    /* two live allocations never overlap */
    unsigned char *p1 = nx_alloc_aligned(64, 64);
    unsigned char *p2 = nx_alloc_aligned(64, 64);
    NX_CHECK(p1 != NULL && p2 != NULL && p1 != p2);
    memset(p1, 1, 64);
    memset(p2, 2, 64);
    NX_CHECK(p1[0] == 1 && p2[63] == 2);
    nx_free_aligned(p1);
    nx_free_aligned(p2);

    /* degenerate and failing requests */
    NX_CHECK(nx_alloc_aligned(0, 16) == NULL);
    NX_CHECK(nx_alloc_aligned(SIZE_MAX / 2, 16) == NULL);

    nx_free_aligned(NULL); /* must not crash */
}
