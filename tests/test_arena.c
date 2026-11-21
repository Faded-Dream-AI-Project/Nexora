/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software. */

#include "tests.h"

#include "core/nx_arena.h"

void test_arena(void)
{
    /* default init: one 64 KiB block */
    nx_arena_t a;
    NX_CHECK_STATUS(nx_arena_init(&a, 0), NX_OK);
    NX_CHECK_EQ_I64((int64_t)nx_arena_capacity(&a), 64 * 1024);
    NX_CHECK_EQ_I64((int64_t)nx_arena_blocks(&a), 1);
    NX_CHECK_EQ_I64((int64_t)nx_arena_used(&a), 0);

    /* bump allocations: distinct, aligned, non-overlapping */
    void *p1 = nx_arena_alloc(&a, 100, 0);
    void *p2 = nx_arena_alloc(&a, 100, 0);
    NX_CHECK(p1 != NULL && p2 != NULL && p1 != p2);
    NX_CHECK((uintptr_t)p1 % 16 == 0);
    NX_CHECK((uintptr_t)p2 % 16 == 0);
    memset(p1, 0xAB, 100);
    memset(p2, 0xCD, 100);
    NX_CHECK(((unsigned char *)p1)[0] == 0xAB && ((unsigned char *)p1)[99] == 0xAB);
    NX_CHECK(((unsigned char *)p2)[0] == 0xCD && ((unsigned char *)p2)[99] == 0xCD);
    NX_CHECK((int64_t)nx_arena_used(&a) >= 200);

    /* 64-byte alignment honored */
    void *p3 = nx_arena_alloc(&a, 32, 64);
    NX_CHECK(p3 != NULL);
    NX_CHECK((uintptr_t)p3 % 64 == 0);

    /* a request larger than the preferred block gets a dedicated block */
    size_t cap_before = nx_arena_capacity(&a);
    void *big = nx_arena_alloc(&a, 200 * 1024, 0);
    NX_CHECK(big != NULL);
    NX_CHECK_EQ_I64((int64_t)nx_arena_blocks(&a), 2);
    NX_CHECK((int64_t)nx_arena_capacity(&a) > (int64_t)cap_before);
    memset(big, 0x11, 200 * 1024);
    NX_CHECK(((unsigned char *)big)[0] == 0x11 && ((unsigned char *)big)[200 * 1024 - 1] == 0x11);

    /* peak accounting */
    NX_CHECK((int64_t)nx_arena_peak(&a) >= (int64_t)nx_arena_used(&a));

    /* reset keeps blocks for reuse */
    nx_arena_reset(&a);
    NX_CHECK_EQ_I64((int64_t)nx_arena_used(&a), 0);
    NX_CHECK_EQ_I64((int64_t)nx_arena_capacity(&a), (int64_t)cap_before + 200 * 1024);
    void *again = nx_arena_alloc(&a, 200 * 1024, 0);
    NX_CHECK(again != NULL);
    NX_CHECK_EQ_I64((int64_t)nx_arena_blocks(&a), 2); /* reused, not grown */
    nx_arena_destroy(&a);

    /* byte limit: growth beyond the cap fails cleanly */
    nx_arena_t c;
    NX_CHECK_STATUS(nx_arena_init(&c, 4096), NX_OK);
    nx_arena_set_limit(&c, 8192);
    NX_CHECK(nx_arena_alloc(&c, 4000, 0) != NULL); /* fits in the first block */
    NX_CHECK(nx_arena_alloc(&c, 8000, 0) == NULL); /* would need a new block */
    NX_CHECK(nx_arena_failed(&c));
    NX_CHECK(nx_error_last_status() == NX_E_NOMEM);
    NX_CHECK(strlen(nx_error_last()) > 0);
    NX_CHECK_EQ_I64((int64_t)nx_arena_blocks(&c), 1); /* nothing was added */
    nx_arena_clear_failed(&c);
    NX_CHECK(!nx_arena_failed(&c));
    /* the limit survives reset (first block not counted against it) */
    nx_arena_reset(&c);
    NX_CHECK(nx_arena_alloc(&c, 8000, 0) == NULL);
    NX_CHECK(nx_arena_failed(&c));
    nx_arena_destroy(&c);

    /* torture: many varied allocations, contents verified afterwards */
    nx_arena_t d;
    NX_CHECK_STATUS(nx_arena_init(&d, 1024), NX_OK);
    enum { N_PTRS = 200 };
    static unsigned char *ptrs[N_PTRS];
    static size_t sizes[N_PTRS];
    uint32_t rng = 12345;
    for (int i = 0; i < N_PTRS; i++) {
        rng = rng * 1664525u + 1013904223u;
        sizes[i] = 1 + (rng >> 8) % 512;
        ptrs[i] = nx_arena_alloc(&d, sizes[i], 0);
        NX_CHECK(ptrs[i] != NULL);
        memset(ptrs[i], i & 0xFF, sizes[i]);
    }
    bool all_ok = true;
    for (int i = 0; i < N_PTRS; i++) {
        for (size_t j = 0; j < sizes[i]; j++) {
            if (ptrs[i][j] != (unsigned char)(i & 0xFF)) {
                all_ok = false;
            }
        }
    }
    NX_CHECK(all_ok);
    nx_arena_destroy(&d);
}

void test_arena_marks(void)
{
    nx_arena_t b;
    NX_CHECK_STATUS(nx_arena_init(&b, 256), NX_OK);

    void *q1 = nx_arena_alloc(&b, 256, 0); /* fills block 1 */
    NX_CHECK(q1 != NULL);
    NX_CHECK_EQ_I64((int64_t)nx_arena_blocks(&b), 1);

    nx_arena_mark_t m = nx_arena_mark(&b); /* before q2: end of block 1 */
    void *q2 = nx_arena_alloc(&b, 256, 0); /* forces block 2 */
    NX_CHECK(q2 != NULL);
    NX_CHECK_EQ_I64((int64_t)nx_arena_blocks(&b), 2);
    size_t cap = nx_arena_capacity(&b);

    /* rewind: q2's bytes come back from block 2, not from malloc */
    nx_arena_rewind(&b, m);
    NX_CHECK_EQ_I64((int64_t)nx_arena_used(&b), 256); /* only q1 remains */
    void *q3 = nx_arena_alloc(&b, 256, 0);
    NX_CHECK(q3 != NULL);
    NX_CHECK(q3 == q2); /* exact reuse of the rewound block */
    NX_CHECK_EQ_I64((int64_t)nx_arena_blocks(&b), 2);
    NX_CHECK_EQ_I64((int64_t)nx_arena_capacity(&b), (int64_t)cap);

    /* nested marks: rewinding to an inner mark frees only the later range,
       rewinding to the outer mark frees both */
    nx_arena_mark_t outer = nx_arena_mark(&b);
    size_t used_outer = nx_arena_used(&b);
    nx_arena_alloc(&b, 100, 0);
    nx_arena_mark_t inner = nx_arena_mark(&b);
    nx_arena_alloc(&b, 100, 0);
    NX_CHECK((int64_t)nx_arena_used(&b) >= (int64_t)used_outer + 200);
    nx_arena_rewind(&b, inner);
    NX_CHECK_EQ_I64((int64_t)nx_arena_used(&b), (int64_t)used_outer + 100);
    nx_arena_rewind(&b, outer);
    NX_CHECK_EQ_I64((int64_t)nx_arena_used(&b), (int64_t)used_outer);
    nx_arena_destroy(&b);
}
