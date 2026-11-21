/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software. */

#include "nx_arena.h"

#include "nx_error.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NX_ARENA_DEFAULT_BLOCK (64 * 1024)
#define NX_ARENA_DEFAULT_ALIGN 16

struct nx_arena_block {
    struct nx_arena_block *next;
    char *data;  /* NX_ALIGNMENT-aligned start of the usable region */
    size_t size; /* usable bytes */
    size_t used; /* bytes consumed, including alignment padding */
};

/* Invariant: every block strictly after `cur` has used == 0. */

static nx_arena_block_t *block_new(size_t usable)
{
    nx_arena_block_t *b = malloc(sizeof *b + NX_ALIGNMENT + usable);
    uintptr_t raw, aligned;

    if (b == NULL) {
        return NULL;
    }
    b->next = NULL;
    b->size = usable;
    b->used = 0;
    raw = (uintptr_t)(b + 1);
    aligned = (raw + NX_ALIGNMENT - 1) & ~(uintptr_t)(NX_ALIGNMENT - 1);
    b->data = (char *)aligned;
    return b;
}

/* Tries to bump-allocate in `b`; returns the pointer and the bytes consumed
 * (padding + size), or NULL if it does not fit. */
static void *block_try(nx_arena_block_t *b, size_t size, size_t align, size_t *consumed)
{
    uintptr_t addr;
    size_t avail, pad;

    if (size > b->size - b->used) {
        return NULL;
    }
    avail = b->size - b->used;
    addr = (uintptr_t)b->data + b->used;
    pad = (align - (addr & (align - 1))) & (align - 1);
    if (pad > avail - size) {
        return NULL;
    }
    b->used += pad + size;
    *consumed = pad + size;
    return b->data + b->used - size;
}

static void arena_note_used(nx_arena_t *a, size_t consumed)
{
    a->used += consumed;
    if (a->used > a->peak) {
        a->peak = a->used;
    }
}

nx_status_t nx_arena_init(nx_arena_t *a, size_t initial_block)
{
    nx_arena_block_t *b;
    size_t bs;

    if (a == NULL) {
        return nx_error_set(NX_E_INVAL, "arena NULL");
    }
    memset(a, 0, sizeof *a);
    bs = initial_block != 0 ? initial_block : NX_ARENA_DEFAULT_BLOCK;
    b = block_new(bs);
    if (b == NULL) {
        return nx_error_set(NX_E_NOMEM, "arena: first block (%zu bytes) failed", bs);
    }
    a->head = b;
    a->tail = b;
    a->cur = b;
    a->block_size = bs;
    a->capacity = bs;
    return NX_OK;
}

void nx_arena_destroy(nx_arena_t *a)
{
    nx_arena_block_t *b, *next;

    if (a == NULL) {
        return;
    }
    for (b = a->head; b != NULL; b = next) {
        next = b->next;
        free(b);
    }
    memset(a, 0, sizeof *a);
}

void nx_arena_set_limit(nx_arena_t *a, size_t total_bytes)
{
    assert(a != NULL);
    a->limit = total_bytes;
}

void *nx_arena_alloc(nx_arena_t *a, size_t size, size_t align)
{
    nx_arena_block_t *b;
    void *p;
    size_t consumed, need;

    assert(a != NULL);
    assert(size > 0);
    if (align == 0) {
        align = NX_ARENA_DEFAULT_ALIGN;
    }
    assert((align & (align - 1)) == 0);

    /* First: the current block and any reusable (rewound) blocks after it. */
    for (b = a->cur; b != NULL; b = b->next) {
        p = block_try(b, size, align, &consumed);
        if (p != NULL) {
            a->cur = b;
            arena_note_used(a, consumed);
            return p;
        }
    }

    /* Nothing fits: append a new block. Big requests get a dedicated block;
     * data is NX_ALIGNMENT-aligned so align <= NX_ALIGNMENT needs no pad. */
    need = size + (align > NX_ALIGNMENT ? align : 0);
    if (need < a->block_size) {
        need = a->block_size;
    }
    if (a->limit != 0 && a->capacity + need > a->limit) {
        a->failed = true;
        nx_error_set(NX_E_NOMEM,
                     "arena: byte limit %zu exceeded (capacity %zu, request %zu)",
                     a->limit, a->capacity, size);
        return NULL;
    }
    b = block_new(need);
    if (b == NULL) {
        a->failed = true;
        nx_error_set(NX_E_NOMEM, "arena: block malloc (%zu bytes) failed for %zu-byte request",
                     need, size);
        return NULL;
    }
    a->tail->next = b;
    a->tail = b;
    a->cur = b;
    a->capacity += need;

    p = block_try(b, size, align, &consumed);
    assert(p != NULL); /* fresh block was sized for this request */
    arena_note_used(a, consumed);
    return p;
}

nx_arena_mark_t nx_arena_mark(const nx_arena_t *a)
{
    nx_arena_mark_t m;

    assert(a != NULL);
    m.block = (struct nx_arena_block *)a->cur;
    m.used = a->cur->used;
    return m;
}

void nx_arena_rewind(nx_arena_t *a, nx_arena_mark_t mark)
{
    nx_arena_block_t *b;
    size_t u;

    assert(a != NULL);
    /* The mark must belong to this arena. */
    for (b = a->head; b != NULL; b = b->next) {
        if (b == (nx_arena_block_t *)mark.block) {
            break;
        }
    }
    assert(b != NULL && "mark from another arena");

    ((nx_arena_block_t *)mark.block)->used = mark.used;
    a->cur = (nx_arena_block_t *)mark.block;
    for (b = a->cur->next; b != NULL; b = b->next) {
        b->used = 0;
    }
    u = 0;
    for (b = a->head; b != NULL; b = b->next) {
        u += b->used;
    }
    a->used = u;
}

void nx_arena_reset(nx_arena_t *a)
{
    nx_arena_mark_t m;

    assert(a != NULL);
    m.block = (struct nx_arena_block *)a->head;
    m.used = 0;
    nx_arena_rewind(a, m);
}

size_t nx_arena_used(const nx_arena_t *a)
{
    assert(a != NULL);
    return a->used;
}

size_t nx_arena_capacity(const nx_arena_t *a)
{
    assert(a != NULL);
    return a->capacity;
}

size_t nx_arena_peak(const nx_arena_t *a)
{
    assert(a != NULL);
    return a->peak;
}

size_t nx_arena_blocks(const nx_arena_t *a)
{
    const nx_arena_block_t *b;
    size_t n = 0;

    assert(a != NULL);
    for (b = a->head; b != NULL; b = b->next) {
        n++;
    }
    return n;
}

bool nx_arena_failed(const nx_arena_t *a)
{
    assert(a != NULL);
    return a->failed;
}

void nx_arena_clear_failed(nx_arena_t *a)
{
    assert(a != NULL);
    a->failed = false;
}
