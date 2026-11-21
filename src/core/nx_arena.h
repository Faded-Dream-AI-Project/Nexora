/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software.
 *
 * Region (bump) allocator over a list of blocks. Tensors, autograd graph
 * nodes and per-step scratch all live in arenas: create in bulk, rewind in
 * O(blocks), never free individually. A mark/rewind pair gives nested
 * temporary lifetimes (e.g. one micro-batch of a training step).
 *
 * Blocks are kept and reused after rewind; the arena only grows (until
 * nx_arena_set_limit stops it). NOT thread-safe: one arena per thread.
 */

#ifndef NX_ARENA_H
#define NX_ARENA_H

#include "nx_common.h"
#include "nx_error.h"

/* Private block type; defined in nx_arena.c. */
typedef struct nx_arena_block nx_arena_block_t;

/* The struct is visible so arenas can live on the stack or be embedded (a
 * per-thread scratch arena needs no extra malloc). Fields after `block_size`
 * are internal bookkeeping: never write them directly. */
typedef struct nx_arena {
    nx_arena_block_t *head;
    nx_arena_block_t *tail;
    nx_arena_block_t *cur; /* allocation happens at cur, then later blocks */
    size_t block_size;     /* preferred size for new blocks */
    size_t limit;          /* capacity cap, 0 = unlimited */
    size_t capacity;       /* sum of block sizes */
    size_t used;           /* handed-out bytes including padding */
    size_t peak;           /* high-water mark of used */
    bool failed;           /* an allocation failed since init/last clear */
} nx_arena_t;

/* A snapshot of arena state; pass to nx_arena_rewind(). Obtain with
 * nx_arena_mark(). Marks stay valid until the arena is destroyed. */
typedef struct nx_arena_mark {
    struct nx_arena_block *block; /* opaque */
    size_t used;
} nx_arena_mark_t;

/* Creates an arena whose first (and preferred) block is `initial_block`
 * usable bytes. Pass 0 for a 64 KiB default. The first block is allocated
 * immediately; a byte limit, if wanted, is set afterwards with
 * nx_arena_set_limit() and applies to further growth only. */
nx_status_t nx_arena_init(nx_arena_t *a, size_t initial_block);

/* Frees every block. Safe to call on a zeroed or failed-init arena. */
void nx_arena_destroy(nx_arena_t *a);

/* Caps total arena capacity at `total_bytes` (0 = no cap). A cap smaller than
 * the current capacity simply prevents all further growth. */
void nx_arena_set_limit(nx_arena_t *a, size_t total_bytes);

/* Bump-allocates `size` bytes aligned to `align` (power of two; 0 = default
 * 16). Alignment above NX_ALIGNMENT is honored with extra padding. Returns
 * NULL and records an error when the byte limit or malloc fails; the arena's
 * latched failure flag is then set (nx_arena_failed()). size == 0 is a caller
 * bug and asserted. */
void *nx_arena_alloc(nx_arena_t *a, size_t size, size_t align);

/* Snapshot for rewind. */
nx_arena_mark_t nx_arena_mark(const nx_arena_t *a);

/* Returns the arena to the mark. Everything allocated after the mark becomes
 * free again; blocks are kept for reuse (no syscall churn). The mark itself
 * stays valid. Marks taken before the one rewound to remain valid too. */
void nx_arena_rewind(nx_arena_t *a, nx_arena_mark_t mark);

/* rewind() to the very beginning. */
void nx_arena_reset(nx_arena_t *a);

/* Live usage counters. `used` counts handed-out bytes including alignment
 * padding; `capacity` sums usable bytes of all blocks. */
size_t nx_arena_used(const nx_arena_t *a);
size_t nx_arena_capacity(const nx_arena_t *a);
size_t nx_arena_peak(const nx_arena_t *a);
size_t nx_arena_blocks(const nx_arena_t *a);

/* True when any allocation has failed since init or the last clear. The flag
 * is informational; allocation keeps being attempted after it is set. */
bool nx_arena_failed(const nx_arena_t *a);
void nx_arena_clear_failed(nx_arena_t *a);

#endif /* NX_ARENA_H */
