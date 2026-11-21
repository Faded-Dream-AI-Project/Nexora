/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software.
 *
 * Error handling: every fallible function returns an nx_status_t and, before
 * returning a failure, records a human-readable message retrievable with
 * nx_error_last(). See docs/CODING.md for the full convention. */

#ifndef NX_ERROR_H
#define NX_ERROR_H

#include "nx_common.h"

typedef enum {
    NX_OK = 0,          /* success */
    NX_E_NOMEM,         /* allocation failed (heap or arena) */
    NX_E_INVAL,         /* invalid argument: NULL, bad shape, bad key, ... */
    NX_E_RANGE,         /* index, dimension or numeric value out of range */
    NX_E_IO,            /* file or stream I/O failure */
    NX_E_FORMAT,        /* malformed input data or file */
    NX_E_UNSUPPORTED,   /* valid request, not supported by this build/backend */
    NX_E_OVERFLOW,      /* arithmetic overflow detected */
    NX_E_INTERNAL,      /* internal invariant violated (bug in Nexora) */
    NX_STATUS_COUNT
} nx_status_t;

/* Records `status` with a formatted message for the calling thread and
 * returns `status`, so callers write:
 *     return nx_error_set(NX_E_INVAL, "arena NULL");
 * The message is truncated at 511 bytes, never crashy. */
nx_status_t nx_error_set(nx_status_t status, const char *fmt, ...) NX_PRINTF(2, 3);

/* Message of the most recent nx_error_set() on this thread. Returns "" when
 * no error was recorded. Valid until the next nx_error_set() on the thread. */
const char *nx_error_last(void);

/* Status code of the most recent recorded error; NX_OK when none. */
nx_status_t nx_error_last_status(void);

/* Forgets any recorded error on this thread. */
void nx_error_clear(void);

/* Stable, short, lowercase description of a status code ("nomem", ...). */
const char *nx_status_str(nx_status_t status);

#endif /* NX_ERROR_H */
