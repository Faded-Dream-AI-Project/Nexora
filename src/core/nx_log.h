/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software.
 *
 * Leveled logging with a replaceable sink. Logs go to stderr by default so
 * CLI results on stdout stay pipeable. */

#ifndef NX_LOG_H
#define NX_LOG_H

#include "nx_common.h"

typedef enum {
    NX_LOG_TRACE = 0,
    NX_LOG_DEBUG = 1,
    NX_LOG_INFO = 2,
    NX_LOG_WARN = 3,
    NX_LOG_ERROR = 4,
    NX_LOG_LEVEL_COUNT = 5
} nx_log_level_t;

/* Sink receives fully formatted message text (no prefix). Set ctx when
 * installing. Set sink=NULL to restore the default stderr sink. */
typedef void (*nx_log_sink_fn)(nx_log_level_t level, const char *module,
                               const char *msg, void *ctx);

/* Sets the minimum level that is emitted (default: NX_LOG_INFO). */
void nx_log_set_level(nx_log_level_t level);
nx_log_level_t nx_log_get_level(void);

/* Installs a sink; sink=NULL restores the default stderr sink. */
void nx_log_set_sink(nx_log_sink_fn sink, void *ctx);

/* Emits one record. Messages below the current level are dropped here. */
void nx_log(nx_log_level_t level, const char *module, const char *fmt, ...)
    NX_PRINTF(3, 4);

/* Uppercase name ("WARN", ...) for a level. */
const char *nx_log_level_str(nx_log_level_t level);

#endif /* NX_LOG_H */
