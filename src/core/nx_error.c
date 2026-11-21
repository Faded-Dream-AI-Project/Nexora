/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software. */

#include "nx_error.h"

#include <stdarg.h>
#include <stdio.h>

#define NX_ERR_MSG_MAX 512

typedef struct {
    nx_status_t status;
    char msg[NX_ERR_MSG_MAX];
} nx_error_state_t;

static _Thread_local nx_error_state_t nx_t_err = { NX_OK, "" };

nx_status_t nx_error_set(nx_status_t status, const char *fmt, ...)
{
    va_list ap;
    int n;

    va_start(ap, fmt);
    n = vsnprintf(nx_t_err.msg, sizeof nx_t_err.msg, fmt, ap);
    va_end(ap);
    if (n < 0) {
        nx_t_err.msg[0] = '\0';
    }
    nx_t_err.status = status;
    return status;
}

const char *nx_error_last(void)
{
    return nx_t_err.msg;
}

nx_status_t nx_error_last_status(void)
{
    return nx_t_err.status;
}

void nx_error_clear(void)
{
    nx_t_err.status = NX_OK;
    nx_t_err.msg[0] = '\0';
}

const char *nx_status_str(nx_status_t status)
{
    static const char *const names[NX_STATUS_COUNT] = {
        [NX_OK] = "ok",           [NX_E_NOMEM] = "nomem", [NX_E_INVAL] = "inval",
        [NX_E_RANGE] = "range",   [NX_E_IO] = "io",       [NX_E_FORMAT] = "format",
        [NX_E_UNSUPPORTED] = "unsupported", [NX_E_OVERFLOW] = "overflow",
        [NX_E_INTERNAL] = "internal",
    };
    if ((int)status < 0 || status >= NX_STATUS_COUNT) {
        return "unknown";
    }
    return names[status];
}
