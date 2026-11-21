/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software. */

#include "nx_log.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define NX_LOG_MSG_MAX 1024

static nx_log_level_t nx_g_level = NX_LOG_INFO;
static nx_log_sink_fn nx_g_sink = NULL; /* NULL = default stderr sink */
static void *nx_g_sink_ctx = NULL;

void nx_log_set_level(nx_log_level_t level)
{
    if ((int)level >= 0 && level < NX_LOG_LEVEL_COUNT) {
        nx_g_level = level;
    }
}

nx_log_level_t nx_log_get_level(void)
{
    return nx_g_level;
}

void nx_log_set_sink(nx_log_sink_fn sink, void *ctx)
{
    nx_g_sink = sink;
    nx_g_sink_ctx = ctx;
}

const char *nx_log_level_str(nx_log_level_t level)
{
    static const char *const names[NX_LOG_LEVEL_COUNT] = {
        [NX_LOG_TRACE] = "TRACE", [NX_LOG_DEBUG] = "DEBUG", [NX_LOG_INFO] = "INFO",
        [NX_LOG_WARN] = "WARN",   [NX_LOG_ERROR] = "ERROR",
    };
    if ((int)level < 0 || level >= NX_LOG_LEVEL_COUNT) {
        return "?";
    }
    return names[level];
}

static const char *level_color(nx_log_level_t level)
{
    switch (level) {
    case NX_LOG_TRACE:
    case NX_LOG_DEBUG:
        return "\033[2m";
    case NX_LOG_WARN:
        return "\033[33m";
    case NX_LOG_ERROR:
        return "\033[31m";
    default:
        return "";
    }
}

static void stderr_sink(nx_log_level_t level, const char *module, const char *msg,
                        void *ctx)
{
    struct timespec ts;
    struct tm tm_buf;
    char stamp[32];
    char time_str[64];
    const char *color = "";
    const char *reset = "";

    (void)ctx;
    clock_gettime(CLOCK_REALTIME, &ts);
    gmtime_r(&ts.tv_sec, &tm_buf);
    strftime(stamp, sizeof stamp, "%H:%M:%S", &tm_buf);
    snprintf(time_str, sizeof time_str, "%s.%03ld", stamp, (long)(ts.tv_nsec / 1000000));

    if (isatty(fileno(stderr))) {
        color = level_color(level);
        reset = "\033[0m";
    }
    fprintf(stderr, "%s[%s] %-5s [%s]%s %s\n", color, time_str, nx_log_level_str(level),
            module != NULL ? module : "-", reset, msg);
}

void nx_log(nx_log_level_t level, const char *module, const char *fmt, ...)
{
    char msg[NX_LOG_MSG_MAX];
    va_list ap;
    int n;

    if ((int)level < 0 || level >= NX_LOG_LEVEL_COUNT) {
        return;
    }
    if (level < nx_g_level) {
        return;
    }

    va_start(ap, fmt);
    n = vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    if (n < 0) {
        msg[0] = '\0';
    }

    if (nx_g_sink != NULL) {
        nx_g_sink(level, module, msg, nx_g_sink_ctx);
    } else {
        stderr_sink(level, module, msg, NULL);
    }
}
