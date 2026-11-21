/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software. */

#include "tests.h"

#include "core/nx_log.h"

/* Counting sink so tests stay quiet and verifiable. */
#define NX_LOG_MSG_TEST_MAX 128

static int sink_calls;
static nx_log_level_t sink_last_level;
static char sink_last_msg[NX_LOG_MSG_TEST_MAX];
static char sink_last_mod[32];

static void counting_sink(nx_log_level_t level, const char *module, const char *msg,
                          void *ctx)
{
    int *count = ctx;
    sink_calls++;
    if (count != NULL) {
        (*count)++;
    }
    sink_last_level = level;
    snprintf(sink_last_mod, sizeof sink_last_mod, "%s", module != NULL ? module : "-");
    snprintf(sink_last_msg, sizeof sink_last_msg, "%s", msg);
}

void test_log(void)
{
    int ctx = 0;

    /* default state */
    NX_CHECK(nx_log_get_level() == NX_LOG_INFO);

    /* level names */
    NX_CHECK_EQ_STR(nx_log_level_str(NX_LOG_TRACE), "TRACE");
    NX_CHECK_EQ_STR(nx_log_level_str(NX_LOG_DEBUG), "DEBUG");
    NX_CHECK_EQ_STR(nx_log_level_str(NX_LOG_INFO), "INFO");
    NX_CHECK_EQ_STR(nx_log_level_str(NX_LOG_WARN), "WARN");
    NX_CHECK_EQ_STR(nx_log_level_str(NX_LOG_ERROR), "ERROR");

    /* filtering and delivery through a custom sink */
    nx_log_set_sink(counting_sink, &ctx);
    nx_log_set_level(NX_LOG_WARN);
    sink_calls = 0;

    nx_log(NX_LOG_INFO, "mod", "this must be dropped %d", 1);
    NX_CHECK_EQ_I64(sink_calls, 0);
    NX_CHECK_EQ_I64(ctx, 0);

    nx_log(NX_LOG_ERROR, "modname", "value %d", 42);
    NX_CHECK_EQ_I64(sink_calls, 1);
    NX_CHECK_EQ_I64(ctx, 1);
    NX_CHECK(sink_last_level == NX_LOG_ERROR);
    NX_CHECK_EQ_STR(sink_last_mod, "modname");
    NX_CHECK_EQ_STR(sink_last_msg, "value 42");

    /* oversized messages are truncated, never crash */
    static char big[2048];
    memset(big, 'y', sizeof big - 1);
    big[sizeof big - 1] = '\0';
    nx_log(NX_LOG_ERROR, "mod", "%s", big);
    NX_CHECK_EQ_I64(sink_calls, 2);
    NX_CHECK_EQ_I64((int64_t)strlen(sink_last_msg), NX_LOG_MSG_TEST_MAX - 1);

    /* invalid levels are ignored entirely */
    nx_log((nx_log_level_t)99, "mod", "dropped");
    nx_log((nx_log_level_t)-1, "mod", "dropped");
    NX_CHECK_EQ_I64(sink_calls, 2);

    /* restore defaults (other tests use the default sink) */
    nx_log_set_sink(NULL, NULL);
    nx_log_set_level(NX_LOG_INFO);
    NX_CHECK(nx_log_get_level() == NX_LOG_INFO);
}
