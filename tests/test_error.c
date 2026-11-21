/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software. */

#include "tests.h"

void test_error(void)
{
    nx_error_clear();
    NX_CHECK_EQ_STR(nx_error_last(), "");
    NX_CHECK(nx_error_last_status() == NX_OK);

    /* set returns its own code for direct `return nx_error_set(...)` use */
    nx_status_t st = nx_error_set(NX_E_INVAL, "bad thing %d", 42);
    NX_CHECK_STATUS(st, NX_E_INVAL);
    NX_CHECK_EQ_STR(nx_error_last(), "bad thing 42");
    NX_CHECK(nx_error_last_status() == NX_E_INVAL);

    /* every status has a distinct non-empty short name */
    for (int c = 0; c < (int)NX_STATUS_COUNT; c++) {
        NX_CHECK(nx_status_str((nx_status_t)c) != NULL);
        NX_CHECK(strlen(nx_status_str((nx_status_t)c)) > 0);
    }
    for (int c = 0; c < (int)NX_STATUS_COUNT; c++) {
        for (int d = c + 1; d < (int)NX_STATUS_COUNT; d++) {
            NX_CHECK(strcmp(nx_status_str((nx_status_t)c), nx_status_str((nx_status_t)d)) !=
                     0);
        }
    }
    NX_CHECK_EQ_STR(nx_status_str((nx_status_t)999), "unknown");

    /* long messages truncate instead of overflowing */
    static char big[2048];
    memset(big, 'x', sizeof big - 1);
    big[sizeof big - 1] = '\0';
    nx_error_set(NX_E_RANGE, "%s", big);
    NX_CHECK_EQ_I64((int64_t)strlen(nx_error_last()), 511);

    nx_error_clear();
    NX_CHECK_EQ_STR(nx_error_last(), "");
    NX_CHECK(nx_error_last_status() == NX_OK);
}
