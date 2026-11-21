/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software. */

#include "tests.h"

int nx_t_checks = 0;
int nx_t_failures = 0;

typedef struct {
    const char *name;
    void (*fn)(void);
} test_case_t;

static const test_case_t k_tests[] = {
    { "alloc", test_alloc },
    { "arena", test_arena },
    { "arena/marks", test_arena_marks },
    { "error", test_error },
    { "log", test_log },
    { "tensor", test_tensor },
    { "tensor/views", test_tensor_views },
    { "config/basic", test_config_basic },
    { "config/errors", test_config_errors },
};

int main(void)
{
    size_t total = sizeof k_tests / sizeof k_tests[0];
    int failed_suites = 0;

    printf("nexora %s core tests\n", NEXORA_VERSION_STRING);
    for (size_t i = 0; i < total; i++) {
        int before = nx_t_failures;
        k_tests[i].fn();
        if (nx_t_failures == before) {
            printf("ok   %s\n", k_tests[i].name);
        } else {
            printf("FAIL %s\n", k_tests[i].name);
            failed_suites++;
        }
    }
    printf("--- %d checks, %d failures, %zu suites\n", nx_t_checks, nx_t_failures,
           total);
    return failed_suites != 0 ? 1 : 0;
}
