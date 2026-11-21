/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software.
 *
 * Test harness shared by all test_*.c files. Check macros are allowed by
 * docs/CODING.md: they print file:line on failure; they do not hide control
 * flow beyond a single if. Add a new test: write tests/test_x.c, declare it
 * here, register it in tests/main.c. */

#ifndef TESTS_H
#define TESTS_H

#include "core/nx_common.h"
#include "core/nx_error.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* Counters, defined in tests/main.c. */
extern int nx_t_checks;
extern int nx_t_failures;

#define NX_CHECK(cond)                                                          \
    do {                                                                        \
        nx_t_checks++;                                                          \
        if (!(cond)) {                                                          \
            nx_t_failures++;                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
        }                                                                       \
    } while (0)

#define NX_CHECK_EQ_I64(a_, b_)                                                  \
    do {                                                                        \
        int64_t va_ = (int64_t)(a_);                                            \
        int64_t vb_ = (int64_t)(b_);                                            \
        nx_t_checks++;                                                          \
        if (va_ != vb_) {                                                       \
            nx_t_failures++;                                                    \
            fprintf(stderr, "FAIL %s:%d: %s == %s (%lld != %lld)\n", __FILE__,  \
                    __LINE__, #a_, #b_, (long long)va_, (long long)vb_);        \
        }                                                                       \
    } while (0)

#define NX_CHECK_EQ_F64(a_, b_, eps_)                                            \
    do {                                                                        \
        double va_ = (double)(a_);                                              \
        double vb_ = (double)(b_);                                              \
        nx_t_checks++;                                                          \
        if (!(fabs(va_ - vb_) <= (eps_))) {                                     \
            nx_t_failures++;                                                    \
            fprintf(stderr, "FAIL %s:%d: %s == %s (%.9g != %.9g)\n", __FILE__,  \
                    __LINE__, #a_, #b_, va_, vb_);                              \
        }                                                                       \
    } while (0)

#define NX_CHECK_EQ_STR(a_, b_)                                                  \
    do {                                                                        \
        const char *va_ = (a_);                                                 \
        const char *vb_ = (b_);                                                 \
        nx_t_checks++;                                                          \
        if (va_ == NULL || vb_ == NULL || strcmp(va_, vb_) != 0) {              \
            nx_t_failures++;                                                    \
            fprintf(stderr, "FAIL %s:%d: '%s' == '%s'\n", __FILE__, __LINE__,   \
                    va_ != NULL ? va_ : "(null)", vb_ != NULL ? vb_ : "(null)");\
        }                                                                       \
    } while (0)

/* nx_status_t checks; on failure print the recorded error message. */
#define NX_CHECK_STATUS(call_, want_)                                            \
    do {                                                                        \
        nx_status_t st_ = (call_);                                              \
        nx_t_checks++;                                                          \
        if (st_ != (want_)) {                                                   \
            nx_t_failures++;                                                    \
            fprintf(stderr, "FAIL %s:%d: %s -> %s (want %s): %s\n", __FILE__,   \
                    __LINE__, #call_, nx_status_str(st_), nx_status_str(want_), \
                    nx_error_last());                                           \
        }                                                                       \
    } while (0)

/* test_alloc.c */
void test_alloc(void);

/* test_arena.c */
void test_arena(void);
void test_arena_marks(void);

/* test_error.c */
void test_error(void);

/* test_log.c */
void test_log(void);

/* test_tensor.c */
void test_tensor(void);
void test_tensor_views(void);

/* test_config.c */
void test_config_basic(void);
void test_config_errors(void);

#endif /* TESTS_H */
