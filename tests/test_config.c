/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software. */

#include "tests.h"

#include "core/nx_config.h"

#include <stdlib.h>
#include <unistd.h>

void test_config_basic(void)
{
    static const char *text =
        "# training profile\n"
        "[model]\n"
        "dim = 512\n"
        "layers = 8\n"
        "dropout = 0.1\n"
        "name = \"tiny-lm\"\n"
        "bias = true\n"
        "\n"
        "[optim]\n"
        "lr = 3e-4\n"
        "weight_decay = 0.01  # inline comment\n"
        "adam = false\r\n";

    nx_config_t *cfg = NULL;
    NX_CHECK_STATUS(nx_config_parse(&cfg, text), NX_OK);
    NX_CHECK(cfg != NULL);
    NX_CHECK_EQ_I64((int64_t)nx_config_count(cfg), 8);

    int64_t i = 0;
    double f = 0.0;
    bool b = false;
    const char *s = NULL;

    NX_CHECK_STATUS(nx_config_get_i64(cfg, "model", "dim", &i), NX_OK);
    NX_CHECK_EQ_I64(i, 512);
    NX_CHECK_STATUS(nx_config_get_i64(cfg, "model", "layers", &i), NX_OK);
    NX_CHECK_EQ_I64(i, 8);
    NX_CHECK_STATUS(nx_config_get_f64(cfg, "optim", "lr", &f), NX_OK);
    NX_CHECK_EQ_F64(f, 3e-4, 1e-12);
    NX_CHECK_STATUS(nx_config_get_f64(cfg, "model", "dropout", &f), NX_OK);
    NX_CHECK_EQ_F64(f, 0.1, 1e-12);
    NX_CHECK_STATUS(nx_config_get_bool(cfg, "model", "bias", &b), NX_OK);
    NX_CHECK(b);
    NX_CHECK_STATUS(nx_config_get_bool(cfg, "optim", "adam", &b), NX_OK);
    NX_CHECK(!b);
    NX_CHECK_STATUS(nx_config_get_str(cfg, "model", "name", &s), NX_OK);
    NX_CHECK_EQ_STR(s, "tiny-lm");

    /* negative ints and floats */
    nx_config_destroy(cfg);
    NX_CHECK_STATUS(nx_config_parse(&cfg, "[a]\nneg = -12\nnf = -0.5\nbig = 9223372036854775807\n"),
                    NX_OK);
    NX_CHECK_STATUS(nx_config_get_i64(cfg, "a", "neg", &i), NX_OK);
    NX_CHECK_EQ_I64(i, -12);
    NX_CHECK_STATUS(nx_config_get_f64(cfg, "a", "nf", &f), NX_OK);
    NX_CHECK_EQ_F64(f, -0.5, 1e-12);
    NX_CHECK_STATUS(nx_config_get_i64(cfg, "a", "big", &i), NX_OK);
    NX_CHECK_EQ_I64(i, INT64_MAX);

    /* presence checks */
    NX_CHECK(nx_config_has(cfg, "a", "neg"));
    NX_CHECK(!nx_config_has(cfg, "a", "nope"));
    NX_CHECK(!nx_config_has(cfg, "nosec", "neg"));

    /* missing key -> error with message */
    NX_CHECK_STATUS(nx_config_get_i64(cfg, "a", "nope", &i), NX_E_INVAL);
    NX_CHECK(strstr(nx_error_last(), "a.nope") != NULL);

    /* wrong type -> error (no silent coercion except int->float) */
    NX_CHECK_STATUS(nx_config_get_bool(cfg, "a", "neg", &b), NX_E_INVAL);
    NX_CHECK_STATUS(nx_config_get_str(cfg, "a", "neg", &s), NX_E_INVAL);
    NX_CHECK_STATUS(nx_config_get_i64(cfg, "a", "nf", &i), NX_E_INVAL);
    NX_CHECK_STATUS(nx_config_get_f64(cfg, "a", "neg", &f), NX_OK); /* int -> float ok */
    NX_CHECK_EQ_F64(f, -12.0, 1e-12);

    /* _or variants: default on missing, error on wrong type */
    NX_CHECK_STATUS(nx_config_get_i64_or(cfg, "a", "nope", 42, &i), NX_OK);
    NX_CHECK_EQ_I64(i, 42);
    NX_CHECK_STATUS(nx_config_get_i64_or(cfg, "a", "neg", 42, &i), NX_OK);
    NX_CHECK_EQ_I64(i, -12);
    NX_CHECK_STATUS(nx_config_get_f64_or(cfg, "a", "nope", 1.5, &f), NX_OK);
    NX_CHECK_EQ_F64(f, 1.5, 1e-12);
    NX_CHECK_STATUS(nx_config_get_bool_or(cfg, "a", "nope", true, &b), NX_OK);
    NX_CHECK(b);
    NX_CHECK_STATUS(nx_config_get_str_or(cfg, "a", "nope", "dflt", &s), NX_OK);
    NX_CHECK_EQ_STR(s, "dflt");
    NX_CHECK_STATUS(nx_config_get_bool_or(cfg, "a", "neg", true, &b), NX_E_INVAL);

    /* allow_keys: clean schema passes, typos are listed */
    nx_config_destroy(cfg);
    NX_CHECK_STATUS(nx_config_parse(&cfg, text), NX_OK);
    static const char *const model_ok[] = { "dim", "layers", "dropout", "name", "bias", NULL };
    NX_CHECK_STATUS(nx_config_allow_keys(cfg, "model", model_ok), NX_OK);
    static const char *const model_bad[] = { "dim", NULL };
    NX_CHECK_STATUS(nx_config_allow_keys(cfg, "model", model_bad), NX_E_FORMAT);
    NX_CHECK(strstr(nx_error_last(), "layers") != NULL);
    /* a section with no keys always passes */
    static const char *const none[] = { NULL };
    NX_CHECK_STATUS(nx_config_allow_keys(cfg, "nothing-here", none), NX_OK);
    nx_config_destroy(cfg);

    /* file loading: happy path, BOM, and a missing file */
    const char *path = "/tmp/nexora_cfg_test.cfg";
    FILE *out = fopen(path, "wb");
    if (out != NULL) {
        fputs("\xEF\xBB\xBF[file]\nmode = \"gpu\"\n", out); /* BOM + LF */
        fclose(out);
        nx_config_t *fc = NULL;
        NX_CHECK_STATUS(nx_config_load(&fc, path), NX_OK);
        NX_CHECK_STATUS(nx_config_get_str(fc, "file", "mode", &s), NX_OK);
        NX_CHECK_EQ_STR(s, "gpu");
        nx_config_destroy(fc);
        unlink(path);
    }
    nx_config_t *missing = NULL;
    NX_CHECK_STATUS(nx_config_load(&missing, "/tmp/nexora_no_such_file.cfg"), NX_E_IO);
    NX_CHECK(missing == NULL);
}

void test_config_errors(void)
{
    struct {
        const char *name;
        const char *text;
        const char *must_mention;
    } bad[] = {
        { "key outside section", "dim = 1\n", "outside" },
        { "duplicate key", "[a]\nk = 1\nk = 2\n", "duplicate" },
        { "duplicate section", "[a]\n[a]\n", "duplicate" },
        { "malformed float", "[a]\nk = 1.2.3\n", "invalid number" },
        { "hex rejected", "[a]\nk = 0x10\n", "invalid value" },
        { "nan rejected", "[a]\nk = nan\n", "invalid value" },
        { "bare sign rejected", "[a]\nk = -\n", "invalid value" },
        { "unterminated string", "[a]\nk = \"abc\n", "unterminated" },
        { "bad escape", "[a]\nk = \"a\\nb\"\n", "unknown escape" },
        { "int overflow", "[a]\nk = 99999999999999999999999\n", "out of range" },
        { "float overflow", "[a]\nk = 1e999\n", "out of range" },
        { "garbage after value", "[a]\nk = 1 junk\n", "unexpected text" },
        { "garbage after section", "[a] junk\n", "unexpected text" },
        { "missing equals", "[a]\nk 1\n", "expected '='" },
        { "missing value", "[a]\nk =\n", "missing value" },
        { "empty section name", "[]\n", "section name" },
        { "boolean must be literal", "[a]\nk = TRUE\n", "invalid value" },
    };

    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        nx_config_t *cfg = NULL;
        NX_CHECK_STATUS(nx_config_parse(&cfg, bad[i].text), NX_E_FORMAT);
        NX_CHECK(cfg == NULL);
        NX_CHECK(nx_error_last_status() == NX_E_FORMAT);
        nx_t_checks++;
        if (strstr(nx_error_last(), bad[i].must_mention) == NULL) {
            nx_t_failures++;
            fprintf(stderr, "FAIL %s:%d: '%s': message '%s' lacks '%s'\n", __FILE__,
                    __LINE__, bad[i].name, nx_error_last(), bad[i].must_mention);
        }
    }

    /* line numbers are reported */
    nx_config_t *cfg = NULL;
    NX_CHECK_STATUS(nx_config_parse(&cfg, "[ok]\nx = 1\n[ok2]\nbad line here\n"),
                    NX_E_FORMAT);
    NX_CHECK(strstr(nx_error_last(), "line 4") != NULL);

    /* empty and comment-only inputs are fine */
    NX_CHECK_STATUS(nx_config_parse(&cfg, ""), NX_OK);
    NX_CHECK_EQ_I64((int64_t)nx_config_count(cfg), 0);
    nx_config_destroy(cfg);
    NX_CHECK_STATUS(nx_config_parse(&cfg, "# nothing but comments\n\n"), NX_OK);
    NX_CHECK_EQ_I64((int64_t)nx_config_count(cfg), 0);
    nx_config_destroy(cfg);

    /* CRLF line endings everywhere */
    NX_CHECK_STATUS(nx_config_parse(&cfg, "[a]\r\nk = 3\r\nj = \"s\"\r\n"), NX_OK);
    int64_t v = 0;
    const char *sv = NULL;
    NX_CHECK_STATUS(nx_config_get_i64(cfg, "a", "k", &v), NX_OK);
    NX_CHECK_EQ_I64(v, 3);
    NX_CHECK_STATUS(nx_config_get_str(cfg, "a", "j", &sv), NX_OK);
    NX_CHECK_EQ_STR(sv, "s");
    nx_config_destroy(cfg);

    /* a line longer than the limit is rejected */
    static char longline[6000];
    longline[0] = 'k';
    longline[1] = '=';
    longline[2] = '"';
    for (size_t i = 3; i < sizeof longline - 2; i++) {
        longline[i] = 'x';
    }
    longline[sizeof longline - 2] = '"';
    longline[sizeof longline - 1] = '\0';
    NX_CHECK_STATUS(nx_config_parse(&cfg, longline), NX_E_FORMAT);
    NX_CHECK(cfg == NULL);

    /* NULL args */
    NX_CHECK_STATUS(nx_config_parse(&cfg, NULL), NX_E_INVAL);
    NX_CHECK_STATUS(nx_config_parse(NULL, "[a]\n"), NX_E_INVAL);
    nx_config_destroy(NULL); /* must not crash */
}
