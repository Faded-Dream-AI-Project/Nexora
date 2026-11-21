/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software.
 *
 * Hand-written strict parser. See nx_config.h for the format and
 * docs/CODING.md for the reasoning. No external config library is allowed
 * by the project constraints. */

#include "nx_config.h"

#include "nx_error.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NX_CFG_LINE_MAX 4096
#define NX_CFG_FILE_MAX (1u << 20)
#define NX_CFG_IDENT_MAX 128

typedef enum {
    CFG_INT,
    CFG_FLOAT,
    CFG_BOOL,
    CFG_STR
} cfg_type_t;

typedef struct {
    char *section;
    char *key;
    cfg_type_t type;
    int line; /* first definition, for duplicate reports */
    union {
        int64_t i;
        double f;
        bool b;
        char *s;
    } v;
} cfg_entry_t;

typedef struct {
    char *name;
    int line;
} cfg_section_t;

struct nx_config {
    cfg_entry_t *entries;
    size_t n, cap;
    cfg_section_t *sections;
    size_t nsec, capsec;
};

/* ---- small char helpers (locale-independent, ASCII only) -------------- */

static bool ch_is_space(char c)
{
    return c == ' ' || c == '\t';
}

static bool ch_is_digit(char c)
{
    return c >= '0' && c <= '9';
}

static bool ch_is_alpha(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

static bool ch_is_ident(char c)
{
    return ch_is_alpha(c) || ch_is_digit(c) || c == '_' || c == '.' || c == '-';
}

/* End of the value token: first space, comment start or line end. */
static const char *token_end(const char *c)
{
    while (*c != '\0' && !ch_is_space(*c) && *c != '#') {
        c++;
    }
    return c;
}

/* ---- allocation helpers ------------------------------------------------ */

static char *dup_range(const char *s, size_t n)
{
    char *d = malloc(n + 1);

    if (d == NULL) {
        return NULL;
    }
    memcpy(d, s, n);
    d[n] = '\0';
    return d;
}

static bool cfg_push_entry(nx_config_t *cfg, cfg_entry_t e)
{
    if (cfg->n == cfg->cap) {
        size_t ncap = cfg->cap != 0 ? cfg->cap * 2 : 16;
        cfg_entry_t *ne = realloc(cfg->entries, ncap * sizeof *ne);
        if (ne == NULL) {
            return false;
        }
        cfg->entries = ne;
        cfg->cap = ncap;
    }
    cfg->entries[cfg->n++] = e;
    return true;
}

static bool cfg_push_section(nx_config_t *cfg, char *name, int line)
{
    if (cfg->nsec == cfg->capsec) {
        size_t ncap = cfg->capsec != 0 ? cfg->capsec * 2 : 8;
        cfg_section_t *ns = realloc(cfg->sections, ncap * sizeof *ns);
        if (ns == NULL) {
            return false;
        }
        cfg->sections = ns;
        cfg->capsec = ncap;
    }
    cfg->sections[cfg->nsec].name = name;
    cfg->sections[cfg->nsec].line = line;
    cfg->nsec++;
    return true;
}

void nx_config_destroy(nx_config_t *cfg)
{
    if (cfg == NULL) {
        return;
    }
    for (size_t i = 0; i < cfg->n; i++) {
        free(cfg->entries[i].section);
        free(cfg->entries[i].key);
        if (cfg->entries[i].type == CFG_STR) {
            free(cfg->entries[i].v.s);
        }
    }
    for (size_t i = 0; i < cfg->nsec; i++) {
        free(cfg->sections[i].name);
    }
    free(cfg->entries);
    free(cfg->sections);
    free(cfg);
}

/* ---- parsing ----------------------------------------------------------- */

/* Scans an identifier at `*c`; advances past it. Returns the end pointer or
 * NULL when the token is not a valid identifier (or too long). */
static const char *scan_ident(const char *c, const char **end, size_t *len)
{
    const char *start;

    if (!ch_is_alpha(*c) && *c != '_') {
        return NULL;
    }
    start = c;
    while (ch_is_ident(*c)) {
        c++;
    }
    if ((size_t)(c - start) > NX_CFG_IDENT_MAX) {
        return NULL;
    }
    *end = c;
    *len = (size_t)(c - start);
    return start;
}

typedef struct {
    nx_config_t *cfg;
    const char *cur_section; /* current [section] name (borrowed from cfg) */
    int line_no;
    char errbuf[512]; /* parse errors accumulate here, set once at the end */
    bool failed;
} cfg_parser_t;

/* All parse failures go through this: records line + detail. `a`/`b` may
 * point into the untrusted line buffer, so both are copied with a hard cap
 * (the error message must never overflow, whatever the input). */
static bool perr(cfg_parser_t *p, const char *fmt, const char *a, const char *b)
{
    char sa[96], sb[96];

    snprintf(sa, sizeof sa, "%.80s", a != NULL ? a : "");
    snprintf(sb, sizeof sb, "%.80s", b != NULL ? b : "");
    snprintf(p->errbuf, sizeof p->errbuf, "config: line %d: %s%s%s%s%s", p->line_no, fmt,
             a != NULL ? ": '" : "", sa, a != NULL ? "'" : "", sb);
    p->failed = true;
    return false;
}

static bool parse_string_value(cfg_parser_t *p, const char **c, char **out)
{
    const char *r = *c + 1; /* past opening quote */
    char buf[NX_CFG_LINE_MAX];
    size_t w = 0;

    while (*r != '"') {
        if (*r == '\0') {
            *out = NULL;
            return perr(p, "unterminated string", NULL, NULL);
        }
        if (*r == '\\') {
            r++;
            if (*r != '"' && *r != '\\') {
                *out = NULL;
                return perr(p, "unknown escape", r, NULL);
            }
        }
        if (w + 1 >= sizeof buf) {
            *out = NULL;
            return perr(p, "string too long", NULL, NULL);
        }
        buf[w++] = *r++;
    }
    buf[w] = '\0';
    /* only whitespace and comments may follow the closing quote */
    const char *rest = r + 1;
    while (ch_is_space(*rest)) {
        rest++;
    }
    if (*rest != '\0' && *rest != '#') {
        *out = NULL;
        return perr(p, "unexpected text after value", rest, NULL);
    }
    *out = dup_range(buf, w);
    if (*out == NULL) {
        p->failed = true; /* OOM: errbuf handled by caller */
        return false;
    }
    *c = r + 1;
    return true;
}

static bool parse_number_value(cfg_parser_t *p, const char **c, cfg_entry_t *e)
{
    const char *tok = *c;
    const char *end = token_end(tok);
    size_t len = (size_t)(end - tok);
    char tmp[NX_CFG_LINE_MAX];
    char *conv_end;
    bool is_float = false;

    if (len == 0 || len >= sizeof tmp) {
        return perr(p, "missing value", NULL, NULL);
    }
    memcpy(tmp, tok, len);
    tmp[len] = '\0';

    /* classify */
    size_t start = 0;
    if (tmp[0] == '+' || tmp[0] == '-') {
        start = 1;
    }
    if (len == start) {
        return perr(p, "invalid value", tmp, NULL);
    }
    bool all_digits = true;
    bool num_chars = true;
    for (size_t i = start; i < len; i++) {
        if (!ch_is_digit(tmp[i])) {
            all_digits = false;
            if (!ch_is_digit(tmp[i]) && tmp[i] != '.' && tmp[i] != 'e' && tmp[i] != 'E' &&
                tmp[i] != '+' && tmp[i] != '-') {
                num_chars = false;
            }
        }
        if (tmp[i] == '.' || tmp[i] == 'e' || tmp[i] == 'E') {
            is_float = true;
        }
    }
    if (!all_digits && (!num_chars || !is_float)) {
        return perr(p, "invalid value", tmp, NULL);
    }

    errno = 0;
    if (all_digits) {
        long long v = strtoll(tmp, &conv_end, 10);
        if (errno == ERANGE) {
            return perr(p, "integer out of range", tmp, NULL);
        }
        if (conv_end != tmp + len) { /* cannot happen after classification */
            return perr(p, "invalid value", tmp, NULL);
        }
        e->type = CFG_INT;
        e->v.i = (int64_t)v;
    } else {
        double v = strtod(tmp, &conv_end);
        if (errno == ERANGE) {
            return perr(p, "float out of range", tmp, NULL);
        }
        if (conv_end != tmp + len) {
            return perr(p, "invalid number", tmp, NULL);
        }
        e->type = CFG_FLOAT;
        e->v.f = v;
    }
    *c = end;
    return true;
}

static bool parse_kv(cfg_parser_t *p, const char *c)
{
    cfg_entry_t e;
    const char *ident_end;
    size_t ident_len;
    const char *key = scan_ident(c, &ident_end, &ident_len);

    if (key == NULL) {
        return perr(p, "expected a key", c, NULL);
    }
    c = ident_end;
    while (ch_is_space(*c)) {
        c++;
    }
    if (*c != '=') {
        return perr(p, "expected '=' after key", NULL, NULL);
    }
    c++;
    while (ch_is_space(*c)) {
        c++;
    }

    memset(&e, 0, sizeof e);
    e.section = dup_range(p->cur_section, strlen(p->cur_section));
    e.key = dup_range(key, ident_len);
    e.line = p->line_no;
    if (e.section == NULL || e.key == NULL) {
        free(e.section);
        free(e.key);
        p->failed = true;
        return false;
    }

    for (size_t i = 0; i < p->cfg->n; i++) {
        if (strcmp(p->cfg->entries[i].section, e.section) == 0 &&
            strcmp(p->cfg->entries[i].key, e.key) == 0) {
            char where[64];
            snprintf(where, sizeof where, " (first set on line %d)",
                     p->cfg->entries[i].line);
            bool r = perr(p, "duplicate key", e.key, where);
            free(e.section);
            free(e.key);
            return r;
        }
    }

    bool ok;
    if (*c == '"') {
        ok = parse_string_value(p, &c, &e.v.s);
        if (ok) {
            e.type = CFG_STR;
        }
    } else if (c[0] == 't' && strncmp(c, "true", 4) == 0 && token_end(c) == c + 4) {
        e.type = CFG_BOOL;
        e.v.b = true;
        c += 4;
        ok = true;
    } else if (c[0] == 'f' && strncmp(c, "false", 5) == 0 && token_end(c) == c + 5) {
        e.type = CFG_BOOL;
        e.v.b = false;
        c += 5;
        ok = true;
    } else {
        ok = parse_number_value(p, &c, &e);
    }
    if (!ok) {
        free(e.section);
        free(e.key);
        return false; /* e.v.s was never set when a value parse failed */
    }

    /* trailing content check for non-string values */
    while (ch_is_space(*c)) {
        c++;
    }
    if (*c != '\0' && *c != '#') {
        free(e.section);
        free(e.key);
        if (e.type == CFG_STR) {
            free(e.v.s);
        }
        return perr(p, "unexpected text after value", c, NULL);
    }

    if (!cfg_push_entry(p->cfg, e)) {
        free(e.section);
        free(e.key);
        if (e.type == CFG_STR) {
            free(e.v.s);
        }
        p->failed = true;
        return false;
    }
    return true;
}

static bool parse_section(cfg_parser_t *p, const char *c)
{
    const char *ident_end;
    size_t ident_len;
    char *name;

    c++; /* past '[' */
    while (ch_is_space(*c)) {
        c++;
    }
    const char *ident = scan_ident(c, &ident_end, &ident_len);
    if (ident == NULL) {
        return perr(p, "expected a section name", c, NULL);
    }
    c = ident_end;
    while (ch_is_space(*c)) {
        c++;
    }
    if (*c != ']') {
        return perr(p, "expected ']' after section name", NULL, NULL);
    }
    c++;
    while (ch_is_space(*c)) {
        c++;
    }
    if (*c != '\0' && *c != '#') {
        return perr(p, "unexpected text after section", c, NULL);
    }

    name = dup_range(ident, ident_len);
    if (name == NULL) {
        p->failed = true;
        return false;
    }
    for (size_t i = 0; i < p->cfg->nsec; i++) {
        if (strcmp(p->cfg->sections[i].name, name) == 0) {
            free(name);
            char where[64];
            snprintf(where, sizeof where, " (first opened on line %d)",
                     p->cfg->sections[i].line);
            return perr(p, "duplicate section", ident, where);
        }
    }
    if (!cfg_push_section(p->cfg, name, p->line_no)) {
        free(name);
        p->failed = true;
        return false;
    }
    p->cur_section = name;
    return true;
}

static bool parse_line(cfg_parser_t *p, const char *c)
{
    while (ch_is_space(*c)) {
        c++;
    }
    if (*c == '\0' || *c == '#') {
        return true; /* blank or comment */
    }
    if (*c == '[') {
        return parse_section(p, c);
    }
    if (p->cur_section == NULL) {
        return perr(p, "key outside of any section", c, NULL);
    }
    return parse_kv(p, c);
}

static nx_status_t cfg_parse_text(nx_config_t *cfg, const char *text, size_t len)
{
    cfg_parser_t p;
    const char *cur = text;
    const char *end = text + len;
    int line_no = 0;
    bool ok = true;

    if (len >= 3 && (unsigned char)text[0] == 0xEF && (unsigned char)text[1] == 0xBB &&
        (unsigned char)text[2] == 0xBF) {
        cur += 3; /* accept and skip a UTF-8 BOM */
    }

    memset(&p, 0, sizeof p);
    p.cfg = cfg;

    while (cur < end && ok) {
        char line[NX_CFG_LINE_MAX];
        const char *nl = memchr(cur, '\n', (size_t)(end - cur));
        size_t n = nl != NULL ? (size_t)(nl - cur) : (size_t)(end - cur);

        line_no++;
        if (n >= sizeof line) {
            nx_error_set(NX_E_FORMAT, "config: line %d longer than %d bytes", line_no,
                         (int)NX_CFG_LINE_MAX);
            return NX_E_FORMAT;
        }
        memcpy(line, cur, n);
        line[n] = '\0';
        if (n > 0 && line[n - 1] == '\r') {
            line[n - 1] = '\0';
        }
        /* reject raw control characters (tab is the only allowed one) */
        for (const char *q = line; *q != '\0'; q++) {
            if (*q != '\t' && (unsigned char)*q < 0x20) {
                nx_error_set(NX_E_FORMAT, "config: line %d: control character in line",
                             line_no);
                return NX_E_FORMAT;
            }
        }

        p.line_no = line_no;
        ok = parse_line(&p, line);
        cur = nl != NULL ? nl + 1 : end;
    }

    if (!ok) {
        if (p.errbuf[0] != '\0') {
            nx_error_set(NX_E_FORMAT, "%s", p.errbuf);
            return NX_E_FORMAT;
        }
        return nx_error_set(NX_E_NOMEM, "config: out of memory while parsing");
    }
    return NX_OK;
}

/* ---- public load/parse -------------------------------------------------- */

nx_status_t nx_config_parse(nx_config_t **out, const char *text)
{
    nx_config_t *cfg;

    if (out == NULL || text == NULL) {
        return nx_error_set(NX_E_INVAL, "config: out or text NULL");
    }
    *out = NULL;
    cfg = calloc(1, sizeof *cfg);
    if (cfg == NULL) {
        return nx_error_set(NX_E_NOMEM, "config: alloc failed");
    }
    nx_status_t st = cfg_parse_text(cfg, text, strlen(text));
    if (st != NX_OK) {
        nx_config_destroy(cfg);
        return st;
    }
    *out = cfg;
    return NX_OK;
}

nx_status_t nx_config_load(nx_config_t **out, const char *path)
{
    FILE *f;
    char *buf = NULL;
    long size;
    size_t got;
    nx_config_t *cfg = NULL;
    nx_status_t st;

    if (out == NULL || path == NULL) {
        return nx_error_set(NX_E_INVAL, "config: out or path NULL");
    }
    *out = NULL;

    f = fopen(path, "rb");
    if (f == NULL) {
        return nx_error_set(NX_E_IO, "config: cannot open '%s': %s", path, strerror(errno));
    }
    if (fseek(f, 0, SEEK_END) != 0 || (size = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET) != 0) {
        nx_error_set(NX_E_IO, "config: cannot size '%s': %s", path, strerror(errno));
        fclose(f);
        return NX_E_IO;
    }
    if ((unsigned long)size > NX_CFG_FILE_MAX) {
        nx_error_set(NX_E_FORMAT, "config: '%s' larger than %u bytes", path,
                     (unsigned)NX_CFG_FILE_MAX);
        fclose(f);
        return NX_E_FORMAT;
    }
    buf = malloc((size_t)size + 1);
    if (buf == NULL) {
        nx_error_set(NX_E_NOMEM, "config: alloc of %ld bytes failed", size);
        fclose(f);
        return NX_E_NOMEM;
    }
    got = fread(buf, 1, (size_t)size, f);
    if (ferror(f) != 0) {
        nx_error_set(NX_E_IO, "config: read error on '%s': %s", path, strerror(errno));
        free(buf);
        fclose(f);
        return NX_E_IO;
    }
    if (got != (size_t)size) {
        nx_error_set(NX_E_IO, "config: short read on '%s' (%zu of %ld)", path, got, size);
        free(buf);
        fclose(f);
        return NX_E_IO;
    }
    fclose(f);
    buf[got] = '\0';

    cfg = calloc(1, sizeof *cfg);
    if (cfg == NULL) {
        nx_error_set(NX_E_NOMEM, "config: alloc failed");
        free(buf);
        return NX_E_NOMEM;
    }
    st = cfg_parse_text(cfg, buf, got);
    free(buf);
    if (st != NX_OK) {
        nx_config_destroy(cfg);
        return st;
    }
    *out = cfg;
    return NX_OK;
}

/* ---- lookup ------------------------------------------------------------- */

static const cfg_entry_t *cfg_find(const nx_config_t *cfg, const char *section,
                                   const char *key)
{
    for (size_t i = 0; i < cfg->n; i++) {
        if (strcmp(cfg->entries[i].section, section) == 0 &&
            strcmp(cfg->entries[i].key, key) == 0) {
            return &cfg->entries[i];
        }
    }
    return NULL;
}

static const char *cfg_type_name(cfg_type_t t)
{
    switch (t) {
    case CFG_INT:
        return "an integer";
    case CFG_FLOAT:
        return "a float";
    case CFG_BOOL:
        return "a bool";
    default:
        return "a string";
    }
}

static nx_status_t missing(const char *section, const char *key)
{
    return nx_error_set(NX_E_INVAL, "config: missing key '%s.%s'", section, key);
}

static nx_status_t wrong_type(const cfg_entry_t *e, const char *want)
{
    return nx_error_set(NX_E_INVAL, "config: '%s.%s' is %s, expected %s", e->section,
                        e->key, cfg_type_name(e->type), want);
}

nx_status_t nx_config_get_i64(const nx_config_t *cfg, const char *section,
                              const char *key, int64_t *out)
{
    const cfg_entry_t *e = cfg_find(cfg, section, key);
    if (e == NULL) {
        return missing(section, key);
    }
    if (e->type != CFG_INT) {
        return wrong_type(e, "an integer");
    }
    *out = e->v.i;
    return NX_OK;
}

nx_status_t nx_config_get_f64(const nx_config_t *cfg, const char *section,
                              const char *key, double *out)
{
    const cfg_entry_t *e = cfg_find(cfg, section, key);
    if (e == NULL) {
        return missing(section, key);
    }
    if (e->type == CFG_FLOAT) {
        *out = e->v.f;
        return NX_OK;
    }
    if (e->type == CFG_INT) {
        *out = (double)e->v.i; /* documented int->float coercion */
        return NX_OK;
    }
    return wrong_type(e, "a float");
}

nx_status_t nx_config_get_bool(const nx_config_t *cfg, const char *section,
                               const char *key, bool *out)
{
    const cfg_entry_t *e = cfg_find(cfg, section, key);
    if (e == NULL) {
        return missing(section, key);
    }
    if (e->type != CFG_BOOL) {
        return wrong_type(e, "a bool");
    }
    *out = e->v.b;
    return NX_OK;
}

nx_status_t nx_config_get_str(const nx_config_t *cfg, const char *section,
                              const char *key, const char **out)
{
    const cfg_entry_t *e = cfg_find(cfg, section, key);
    if (e == NULL) {
        return missing(section, key);
    }
    if (e->type != CFG_STR) {
        return wrong_type(e, "a string");
    }
    *out = e->v.s;
    return NX_OK;
}

nx_status_t nx_config_get_i64_or(const nx_config_t *cfg, const char *section,
                                 const char *key, int64_t dflt, int64_t *out)
{
    const cfg_entry_t *e = cfg_find(cfg, section, key);
    if (e == NULL) {
        *out = dflt;
        return NX_OK;
    }
    if (e->type != CFG_INT) {
        return wrong_type(e, "an integer");
    }
    *out = e->v.i;
    return NX_OK;
}

nx_status_t nx_config_get_f64_or(const nx_config_t *cfg, const char *section,
                                 const char *key, double dflt, double *out)
{
    const cfg_entry_t *e = cfg_find(cfg, section, key);
    if (e == NULL) {
        *out = dflt;
        return NX_OK;
    }
    if (e->type == CFG_FLOAT) {
        *out = e->v.f;
        return NX_OK;
    }
    if (e->type == CFG_INT) {
        *out = (double)e->v.i;
        return NX_OK;
    }
    return wrong_type(e, "a float");
}

nx_status_t nx_config_get_bool_or(const nx_config_t *cfg, const char *section,
                                  const char *key, bool dflt, bool *out)
{
    const cfg_entry_t *e = cfg_find(cfg, section, key);
    if (e == NULL) {
        *out = dflt;
        return NX_OK;
    }
    if (e->type != CFG_BOOL) {
        return wrong_type(e, "a bool");
    }
    *out = e->v.b;
    return NX_OK;
}

nx_status_t nx_config_get_str_or(const nx_config_t *cfg, const char *section,
                                 const char *key, const char *dflt, const char **out)
{
    const cfg_entry_t *e = cfg_find(cfg, section, key);
    if (e == NULL) {
        *out = dflt;
        return NX_OK;
    }
    if (e->type != CFG_STR) {
        return wrong_type(e, "a string");
    }
    *out = e->v.s;
    return NX_OK;
}

bool nx_config_has(const nx_config_t *cfg, const char *section, const char *key)
{
    return cfg_find(cfg, section, key) != NULL;
}

nx_status_t nx_config_allow_keys(const nx_config_t *cfg, const char *section,
                                 const char *const *allowed)
{
    char unknown[256];
    size_t shown = 0, count = 0;

    unknown[0] = '\0';
    for (size_t i = 0; i < cfg->n; i++) {
        if (strcmp(cfg->entries[i].section, section) != 0) {
            continue;
        }
        bool ok = false;
        for (const char *const *a = allowed; *a != NULL; a++) {
            if (strcmp(cfg->entries[i].key, *a) == 0) {
                ok = true;
                break;
            }
        }
        if (!ok) {
            count++;
            if (shown < 3) {
                size_t used = strlen(unknown);
                snprintf(unknown + used, sizeof unknown - used, "%s'%s'",
                         shown != 0 ? ", " : "", cfg->entries[i].key);
                shown++;
            }
        }
    }
    if (count == 0) {
        return NX_OK;
    }
    char more[64];
    if (count > shown) {
        snprintf(more, sizeof more, " (+%zu more)", count - shown);
    } else {
        more[0] = '\0';
    }
    return nx_error_set(NX_E_FORMAT, "config: unknown key%s in section [%s]: %s", more,
                        section, unknown);
}

size_t nx_config_count(const nx_config_t *cfg)
{
    return cfg->n;
}
