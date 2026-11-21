/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software.
 *
 * One config file drives an entire training run (model, data, optimizer,
 * schedule, hardware mode, output). The format is deliberately strict so a
 * typo fails loudly instead of silently changing a run:
 *
 *   - `key = value` lines inside `[section]` blocks; `#` starts a comment.
 *   - Values: 64-bit integers, floats (decimals with optional exponent),
 *     booleans (true/false), and "quoted strings" (escapes \" and \\ only).
 *   - The parser rejects: duplicate keys, keys outside any section,
 *     malformed numbers, unknown escapes, unterminated strings, garbage
 *     after a value, and lines longer than the limit. Errors name the line.
 *   - Unknown keys are a *semantic* check done by the consuming code via
 *     nx_config_allow_keys(), because each subcommand knows its own schema.
 *
 * Query rules: types never coerce silently except int->float (exact for all
 * practical values), documented in docs/CODING.md. Missing key = NX_E_INVAL;
 * wrong type = NX_E_INVAL with a message naming the key. Use the _or
 * variants for optional keys with defaults. */

#ifndef NX_CONFIG_H
#define NX_CONFIG_H

#include "nx_common.h"
#include "nx_error.h"

typedef struct nx_config nx_config_t;

/* Loads and parses a config file. A leading UTF-8 BOM is accepted; CRLF and
 * LF line endings both work. Files above 1 MiB are rejected. */
nx_status_t nx_config_load(nx_config_t **out, const char *path);

/* Parses from memory (tests, programmatic configs). NUL-terminated text. */
nx_status_t nx_config_parse(nx_config_t **out, const char *text);

/* Frees the config. Safe on NULL. */
void nx_config_destroy(nx_config_t *cfg);

/* Lookups. *out stays owned by the config (strings are borrowed pointers,
 * valid until nx_config_destroy). Missing key or wrong type -> NX_E_INVAL
 * with a recorded message. */
nx_status_t nx_config_get_i64(const nx_config_t *cfg, const char *section,
                              const char *key, int64_t *out);
nx_status_t nx_config_get_f64(const nx_config_t *cfg, const char *section,
                              const char *key, double *out);
nx_status_t nx_config_get_bool(const nx_config_t *cfg, const char *section,
                               const char *key, bool *out);
nx_status_t nx_config_get_str(const nx_config_t *cfg, const char *section,
                              const char *key, const char **out);

/* Optional-key variants: return the default when the key is absent, but a
 * present key with the wrong type is still a hard error. */
nx_status_t nx_config_get_i64_or(const nx_config_t *cfg, const char *section,
                                 const char *key, int64_t dflt, int64_t *out);
nx_status_t nx_config_get_f64_or(const nx_config_t *cfg, const char *section,
                                 const char *key, double dflt, double *out);
nx_status_t nx_config_get_bool_or(const nx_config_t *cfg, const char *section,
                                  const char *key, bool dflt, bool *out);
nx_status_t nx_config_get_str_or(const nx_config_t *cfg, const char *section,
                                 const char *key, const char *dflt, const char **out);

/* True when the key exists (any type). */
bool nx_config_has(const nx_config_t *cfg, const char *section, const char *key);

/* Semantic strictness: fails with NX_E_FORMAT listing every key in `section`
 * that is not in the NULL-terminated `allowed` list. The subcommand owning
 * the section calls this right after loading. */
nx_status_t nx_config_allow_keys(const nx_config_t *cfg, const char *section,
                                 const char *const *allowed);

/* Number of stored keys (all sections). */
size_t nx_config_count(const nx_config_t *cfg);

#endif /* NX_CONFIG_H */
