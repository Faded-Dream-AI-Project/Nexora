# Nexora coding standards

Binding for all contributions. The CI build enforces the mechanical parts
(`-std=c11 -Wall -Wextra -Werror -Wpedantic -Wshadow -Wstrict-prototypes
-Wmissing-prototypes -Wformat=2`); the rest is review discipline.

## Language and dependencies

- C11 only. No GNU extensions except guarded `__attribute__`/builtins behind
  `#if defined(__GNUC__)` (see `NX_PRINTF`).
- `_POSIX_C_SOURCE 200809L` is defined in `src/core/nx_common.h`. That header
  (directly or via another `nx_*.h`) must be the first include of every
  translation unit; a `.c` file never includes a system header before its
  own project header.
- Allowed dependencies: libc, pthreads, the dynamically loaded Vulkan
  loader. Anything else requires written justification and approval in
  writing before it is added (project constraint, not a style preference).
- Prefer simple, readable C. Small functions, clear names, comments only
  where the reason is not obvious. No clever macros that hide control flow
  (the test CHECK macros are the sanctioned exception; they are a single
  plain `if` plus a print).

## Naming

- Types: `nx_arena_t`, `nx_tensor_t` — lowercase with `_t`, `nx_` prefix.
- Functions: `nx_module_verb` / `nx_module_verb_object`:
  `nx_arena_alloc`, `nx_tensor_create`, `nx_config_get_i64`.
- Enum constants: `NX_E_NOMEM`, `NX_F32`, `NX_LOG_WARN` — uppercase.
- Macros and constants: `NX_MAX_DIMS`, `NX_ALIGNMENT`.
- Static (file-local) helpers: normal lowercase names, no prefix needed.
- One function per concern; if a comment says "and then", split it.

## Error handling

- Fallible functions return `nx_status_t` (or a pointer that may be NULL)
  and call `nx_error_set(...)` before every failure return. `nx_error_set`
  returns its own status, so the common shape is:
  `return nx_error_set(NX_E_INVAL, "shape NULL");`
- Check every allocation result, every Vulkan return code, every file I/O
  return. No silent failures, ever.
- Do not invent sentinel returns (-1, NULL-as-boolean) where a status code
  exists.
- Asserts are for internal invariants ("this cannot be false if the caller
  obeyed the contract"), e.g. `assert(a != NULL)` in arena internals. Input
  validation from outside (config values, file content, shapes coming from
  user data) uses status codes, not asserts. Asserts stay ON in release
  builds.

## Memory

- Dynamic per-object lifetime (rare): `nx_alloc_aligned` / `nx_free_aligned`.
- Bulk / per-step lifetime (the norm): arenas. Structs and buffers both come
  from the arena; freeing happens by `nx_arena_reset`/`rewind`, never object
  by object.
- Document ownership at API boundaries: who owns the buffer (arena vs
  caller), when borrowed pointers expire (e.g. after `nx_arena_rewind`).
- Every `size_t` computation that multiplies user-controlled sizes goes
  through `nx_size_mul_overflows` / `nx_size_add_overflows`.
- Tensor buffers are `NX_ALIGNMENT` (64) aligned so SIMD paths can assume it.

## Integers and floats

- Sizes, element counts, offsets: `size_t`. Dimensions, ranks: `int32_t`.
  Config integers: `int64_t`.
- Make narrowing conversions explicit with a cast; the compiler flags the
  rest.
- Numerical tests compare with an explicit tolerance and PRINT THE MAX ERROR
  on failure (project rule; see the test macros).
- Never claim two floats are "equal" without an epsilon; exact copies and
  integer roundtrips are the only exceptions.

## Tests

- Every module ships with tests in `tests/test_<module>.c`, registered in
  `tests/main.c` and declared in `tests/tests.h`. `make test` runs them under
  ASan+UBSan; `make test-release` runs the same tests in release mode.
- GPU kernels are additionally validated against the CPU reference op
  (phase 3+), with max error printed.
- A bug fix gets a test that fails before the fix and passes after.
- No performance claim enters docs/STATUS.md without a measured number and
  the machine it was measured on.

## Files and headers

- Every file starts with the two-line copyright header (see any file).
- Headers are self-contained: include what they use; include guards named
  after the file (`NX_TENSOR_H`).
- Headers document the contract: params, ownership, failure modes.
- `TODO(phase N)` marks known future work; every TODO is listed in
  docs/STATUS.md. No stubs presented as finished work.

## Git

- Small commits with imperative subjects ("arena: retain blocks after
  rewind"). Never commit secrets, tokens or credentials. No force-push to
  main.
- `main` stays green: `./scripts/run_tests.sh` must pass before merging.

## Style

- 4 spaces, no tabs (Makefile excepted); braces K&R; opening brace of a
  function on its own line; ~100-column soft limit.
- `clang-format -i` with the repo `.clang-format` settles formatting debates.
