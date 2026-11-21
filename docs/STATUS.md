# Nexora status

Updated: 2026-11-21 (session 1).

## Phase

Phase 0 complete. Phase 1 in progress: the core/ module is done and tested;
the CPU kernels, autograd and the tiny end-to-end training run are not
started.

## What works (verified this session)

Everything below was actually built and run on this machine; commands and
outputs are quoted verbatim.

- **Repo skeleton**: Makefile (release + debug/ASan+UBSan object trees,
  `test` / `test-release` / `format` / `clean`), `.clang-format`,
  `.editorconfig`, `.gitignore`, CI script `scripts/run_tests.sh` and a
  GitHub Actions workflow that runs it.
- **core/nx_error**: status codes + per-thread error messages
  (`nx_error_set` returns its own code; 512-byte cap; `nx_status_str`).
- **core/nx_log**: leveled logging (TRACE..ERROR), custom sink support,
  colored default stderr sink, level filtering.
- **core/nx_alloc**: aligned heap allocation (`posix_memalign`).
- **core/nx_arena**: block-list bump allocator: arbitrary power-of-two
  alignment, dedicated blocks for big requests, mark/rewind with block reuse
  (no malloc churn), reset, byte limits (the future VRAM budgeting hook),
  used/capacity/peak/block stats, latched failure flag.
- **core/nx_tensor**: dtype table (f32/f16/bf16/i32/i16/i8/u8), creation
  from arena (64-byte-aligned buffers), views: alias/reshape (with -1
  inference, contiguous-only)/permute/narrow/wrap (borrowed memory, custom
  strides), logical multi-index access, row-major iterator, fill, debug
  print. Scalar (0-dim) support. Overflow-checked shapes.
- **core/nx_config**: strict hand-written parser (sections, `key = value`,
  `#` comments, int/float/bool/quoted-string, CRLF + BOM tolerated,
  duplicate/unknown-escape/trailing-garbage/overlong-line rejection with
  line numbers), typed queries with `_or` defaults, per-section key
  whitelisting (`nx_config_allow_keys`). Only coercion: int→float.
- **tools/nexora**: CLI with `info`, `version`, `help`. Unknown commands
  exit 2 with usage.

Verification environment (this session; NOT the reference machine):

```
Linux ... 5.10.134-... x86_64 #1 SMP ... x86_64 GNU/Linux
cc (Debian 14.2.0-19) 14.2.0
GNU Make 4.4.1
```

Full CI pass (`./scripts/run_tests.sh`, exit 0): release build with
`-std=c11 -Wall -Wextra -Werror -Wpedantic -Wshadow -Wstrict-prototypes
-Wmissing-prototypes -Wformat=2`, CLI smoke, then the test suite twice:

```
$ make test          # debug, ASan + UBSan
nexora 0.1.0 core tests
ok   alloc
ok   arena
ok   arena/marks
ok   error
ok   log
ok   tensor
ok   tensor/views
ok   config/basic
ok   config/errors
--- 631 checks, 0 failures, 9 suites

$ make test-release
--- 631 checks, 0 failures, 9 suites
```

CLI smoke:

```
$ ./build/release/nexora info
nexora 0.1.0
license: source-available (not open source); see LICENSE.md
build: gcc 14.2.0, C11, Linux 5.10.134-...
platform: x86_64, 2 online processors
types: void*=8 size_t=8 float=4 double=8 nx_tensor_t=152
arena: max rank 8, alignment 64 bytes
dtypes: f32=4 f16=2 bf16=2 i32=4 i16=2 i8=1 u8=1
```

## Not measured yet

No performance numbers at all yet — no kernels exist to benchmark. The first
real measurements land with the CPU matmul (phase 1): tokens/s, step time,
memory. Phase 3+ adds Vulkan numbers. All numbers will name the machine;
none will be estimates. Note: this session's machine (2-core Xeon, sandbox)
is not the i5-8400 + GTX 1050 Ti reference machine; reference-machine
benchmarks need to be run there.

## Known issues / limitations

- `nx_config_get_f64` accepts an int-valued key (documented coercion);
  everything else is strict.
- f16/bf16 tensors print as raw hex; arithmetic/decode lands with cpu/ (the
  emulation path is deliberately not in core).
- The arena is single-threaded by design (one arena per thread); the thread
  pool arrives with cpu/.
- `docs/NXCK_SPEC.md` does not exist yet (phase 2 writes the spec BEFORE the
  implementation).
- No Windows support yet; platform isolation is in place (see
  docs/ARCHITECTURE.md).

## TODO (by phase)

- TODO(phase 1): cpu/ — thread pool; SIMD matmul (SSE/AVX2 runtime dispatch +
  scalar fallback), softmax, RMSnorm/LayerNorm, activations, attention.
- TODO(phase 1): autograd/ — tape in an arena; finite-difference gradient
  checks printing max error.
- TODO(phase 1): nn/ + optim/ + train/ minimal path — config-driven tiny
  transformer trained end to end on CPU (the phase-1 exit criterion).
- TODO(phase 2): data/ — BPE tokenizer trainer + encoder in C; streaming
  loader; `docs/NXCK_SPEC.md` first, then packer/loader/validator; report
  real compression ratio and decode throughput.
- TODO(phase 3): vk/ — dlopen the loader, one matmul validated against CPU
  (max error printed), then the full op set, buffer pools, staging.
- TODO(phase 4): sched/ — hybrid placement, forced GPU-only/CPU-only;
  gradient checkpointing, offloading, low precision where benchmarks justify.
- TODO(phase 5): fine-tuning: full + LoRA-style adapters, resume.
- TODO(phase 6): rl/ + infer/ — policy gradient, GRPO-style, reward plugins,
  rollouts, KV cache.
- TODO(phase 7): scaling, benchmarks, profiler, docs polish.
- TODO(phases 1-7): CLI subcommands train/finetune/rl/pack/bench.

## Next step (concrete)

Phase 1, CPU backend part 1: `src/cpu/` with the thread pool and the first
kernels (elementwise, softmax, RMSNorm) plus a strided batched matmul with
SSE/AVX2 runtime dispatch — each validated against a scalar reference with
max error printed, then wired into a first autograd tape. Exit criterion for
phase 1 remains: a config-driven tiny model trained on CPU, end to end.
