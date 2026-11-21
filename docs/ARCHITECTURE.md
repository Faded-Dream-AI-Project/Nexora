# Nexora architecture

Lead document for module boundaries, build-system decisions and the phase
plan. Read together with docs/STATUS.md (current state) and docs/CODING.md
(how code is written).

## Mission and hard constraints

Re-implement the AI training stack in pure C so that training is faster,
simpler and runs on consumer hardware:

- **Language: C11 only.** No C++, no Python anywhere in the runtime or
  training path. If a Python or C++ library would normally be needed, the
  required part is re-implemented in C.
- **No CUDA, ROCm, cuDNN, cuBLAS, no external ML libraries.**
- **GPU: Vulkan compute only.** The Vulkan loader is dlopen'd, never linked.
  GLSL compute shaders are compiled to SPIR-V at build time and the SPIR-V is
  shipped embedded, so users need no shader compiler.
- **Dependencies: libc, pthreads, the Vulkan loader.** Anything beyond that
  needs written justification and explicit approval before being added.
- **Linux first**, platform code isolated so Windows can follow.
- **Reference machine** (all features must work within these limits):
  i5-8400 (6C/6T, AVX2+FMA, no AVX-512), GTX 1050 Ti 4 GB (Pascal, no tensor
  cores, no native fast fp16 — benchmark fp16 vs fp32 before assuming it
  helps), 16 GB DDR4. Budget VRAM explicitly (weights, activations, gradients,
  optimizer state, KV cache), spill to system RAM when needed, and expect
  PCIe transfers to dominate in hybrid mode.

## Module map and phase order

| dir | responsibility | phase |
| --- | --- | --- |
| `src/core/` | arenas, aligned alloc, tensors, error handling, logging, config parser | 0-1 |
| `src/cpu/` | SIMD kernels (SSE/AVX2, runtime dispatch, scalar fallback), matmul, softmax, norms, activations, attention, thread pool | 1 |
| `src/autograd/` | reverse-mode autodiff (tape in an arena), gradient checks vs finite differences | 1 |
| `src/vk/` | device/queue/memory management, buffer pools, command batching, compute pipelines, push constants, staging + transfer, same op set as cpu/ | 3 |
| `src/sched/` | hybrid scheduler: CPU/GPU placement by VRAM, measured throughput, transfer cost; forced GPU-only and CPU-only modes | 4 |
| `src/nn/` | embeddings, RoPE, memory-efficient attention, MLP, MoE (later), config-driven model definitions | 1+ |
| `src/optim/` | AdamW, SGD, LR schedules, clipping, accumulation, fp16/bf16 emulation, int8 where it helps, gradient checkpointing | 1+ |
| `src/data/` | BPE tokenizer (trainer + encoder in C), streaming loader, shuffling, packing, `.nxck` | 2 |
| `src/train/` | pre-training loop, full + PEFT fine-tuning (LoRA-style), checkpointing and resume | 1/5 |
| `src/rl/` | policy gradient, GRPO-style methods, reward plugins, rollouts | 6 |
| `src/infer/` | minimal sampling + KV cache for evaluation and rollouts | 6 |
| `src/tools/` | `nexora` CLI: train, finetune, rl, pack, bench, info | 0-7 |

Module dependency rule: dependencies point downward in that table plus
sideways into `core`. `cpu/` and `vk/` never include each other; `sched/`
calls both through op tables. Later modules must not leak upward: `core`
stays independent of everything.

## Design decisions already made

### Arena-first memory

Everything short-lived in a training step (graph nodes, activations, views,
scratch) is bump-allocated from `nx_arena_t` and freed in bulk by
mark/rewind. Rationale: allocation on the hot path becomes a pointer bump,
per-object free disappears (a major source of C bugs), and VRAM budgeting
later becomes "set an arena limit" — the same code path as the CPU
(see `nx_arena_set_limit`). Blocks are retained after rewind so a step's
arena reaches steady state with zero mallocs.

### Tensors as views over shared storage

`nx_tensor_t` carries dtype, shape, strides (in elements) and a data pointer;
`create` allocates the buffer from an arena, while
`alias/reshape/permute/narrow/wrap` allocate only a small struct and share or
borrow storage. Bulk kernels live in `cpu/` and `vk/`, not in core — core
provides layout, indexing and debugging only. F16/BF16 are storage dtypes
from day one; their arithmetic is emulated where the hardware lacks it.

### Errors: status codes plus a per-thread message

Every fallible function returns `nx_status_t` and records a human-readable
message via `nx_error_set()` (which conveniently returns its argument).
No silent failures: allocation results, Vulkan return codes and file I/O are
all checked. The message is per-thread (`_Thread_local`), ready for the
phase-1 thread pool.

### Config: one strict file, hand-written parser

The config format (sections, `key = value`, `#` comments, int/float/bool/
quoted-string values) is parsed by a hand-written strict parser: duplicates,
malformed numbers, unknown escapes, trailing garbage and overlong lines are
all hard errors with line numbers. Semantic strictness (unknown keys for a
given subcommand) is enforced by `nx_config_allow_keys()`. The only coercion
is int→float. One config file drives an entire run; no code changes to start
one.

### Platform isolation

Strict `-std=c11` with `_POSIX_C_SOURCE 200809L` defined once in
`nx_common.h` (the mandatory first include). POSIX calls used so far:
`posix_memalign`, `clock_gettime`, `isatty`, `open_memstream` (tests).
Windows support will slot behind `nx_alloc`/time functions without touching
module code.

### Build system

Plain GNU make (no CMake): zero dependencies beyond `cc`, matches the
"minimal dependencies" constraint, and the whole build is one readable file.
Release keeps asserts on by default (a crash is better than silent
corruption); a `-DNDEBUG` perf profile can be added once benchmarks justify
it. Tests build in a separate object tree with ASan+UBSan and also run in
release mode, because sanitizers can hide UB-dependent behavior.

## The .nxck format (phase 2, spec first)

A compact chunked container for training data and text: header, version,
chunk table, checksums, random access + streaming, token-level compression
(varint, dictionary and delta schemes, optional entropy coding), sequence
packing, metadata, deduplication. The full spec will be written in
`docs/NXCK_SPEC.md` BEFORE the packer/loader is implemented, and the packer
will report real compression ratios and decode throughput measured on real
data.

## Vulkan backend (phase 3)

The loader is opened with dlopen at runtime; the CLI must run on machines
with no Vulkan at all (that is why GPU-only mode is a config choice, not a
requirement). First milestone: one matmul on GPU validated against the CPU
reference, then the full op set. Every kernel is validated against the CPU
reference; numerical tests print max error.
