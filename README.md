# Nexora

Nexora: a from-scratch AI training stack in pure C. No CUDA, no Python, no
C++ dependencies. Config-driven training, fine-tuning and reinforcement
learning, powered by Vulkan compute with GPU-only, CPU-only or hybrid modes.
Built for fast, simple training on consumer hardware. Source-available.

**License: source-available, NOT open source.** See
[LICENSE.md](LICENSE.md). No OSI license is granted or implied.

## Status

Early. Phase 0 (repo skeleton, build, CI, standards) and the core of Phase 1
(arenas, tensors, strict config parser, error handling, logging, aligned
allocation) are implemented and tested. No neural-network training runs yet.
See [docs/STATUS.md](docs/STATUS.md) for what works, what is measured, and
what comes next — nothing on that list is claimed without having been run.

## Build and test

Requires a C11 compiler, GNU make, and libc. Nothing else. (Vulkan arrives in
phase 3 and will be loaded dynamically, not linked.)

```sh
make            # release build -> build/release/nexora
make test       # tests under ASan + UBSan
make test-release
./scripts/run_tests.sh   # full CI matrix (what GitHub Actions runs)
```

Everything builds with `-std=c11 -Wall -Wextra -Werror` plus
`-Wpedantic -Wshadow -Wstrict-prototypes -Wmissing-prototypes -Wformat=2`.

## Layout

```
src/core/    arenas, aligned alloc, tensors, config parser, errors, logging
src/cpu/     SIMD kernels, matmul, thread pool          (phase 1)
src/autograd/ reverse-mode autodiff                     (phase 1)
src/vk/      Vulkan compute backend                     (phase 3)
src/sched/   hybrid GPU/CPU scheduler                   (phase 4)
src/nn/      transformer building blocks                (phase 1+)
src/optim/   AdamW, schedules, clipping, checkpointing  (phase 1+)
src/data/    BPE tokenizer, streaming loader, .nxck     (phase 2)
src/train/   pre-training and fine-tuning loops         (phase 1/5)
src/rl/      policy gradient / GRPO-style RL            (phase 6)
src/infer/   sampling and KV cache                      (phase 6)
src/tools/   the nexora CLI
tests/       one test file per module, run by make test
docs/        STATUS, ARCHITECTURE, CODING, NXCK_SPEC (phase 2)
scripts/     CI entry point
```

Directories for phases that have not landed yet do not exist in the tree;
this table is the map of where they will. `docs/ARCHITECTURE.md` has the
details.

## CLI

```sh
./build/release/nexora info      # build, platform and dtype information
./build/release/nexora version
```

`train`, `finetune`, `rl`, `pack` and `bench` are planned commands (see
docs/STATUS.md); they are not implemented yet and the CLI says so honestly.

## Reference hardware

The stack is designed and will be benchmarked against:

- CPU: Intel i5-8400 (6C/6T, AVX2+FMA, no AVX-512, 9 MB L3)
- GPU: NVIDIA GTX 1050 Ti 4 GB (Pascal, via Vulkan — never CUDA)
- RAM: 16 GB DDR4

Performance numbers published in docs/STATUS.md will name the machine they
were measured on. No estimates.
