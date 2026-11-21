/* Nexora: AI training stack in pure C. Copyright (c) 2026 Faded-Dream-AI-Project.
 * Source-available; see LICENSE.md. Not open-source software.
 *
 * nexora CLI. Subcommands land with their phases (see docs/STATUS.md):
 * currently `info` and `version` exist; train/finetune/rl/pack/bench are
 * phase 1-7 work and are NOT implemented yet. */

#include "core/nx_common.h"
#include "core/nx_tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <unistd.h>

static const char *const k_usage =
    "usage: nexora <command> [args]\n"
    "\n"
    "commands:\n"
    "  info      build, platform and dtype information\n"
    "  version   print the Nexora version\n"
    "  help      show this help\n"
    "\n"
    "not implemented yet (see docs/STATUS.md): train, finetune, rl, pack, bench\n";

static int cmd_info(void)
{
    struct utsname u;
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    const char *compiler;

#if defined(__clang__)
    compiler = "clang " __clang_version__;
#elif defined(__GNUC__)
    compiler = "gcc " __VERSION__;
#else
    compiler = "unknown";
#endif

    if (uname(&u) != 0) {
        strcpy(u.sysname, "?");
        strcpy(u.machine, "?");
        strcpy(u.release, "?");
    }

    printf("nexora %s\n", NEXORA_VERSION_STRING);
    printf("license: source-available (not open source); see LICENSE.md\n");
    printf("build: %s, C11, %s %s\n", compiler, u.sysname, u.release);
    printf("platform: %s, %ld online processors\n", u.machine, cpus);
    printf("types: void*=%zu size_t=%zu float=%zu double=%zu nx_tensor_t=%zu\n",
           sizeof(void *), sizeof(size_t), sizeof(float), sizeof(double),
           sizeof(nx_tensor_t));
    printf("arena: max rank %d, alignment %d bytes\n", NX_MAX_DIMS, NX_ALIGNMENT);
    printf("dtypes:");
    for (int d = 0; d < (int)NX_DTYPE_COUNT; d++) {
        printf(" %s=%zu", nx_dtype_name((nx_dtype_t)d), nx_dtype_size((nx_dtype_t)d));
    }
    printf("\n");
    return 0;
}

int main(int argc, char **argv)
{
    const char *cmd;

    if (argc < 2) {
        fputs(k_usage, stderr);
        return 2;
    }
    cmd = argv[1];
    if (strcmp(cmd, "info") == 0) {
        return cmd_info();
    }
    if (strcmp(cmd, "version") == 0) {
        printf("%s\n", NEXORA_VERSION_STRING);
        return 0;
    }
    if (strcmp(cmd, "help") == 0 || strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        fputs(k_usage, stdout);
        return 0;
    }
    fprintf(stderr, "nexora: unknown command '%s'\n\n", cmd);
    fputs(k_usage, stderr);
    return 2;
}
