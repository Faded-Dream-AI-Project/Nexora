# Nexora build. Copyright (c) 2026 Faded-Dream-AI-Project.
# Source-available; see LICENSE.md. Not open-source software.
#
# GNU make required. Nothing in the stack may depend on anything beyond
# libc, pthreads and (later) the Vulkan loader.
#
# Targets:
#   make / make all   release build: build/release/nexora + libnexora.a
#   make debug        debug build (-O0 -g, asserts on): build/debug/nexora
#   make test         debug tests with address+UB sanitizers, then run
#   make test-release release tests, then run (sanitizers can hide UB)
#   make format       clang-format the tree if the tool is installed
#   make clean
#
# Notes:
# - Asserts stay ON in release: a crashed run is better than a corrupted one.
#   A `-DNDEBUG` perf profile can be added once benchmarks justify it.

CC      ?= cc
AR      ?= ar
BUILD   := build
REL     := $(BUILD)/release
DBG     := $(BUILD)/debug

WARN    := -Wall -Wextra -Werror -Wpedantic -Wshadow -Wstrict-prototypes \
           -Wmissing-prototypes -Wformat=2
BASE    := -std=c11 $(WARN) -Isrc -Itests
REL_C   := $(BASE) -O2 -g
DBG_C   := $(BASE) -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer
SAN_L   := -fsanitize=address,undefined
LDLIBS  := -lm

SRC_CORE  := $(wildcard src/core/*.c)
SRC_TOOLS := $(wildcard src/tools/*.c)
SRC_TESTS := $(wildcard tests/test_*.c) tests/main.c

OBJ_REL_CORE  := $(patsubst %.c,$(REL)/%.o,$(SRC_CORE))
OBJ_REL_TOOLS := $(patsubst %.c,$(REL)/%.o,$(SRC_TOOLS))
OBJ_REL_TESTS := $(patsubst %.c,$(REL)/%.o,$(SRC_TESTS))
OBJ_DBG_CORE  := $(patsubst %.c,$(DBG)/%.o,$(SRC_CORE))
OBJ_DBG_TOOLS := $(patsubst %.c,$(DBG)/%.o,$(SRC_TOOLS))
OBJ_DBG_TESTS := $(patsubst %.c,$(DBG)/%.o,$(SRC_TESTS))

DEPS := $(wildcard $(REL)/*.d $(REL)/*/*.d $(DBG)/*.d $(DBG)/*/*.d)

.PHONY: all debug test test-release format clean help
.DEFAULT_GOAL := all

all: $(REL)/nexora

debug: $(DBG)/nexora

$(REL)/libnexora.a: $(OBJ_REL_CORE)
	$(AR) rcs $@ $^

$(DBG)/libnexora.a: $(OBJ_DBG_CORE)
	$(AR) rcs $@ $^

$(REL)/nexora: $(OBJ_REL_TOOLS) $(REL)/libnexora.a
	$(CC) $(REL_C) -o $@ $(OBJ_REL_TOOLS) $(REL)/libnexora.a $(LDLIBS)

$(DBG)/nexora: $(OBJ_DBG_TOOLS) $(DBG)/libnexora.a
	$(CC) $(DBG_C) $(SAN_L) -o $@ $(OBJ_DBG_TOOLS) $(DBG)/libnexora.a $(LDLIBS)

$(REL)/nx_tests: $(OBJ_REL_TESTS) $(REL)/libnexora.a
	$(CC) $(REL_C) -o $@ $(OBJ_REL_TESTS) $(REL)/libnexora.a $(LDLIBS)

$(DBG)/nx_tests: $(OBJ_DBG_TESTS) $(DBG)/libnexora.a
	$(CC) $(DBG_C) $(SAN_L) -o $@ $(OBJ_DBG_TESTS) $(DBG)/libnexora.a $(LDLIBS)

test: $(DBG)/nx_tests
	ASAN_OPTIONS=allocator_may_return_null=1 ./$(DBG)/nx_tests

test-release: $(REL)/nx_tests
	./$(REL)/nx_tests

$(REL)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(REL_C) -MMD -MP -c $< -o $@

$(DBG)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(DBG_C) -MMD -MP -c $< -o $@

format:
	@if command -v clang-format >/dev/null 2>&1; then \
		clang-format -i src/core/*.c src/core/*.h src/tools/*.c tests/*.c tests/*.h; \
		echo "formatted"; \
	else \
		echo "clang-format not installed; skipping"; \
	fi

clean:
	rm -rf $(BUILD)

help:
	@sed -n '1,22p' Makefile | sed 's/^# \{0,1\}//'

-include $(DEPS)
