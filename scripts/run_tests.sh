#!/bin/sh
# Nexora CI entry point. Copyright (c) 2026 Faded-Dream-AI-Project.
# Source-available; see LICENSE.md. Not open-source software.
#
# Runs the full verification matrix locally and in CI:
#   1. release build with warnings as errors + CLI smoke test
#   2. debug tests under address+UB sanitizers
#   3. release tests (sanitizers can mask UB-dependent behavior)
# Exits non-zero on the first failure.

set -eu
cd "$(dirname "$0")/.."

CC="${CC:-cc}"

echo "== nexora ci =="
echo "-- environment --"
uname -a
"$CC" --version | head -n 1
make --version | head -n 1

echo "-- release build (warnings as errors) --"
make clean
make -j"$(nproc 2>/dev/null || echo 2)" CC="$CC"

echo "-- cli smoke --"
./build/release/nexora version
./build/release/nexora info

echo "-- tests: debug, ASan + UBSan --"
make clean
make test CC="$CC"

echo "-- tests: release --"
make clean
make test-release CC="$CC"

echo "== all green =="
