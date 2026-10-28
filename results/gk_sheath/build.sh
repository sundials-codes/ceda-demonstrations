#!/bin/bash
# Compile the local rt_gk_sheath_2x2v_p1.c against the installed Gkeyll.
# Usage: ./build.sh [--lib]    (--lib first runs `make install` in the fork)
#
# Daniel R. Reynolds
# Math & Stat @ UMBC
set -euo pipefail
source "$(dirname "$0")/common.sh"
if [ "${1:-}" = "--lib" ]; then
  (cd "$FORK" && make -j16 install)
fi
cd "$HERE"
gk_compile "$INPUT.c" "$INPUT"
echo "built $HERE/$INPUT"
