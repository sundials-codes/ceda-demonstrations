#!/bin/bash
# Remove old output and run the simulation. All arguments go to the executable
# (e.g. ./run.sh -s300). For MPI: MPIRUN="-np 2" ./run.sh -M -c 2
#
# Daniel R. Reynolds
# Math & Stat @ UMBC
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$HERE"
./clean.sh
if [ -n "${MPIRUN:-}" ]; then
  "$MPIRUN_BIN" $MPIRUN "./$INPUT" "$@"
else
  "./$INPUT" "$@"
fi
