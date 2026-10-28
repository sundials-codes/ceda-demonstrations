#!/bin/bash
# Plot field energy of the current run against ref/, and print the max relative difference.
# Usage: ./plot.sh [label]   (label is appended to the output image name)
#
# Daniel R. Reynolds
# Math & Stat @ UMBC
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$HERE"
"$PYTHON" plot_field_energy.py "$INPUT-field_energy.gkyl" ref/gk_sheath_2x2v_p1-field_energy.gkyl "${1:-}"
