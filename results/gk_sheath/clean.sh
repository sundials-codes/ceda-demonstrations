#!/bin/bash
# Remove run output (everything named <input>-*), keeping the source, executable and ref/.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC
source "$(dirname "$0")/common.sh"
cd "$HERE"
rm -f "$INPUT"-*
