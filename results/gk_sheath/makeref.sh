#!/bin/bash
# Run the full simulation with Gkeyll's native SSP-RK3 (GKYL_SUNDIALS_METHOD_NONE) in a
# scratch directory and REPLACE ref/ with its output, renamed gk_sheath_2x2v_p1-*.
# Everything else in the input file is used as is.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC
set -euo pipefail
source "$(dirname "$0")/common.sh"
WORK="$(mktemp -d /tmp/gk_sheath_ref.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

# Scratch copy of the input with the active .rk_method line replaced by the native method.
sed -E 's|^( *)\.rk_method = .*|\1.rk_method = GKYL_SUNDIALS_METHOD_NONE,|' "$HERE/$INPUT.c" > "$WORK/$INPUT.c"
if ! grep -q 'rk_method = GKYL_SUNDIALS_METHOD_NONE' "$WORK/$INPUT.c"; then
  echo "makeref.sh: could not set the rk_method line" >&2; exit 1
fi
gk_compile "$WORK/$INPUT.c" "$WORK/$INPUT"
(cd "$WORK" && "./$INPUT" "$@" | tee run.log)

rm -rf "$HERE/ref"; mkdir -p "$HERE/ref"
for f in "$WORK/$INPUT"-*; do
  mv "$f" "$HERE/ref/gk_sheath_2x2v_p1-${f##*/$INPUT-}"
done
cp "$WORK/run.log" "$HERE/ref/run.log"
cat > "$HERE/ref/README.md" <<EOR
Reference output of \`rt_gk_sheath_2x2v_p1\` made by \`makeref.sh\` on $(date -I) on $(hostname).
Method: Gkeyll native SSP-RK3 (\`GKYL_SUNDIALS_METHOD_NONE\`); all other inputs as in \`../$INPUT.c\`.
Gkeyll fork commit: $(git -C "$FORK" rev-parse --short HEAD).
EOR
echo "ref/ regenerated"
