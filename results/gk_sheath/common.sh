# Shared settings for the gk_sheath staging scripts (sourced, not executed).
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
FORK="$REPO/deps/gkeyll-extsts"
PREFIX="$FORK/gkylsoft"
GK="$PREFIX/gkeyll"
MPIRUN_BIN="$PREFIX/openmpi/bin/mpirun"
PYTHON="$REPO/.venv/bin/python"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache-$USER}"

# Name of the input file (without .c) and of the output-file prefix.
INPUT=rt_gk_sheath_2x2v_p1

# Compile one input file ($1 = path to .c, $2 = output executable), using the same
# flags as the fork's in-tree `make build/gyrokinetic/creg/<name>` (Linux: OpenBLAS).
gk_compile() {
  local src="$1" exe="$2"
  gcc -O3 -g -ffast-math -fPIC \
    -DGKYL_HAVE_PKPM -DGKYL_HAVE_GYROKINETIC -DGKYL_HAVE_VLASOV -DGKYL_HAVE_MOMENTS \
    -DGKYL_HAVE_MPI -DGKYL_HAVE_LUA -DGKYL_HAVE_SUNDIALS \
    -DGKYL_SHARE_DIR="\"$GK/share\"" \
    -I"$GK/include" -I"$PREFIX/openmpi/include" -I"$PREFIX/luajit/include/luajit-2.1" \
    -I"$PREFIX/OpenBLAS/include" -I"$PREFIX/superlu/include" -I"$PREFIX/sundials/include" \
    -o "$exe" "$src" \
    -L"$GK/lib" -Wl,-rpath,"$GK/lib" -lg0gyrokinetic \
    -L"$PREFIX/OpenBLAS/lib" -lopenblas -L"$PREFIX/superlu/lib" -lsuperlu \
    -L"$PREFIX/openmpi/lib" -Wl,-rpath,"$PREFIX/openmpi/lib" -lmpi \
    -L"$PREFIX/luajit/lib" -Wl,-rpath,"$PREFIX/luajit/lib" -lluajit-5.1 \
    -L"$PREFIX/sundials/lib" -Wl,-rpath,"$PREFIX/sundials/lib" \
    -lsundials_arkode -lsundials_core -lsundials_sundomeigestpower -lsundials_sundomeigestarnoldi \
    -lm -lpthread -ldl
}
