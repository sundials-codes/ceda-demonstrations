# Super Time Stepping Methods for Diffusion using Discontinuous-Galerkin Spatial Discretizations

This is a `locked` branch allocated for the publication:

* [STS_diffusion_with_DG](https://github.com/sundials-codes/ceda-demonstrations/tree/STS_diffusion_with_DG) branch contains all testing code for the article:\
Aggul, M., Francisquez, M., Reynolds, D.R., Amihere, S., "Super Time Stepping Methods for Diffusion using Discontinuous-Galerkin Spatial Discretizations," 2026, [arXiv:2601.14508](https://arxiv.org/abs/2601.14508)

The purpose of this branch is to provide the full set of test codes related to this publication(s).  The following instructions should guide you through the steps required to compile, run, and postprocess those tests.

This is a repository of [SUNDIALS](https://github.com/LLNL/sundials)-based applications to assess and demonstrate the parallel performance of new super-time-stepping (STS) method capabilities that have been added to SUNDIALS as part of the [CEDA SciDAC project](https://sites.google.com/pppl.gov/ceda-scidac-5?usp=sharing).


## Installation

The following steps describe how to build the demonstration code in a Linux or OS X environment.


### Gettting the Code

To obtain the code, clone this repository with Git:

```bash
  git clone https://github.com/sundials-codes/ceda-demonstrations.git
```


### Requirements

To compile the codes in this repository you will need:

* [CMake](https://cmake.org) 3.20 or newer (both for SUNDIALS and for this repository)

* C compiler (C99 standard) and C++ compiler (C++11 standard)

* an MPI library e.g., [OpenMPI](https://www.open-mpi.org/), [MPICH](https://www.mpich.org/), etc.


The codes in this repository depend on three external libraries:

* [SUNDIALS](https://github.com/LLNL/sundials)

* [GkeyllZero](https://github.com/ammarhakim/gkylzero) -- note that is this is an older version of [Gkeyll](https://github.com/ammarhakim/gkeyll/tree/gk-g0-app_sundials)

If these are not already available on your system, they may be cloned from GitHub as submodules.  After cloning this repository using the command above, you can retrieve these submodules via:

```bash
  cd ceda-demonstrations/deps
  git submodule init
  git submodule update
```

We note that a particular benefit of retrieving these dependencies using the submodules is that these point to specific revisions of both libraries that are known to work correctly with the codes in this repository.

### Building the Dependencies

We recommend that users follow the posted instructions for installing both SUNDIALS and Gkeyll.

> Note: Deactivate any environments (e.g., Spack or environment modules) that may conflict with the installations below before installing.

#### GkeyllZero

[The GkeyllZero build instructions are linked here](https://gkeyll.readthedocs.io/en/latest/install.html).  Note that these instructions are for Gkeyll as a whole, only a subset of these instructions pertain to GkeyllZero.

GkeyllZero uses a Makefile-based build system, that relies on "machine files" for configuration.  For systems where existing machine files can be used, we recommend that users follow the "Gkeyll build instructions" linked above.  We recommend that the same MPI library is used when building SUNDIALS, GkeyllZero's dependencies, GkeyllZero, and this repository, so it may be necessary to rebuild SUNDIALS above using the same MPI compiler wrappers as are used in the Gkeyll machine files.

The remainder of this section assumes that GkeyllZero has not been built on this machine before, and summarizes the minimal steps to install GkeyllZero and its dependencies into the `deps/gkylsoft` folder.  These closely follow the Gkeyll documentation steps for ["Installing from source manually"](https://gkeyll.readthedocs.io/en/latest/install.html#installing-from-source-manually), and so we omit explanation except where necessary.

We assume that this repository will be built using the `gcc`, `g++` and `gfortran` compilers, and that these are already in the user's current `$PATH`.

To install GkeyllZero and its dependencies, from the top-level folder for this repository,

```bash
cd deps
export GKYLSOFT=$PWD/gkylsoft
cd gkylzero/install-deps
./mkdeps.sh CC=gcc CXX=g++ FC=gfortran --prefix=$GKYLSOFT --build-superlu=yes --build-openmpi=yes --build-openblas=yes
cd ..
./configure CC=gcc --prefix=$GKYLSOFT --use-mpi=yes
make -j install
```

*Note: the `mkdeps.sh` command may fail with some more recent versions of CMake.  If that occurs, edit line 32 in `deps/gkylzero/install-deps/build-superlu.sh` to:*

```bash
cmake .. -DCMAKE_C_FLAGS="-g -O3 -fPIC" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_INSTALL_LIBDIR=lib -Denable_tests=NO -Denable_internal_blaslib=NO -DXSDK_ENABLE_Fortran=NO -DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

*Note: we've experienced OpenBLAS build failures on some Linux systems when using the `mkdeps.sh` command above.  If that occurs, you can omit that from the dependency build,

```bash
./mkdeps.sh CC=gcc CXX=g++ FC=gfortran --prefix=$GKYLSOFT --build-superlu=yes --build-openmpi=yes
```

you should install OpenBLAS separately on your system (e.g., using `apt-get` or `yum`), and then in the subsequent `configure` line you should manually specify `include` and `library` locations for where the OpenBLAS-installed `lapacke` static library and headers were installed, e.g.

```bash
./configure CC=gcc --prefix=$GKYLSOFT --use-mpi=yes --lapack-lib=/usr/lib/x86_64-linux-gnu/liblapacke.a --lapack-inc=/usr/include/x86_64-linux-gnu/
```

#### SUNDIALS

[The SUNDIALS build instructions are linked here](https://sundials.readthedocs.io/en/latest/sundials/Install_link.html#building-and-installing-with-cmake).  Note that of the many SUNDIALS build options, this repository requires only a minimal SUNDIALS build with (**required**) MPI.

The following steps can be used to build SUNDIALS using a minimal configuration that leverages the dependencies that were already installed by Gkeyll:

```bash
mkdir deps/sundials/build
cd deps/sundials/build
cmake -DCMAKE_INSTALL_PREFIX=../../sundials-install -DENABLE_MPI=ON -DSUNDIALS_INDEX_SIZE=32 -DMPI_C_COMPILER=$GKYLSOFT/openmpi-4.1.6/bin/mpicc -DMPIEXEC_EXECUTABLE=$GKYLSOFT/openmpi-4.1.6/bin/mpiexec ..
make -j install
```

### Building the CMake-based test (`diffusion_2D`)

The CMake-based test problem follows the standard pattern for CMake-based projects: in-source builds are not permitted, so the code should be configured and built from a separate build directory, e.g.,

```bash
  mkdir ceda-demonstrations/build
  cd ceda-demonstrations/build
  cmake -DSUNDIALS_ROOT="[sundials-path]" ..
  make -j install
```

where `[sundials-path]` is the path to the top-level folder containing the SUNDIALS installation.  Upon completion of these commands, the executables for each test problem are saved in the `ceda-demonstrations/bin` directory.

If both SUNDIALS and Gkeyll were installed using the submodule-based instructions above, then the following commands should be sufficient:

```bash
  mkdir ceda-demonstrations/build
  cd ceda-demonstrations/build
  cmake -DSUNDIALS_ROOT=../deps/sundials-install -DCMAKE_CXX_COMPILER=$GKYLSOFT/openmpi/bin/mpicxx -DCMAKE_C_COMPILER=$GKYLSOFT/openmpi/bin/mpicc ..
  make -j install
```


### Building the Makefile-based tests (`gkeyll_diffusion`)

The examples that leverage GkeyllZero do not currently use CMake for compilation, and must be handled separately.

Assuming that both SUNDIALS and GkylZero were installed following the above instructions, then the following commands will build the `gkeyll_diffusion` example:

```bash
  cd ceda-demonstrations/gkeyll_diffusion
  export CC=$GKYLSOFT/openmpi/bin/mpicc
  make
```


### Running the paper tests ###

Computational experiments using the executables built in the preceding steps are run below.  These are run using pairs of Python scripts.  The first script in each folder runs the corresponding code with a wide range of input parameters, and stores the resulting runtime statistics to disk.  These are then followed with plotting scripts, that load the runtime statistics and generate plots.

These Python scripts depend on a small number of widely-used packages: `matplotlib`, `pandas`, and `openpyxl`. If these are not already installed on your system, then we recommend that you create a Python virtual environment, install the packages specified in the top-level `requirements.txt` file,

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python_requirements.txt
```

and launch these Python scripts from within that virtual environment.


#### `diffusion_2D` tests ####

After building the executables using the above instructions, the full set of finite-difference-based 2D diffusion tests may be run using the commands from the top-level repository directory.  The MPI-parallel runs should be launched with the `mpiexec` corresponding to the gkylzero and SUNDIALS builds; this can be put in the `MPIEXEC` environment variable, which will be used when running the tests:

```bash
export MPIEXEC=$GKYLSOFT/openmpi/bin/mpiexec
python ./bin/runtests-diffusion2d.py
python ./bin/makeplots-diffusion2d.py
```

*Note: these tests expect that your system can run MPI-parallel simulations using up to 64 CPU cores.  If your system is smaller, then you should edit lines 89-92 in `bin/runtests-diffusion2d.py` to remove inputs with `np` that exceeds your available resources.*

The `runtests` script runs a wide range of tests using different diffusion constants, grids, and time integration methods, storing all results in a Pandas dataframe, and then saving that to the files `results_diffusion_2D.xlsx` and `results_diffusion_2D_fixedstep.xlsx`.  The `plot` script reads this file and generates the relevant plots in the above-referenced paper.


On some Linux systems, CMake may not automatically embed the path to the SUNDIALS libraries into the executable, causing runtime errors of the form

```
./bin/diffusion_2D_mpi: error while loading shared libraries: libsundials_arkode.so.6: cannot open shared object file: No such file or directory
```

This may be fixed by adding the SUNDIALS library folder to the `LD_LIBRARY_PATH` environment variable before running the tests, e.g.,

```bash
export MPIEXEC=$GKYLSOFT/openmpi/bin/mpiexec
export LD_LIBRARY_PATH=$GKYLSOFT/../sundials-install/lib:$LD_LIBRARY_PATH
python ./bin/runtests-diffusion2d.py
python ./bin/makeplots-diffusion2d.py
```


#### `gkeyll_diffusion` tests ####

After building the executables using the above instructions, the full set of discontinuous Galerkin 2D diffusion tests may be run using the commands from the top-level repository directory:

```bash
cd gkeyll_diffusion
python ./runtests-gk_diffusion_1x1v_p1.py
```

The `runtests` script runs a wide range of tests using different diffusion constants and time integration methods, storing all results in a Pandas dataframe, and then saving that to the files `full_results_gk_diffusion_1x1v_p1_adaptive.xlsx` and `full_results_gk_diffusion_1x1v_p1_fixed.xlsx`.  Due to different mechanisms for processing command-line options between the underlying Gkeyll infrastructure and the `main` routine that runs the tests, this script will output multiple lines of the form

```
/gk_diffusion_1x1v_p1: illegal option --
```

Those warning messages can safely be ignored.  The `plot` script reads these `.xlsx` files, and generates the relevant plots in the above-referenced paper. After the complition of the `runtests` run, plots can be obtained by running corresponding scripts in the scripts directory:

```bash
cd scripts
python Fig7_DomEig_comparison.py # For the Figure 7
```

