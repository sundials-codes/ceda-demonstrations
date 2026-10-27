# CEDA Demonstration Codes

[Note: this project is in active development.]

A separate `locked` branch is allocated for each publication:

* [STS_diffusion_with_DG](https://github.com/sundials-codes/ceda-demonstrations/tree/STS_diffusion_with_DG) branch contains all testing code for the article:\
Aggul, M., Francisquez, M., Reynolds, D.R., Amihere, S., "Super Time Stepping Methods for Diffusion using Discontinuous-Galerkin Spatial Discretizations," 2026, [arXiv:2601.14508](https://arxiv.org/abs/2601.14508)

To run the test codes related to the above publication(s), we recommend that you first checkout the relevant branch, and follow the instructions in the `README.md` therein.


This is a repository of [SUNDIALS](https://github.com/LLNL/sundials)- and [Gkeyll](https://github.com/ammarhakim/gkeyll)-based applications to assess and demonstrate the performance of new time integration methods that have been added to SUNDIALS as part of the [CEDA SciDAC project](https://sites.google.com/pppl.gov/ceda-scidac-5?usp=sharing).


## Installation

The following steps describe how to build these codes in a Linux or OS X environment.


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

* optionally, the NVIDIA [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (**if building with GPU support**)


The codes in this repository depend on two external libraries:

* [SUNDIALS](https://github.com/LLNL/sundials)

* [Gkeyll](https://github.com/ammarhakim/gkylzero)

In particular, we use some specific branches from these repositories, so even if these are already available on your system, we recommend that you follow the build instructions below to obtain and build these.

```bash
  cd ceda-demonstrations/deps
  git submodule init
  git submodule update
```

We note that a particular benefit of retrieving these dependencies using the submodules is that these point to specific revisions of both libraries that are known to work correctly with the codes in this repository.  If *hypre* is desired, then any recent version (e.g., v2.20.0 or higher) should work, but you must install this separately.


### Building the Dependencies

We recommend that users follow the posted instructions for installing both Gkeyll and SUNDIALS.  As currently configured, Gkeyll will download and install SUNDIALS as a dependency, so we only need to specify the appropriate Gkeyll build instructions.

1. The first step is to download Gkeyll.  This may be retrieved either as a submodule of this repository,

```bash
     cd ceda-demonstrations/deps
     git submodule init
     git submodule update
```

   or by cloning the relevant branch directly,

```bash
     cd ceda-demonstrations/deps
     git clone -b sundials-ssp_sts https://github.com/ammarhakim/gkeyll
```

2. Go into the new folder,

```bash
     cd gkeyll
```

3. When building on MacOS, edit `machines/mkdeps.macos.sh` and `machines/configure.macos.sh` to have

```bash
     : "${PREFIX:=<gkeyll_dir>/gkylsoft}"
```

   near the top.

4. Build the Gkeyll dependencies,

```bash
     ./machines/mkdeps.macos.sh
```

   Note that this step automatically builds dependencies, including SUNDIALS (in the `feature/fes-get-stage-index` branch) using the script `gkeyll/install-deps/build-sundials.sh`.

5. Configure the Gkeyll build -- again focusing on MacOS,

```bash
     ./machines/configure.macos.sh
```

6. Start the Gkeyll build

```bash
     make -j4 install
```


### Building and running the Makefile-based tests


We've been using the regression test `gyrokinetic/creg/rt_gk_sheath_2x2v_p1.c`. First, build this via

```bash
   make -j4 build/gyrokinetic/creg/rt_gk_sheath_2x2v_p1
```

This may be run via

```bash
   build/gyrokinetic/creg/rt_gk_sheath_2x2v_p1
```

There are some command-line options, not many. Maybe the only relevant one is `-s` for running a specific number of steps. For example, to run 3 steps do

```bash
   build/gyrokinetic/creg/rt_gk_sheath_2x2v_p1 -s3
```

#### SUNDIALS-specific configuration

Below line 788 of `gyrokinetic/creg/rt_gk_sheath_2x2v_p1.c` there are some input-file SUNDIALS options. There may be others, just search for "sundials" in `gyrokinetic/apps/gkyl_gyrokinetic.h`, which is the main header file input files see. Every time the input file is changed one must recompile it with

```bash
   build/gyrokinetic/creg/rt_gk_sheath_2x2v_p1
```
The Gkeyll interface to SUNDIALS code is mainly contained in 2 files, `core/zero/sundials.c`, `gyrokinetic/apps/sundials_gyrokinetic.c`,
and their associated header files. These files contain the initialization of SUNDIALS solvers.

After the solvers are initialized, and after initial conditions are constructed, we call a `_reset` method in L999 of `core/zero/sundials.c`.

Then the code steps in time by with the `_update` function in L754 of `gyrokinetic/apps/gyrokinetic.c`

There are currently some print statements to see what's going on with the time step size, `dt`. Those can be turned on/off by uncommenting/commenting L14 of `core/zero/gkyl_sundials.h`. Every time this is changed one needs to re-build Gkeyll itself with

```bash
   make -j4 install
```

from within the top-level `deps/gkeyll` folder.


### Building the CMake-based tests (`diffusion_2D` and `adr`)

The CMake-based test problems follow the standard pattern for CMake-based projects: in-source builds are not permitted, so the code should be configured and built from a separate build directory, e.g.,

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

If SUNDIALS was installed with *hypre* support, then the configuration above should be changed slightly:

```bash
  mkdir ceda-demonstrations/build
  cd ceda-demonstrations/build
  cmake -DSUNDIALS_ROOT=../deps/sundials-install -DCMAKE_CXX_COMPILER=$GKYLSOFT/openmpi/bin/mpicxx -DCMAKE_C_COMPILER=$GKYLSOFT/openmpi/bin/mpicc -DUSE_HYPRE=ON ..
  make -j install
```
