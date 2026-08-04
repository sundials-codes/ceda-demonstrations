# CEDA Demonstration Codes

[Note: this project is in active development.]

A separate `locked` branch is allocated for each publication:

* [STS_diffusion_with_DG](https://github.com/sundials-codes/ceda-demonstrations/tree/STS_diffusion_with_DG) branch contains all testing code for the article:\
Aggul, M., Francisquez, M., Reynolds, D.R., Amihere, S., "Super Time Stepping Methods for Diffusion using Discontinuous-Galerkin Spatial Discretizations," 2026, [arXiv:2601.14508](https://arxiv.org/abs/2601.14508)

* [ExtSTS](https://github.com/sundials-codes/ceda-demonstrations/tree/extsts) branch contains all testing code for the article:\
Reynolds, D.R., Amihere, S., Aggul, M., "Implicit-Explicit and Split-Explicit Super-Time-Stepping Methods," 2026, [arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)

To run the test codes related to the above publication(s), we recommend that you first checkout the relevant branch, and follow the instructions in the `README.md` therein.

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

The codes for this publication depend on one external library:

* [SUNDIALS](https://github.com/LLNL/sundials)

If this is not already available on your system, it may be cloned from GitHub as a submodule.  After cloning this repository using the command above, you can retrieve this submodule via:

```bash
  cd ceda-demonstrations/deps
  git submodule init
  git submodule update
```

We note that a particular benefit of retrieving this dependency as a submodule is that it points to specific revision of the library that is known to work correctly with the codes in this repository.

### Building the Dependency

We recommend that users follow the posted instructions for installing both SUNDIALS.

#### SUNDIALS

[The SUNDIALS build instructions are linked here](https://sundials.readthedocs.io/en/latest/sundials/Install_link.html#building-and-installing-with-cmake).  Note that of the many SUNDIALS build options, this repository requires only a minimal SUNDIALS.  The following steps can be used to build SUNDIALS using a minimal configuration that will work with the codes for this paper:

```bash
mkdir deps/sundials/build
cd deps/sundials/build
cmake -DCMAKE_INSTALL_PREFIX=../../sundials-install -DCMAKE_BUILD_TYPE=Release ..
make -j install
```

Instructions for building SUNDIALS with additional options [may be found here](https://sundials.readthedocs.io/en/latest/sundials/Install_link.html).

### Building the tests for the paper

The codes for this paper follow the standard pattern for CMake-based projects: in-source builds are not permitted, so the code should be configured and built from a separate build directory, e.g.,

```bash
  mkdir ceda-demonstrations/build
  cd ceda-demonstrations/build
  cmake -DSUNDIALS_ROOT="[sundials-path]" -DCMAKE_BUILD_TYPE=Release ..
  make -j install
```

where `[sundials-path]` is the path to the top-level folder containing the SUNDIALS installation.  Upon completion of these commands, the executables for each test problem are saved in the `ceda-demonstrations/bin` directory.

If SUNDIALS was installed using the submodule-based instructions above, then the following commands should be sufficient:

```bash
  mkdir ceda-demonstrations/build
  cd ceda-demonstrations/build
  cmake -DSUNDIALS_ROOT=../deps/sundials-install -DCMAKE_BUILD_TYPE=Release ..
  make -j install
```

### Running the tests for the paper ###

The codes for this paper are contained in the folder `adr`.  After building the executables using the above instructions, the full set of 1D and 2D test may be run using the commands from the top-level repository directory:

```bash
python ./bin/runtests-adr1d.py
python ./bin/runtests-adr2d.py
```

These scripts run a wide range of tests using different diffusion constants, grids, and time integration methods, storing all results in a Pandas dataframe, and then saving those results to a set of `.xlsx` files.  *Note: this repository already includes the `.xlsx` files that we generated and used to create the plots in the paper; the `runtests` scripts above will overwrite those files.*. Once these results files are in place, the plots for the paper may be generated with the commands:

```bash
python ./bin/plot-adr1d.py
python ./bin/plot-adr2d.py
```
