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
  git clone -b extsts https://github.com/sundials-codes/ceda-demonstrations.git
```

All remaining steps in these instructions assume that you are sitting inside the top-level folder that was cloned in this step:

```bash
  cd ceda-demonstrations
```

### Requirements

To compile the codes in this repository you will need:

* [CMake](https://cmake.org) 3.20 or newer (both for SUNDIALS and for this repository)

* C compiler (C99 standard) and C++ compiler (C++11 standard)

The codes for this publication depend on one external library:

* [SUNDIALS](https://github.com/LLNL/sundials)

If this is not already available on your system, it may be cloned from GitHub as a submodule.  After cloning this repository using the command above, you can retrieve this submodule via:

```bash
  cd deps
  git submodule init
  git submodule update
  cd ..
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
cd -
```

Instructions for building SUNDIALS with additional options [may be found here](https://sundials.readthedocs.io/en/latest/sundials/Install_link.html).

### Building the tests for the paper

The codes for this paper follow the standard pattern for CMake-based projects: in-source builds are not permitted, so the code should be configured and built from a separate build directory, e.g.,

```bash
  mkdir build
  cd build
  cmake -DSUNDIALS_ROOT="[sundials-path]" -DCMAKE_BUILD_TYPE=Release ..
  make -j install
  cd -
```

where `[sundials-path]` is the path to the top-level folder containing the SUNDIALS installation.  Upon completion of these commands, the executables for each test problem are saved in the `ceda-demonstrations/bin` directory.

If SUNDIALS was installed using the submodule-based instructions above, then the following commands should be sufficient:

```bash
  mkdir build
  cd build
  cmake -DSUNDIALS_ROOT=../deps/sundials-install -DCMAKE_BUILD_TYPE=Release ..
  make -j install
  cd -
```

### Running the tests for the paper ###

The source code files for this paper are contained in the folder `adr`.  After building the executables using the above instructions, the executables and Python run/plot scripts are in the folder `bin`.  The full set of 1D and 2D tests include a wide range of parameters and time integration methods.  These scripts will store all results in Pandas dataframes, and then save those results to a set of `.xlsx` files into the `data` folder. *Note: these scripts can take some time to complete,* so the `data` folder already contains a set of `.xlsx` files that we generated for the paper and that can be used to create the plots in the paper.  These data files will be overwritten if you run the test scripts again.

To run the full set of tests yourself, use the following commands from the top-level repository directory:

```bash
python ./bin/runtests-adr1d.py
python ./bin/runtests-adr2d.py
```

*Note: on Linux systems, you may need to update your `LD_LIBRARY_PATH` to include the installation folder for the SUNDIALS libraries, e.g.*

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/deps/sundials-install/lib
```
*before running the above `runtests-X` commands.*

Once these complete writing the new `.xlsx` files in the `data` folder, the plots for the paper may be generated with the commands:

```bash
python ./bin/plot-adr1d.py
python ./bin/plot-adr2d.py
```

These plots will be stored in a new top-level `plots` folder.
