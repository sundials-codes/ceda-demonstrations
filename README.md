# Super-time-stepping Demonstration Codes

[Note: this project is in active development.]

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

* optionally, the NVIDIA [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (**if building with GPU support**)


The codes in this repository depend on two external libraries:

* [SUNDIALS](https://github.com/LLNL/sundials)

* [GkeyllZero](https://github.com/ammarhakim/gkylzero) -- note that this is a subset of [Gkeyll](https://github.com/ammarhakim/gkyl)

* [PostGkeyll](https://github.com/ammarhakim/postgkyl) -- this is only used for postprocessing results from Gkeyll-based runs

If these are not already available on your system, they may be cloned from GitHub as submodules.  After cloning this repository using the command above, you can retrieve these submodules via:

```bash
  cd ceda-demonstrations/deps
  git submodule init
  git submodule update
```

We note that a particular benefit of retrieving these dependencies using the submodules is that these point to specific revisions of both libraries that are known to work correctly with the codes in this repository.


### Building the Dependencies

We recommend that users follow the posted instructions for installing both SUNDIALS and Gkeyll.

#### SUNDIALS

[The SUNDIALS build instructions are linked here](https://sundials.readthedocs.io/en/latest/sundials/Install_link.html#building-and-installing-with-cmake).  Note that of the many SUNDIALS build options, this repository requires only a minimal SUNDIALS build with:

* MPI (**required**) -- note that if building with either CUDA or HIP GPU support then the MPI implementation is assumed to be GPU-aware

* *[HYPRE](https://github.com/hypre-space/hypre)* (**optional**, for enabling multigrid preconditioning)

* *CUDA Toolkit >=12.0* (**optional**, for building with NVIDIA GPU support)

* *HIP >=5.0.0* (**optional**, for building with AMD GPU support)

The following steps can be used to build SUNDIALS using a minimal configuration (without the optional features above):

```bash
mkdir deps/sundials/build
cd deps/sundials/build
cmake -DCMAKE_INSTALL_PREFIX=../../sundials-install -DENABLE_MPI=ON -DSUNDIALS_INDEX_SIZE=32 ..
make -j install
```

Alternately, if CMake is able to find both *hypre* and CUDA automatically (e.g., these were enabled via `module load` or `spack load` on a system where [Linux environment modules](https://modules.readthedocs.io/en/latest/) and/or [Spack](https://spack.readthedocs.io/en/latest/) are available), a build that enables both *hypre* and CUDA may be possible via the steps:

```bash
mkdir deps/sundials/build
cd deps/sundials/build
cmake -DCMAKE_INSTALL_PREFIX=../../sundials-install -DENABLE_MPI=ON -DSUNDIALS_INDEX_SIZE=32 -DENABLE_CUDA=ON -DENABLE_HYPRE=ON ..
make -j install
```

Instructions for building SUNDIALS with additional options (including *hypre*, CUDA and HIP) [may be found here](https://sundials.readthedocs.io/en/latest/sundials/Install_link.html).

#### GkeyllZero

[The GkeyllZero build instructions are linked here](https://gkeyll.readthedocs.io/en/latest/install.html).  Note that these instructions are for Gkeyll as a whole, only a subset of these instructions pertain to GkeyllZero.

GkeyllZero uses a Makefile-based build system, that relies on "machine files" for configuration.  For systems where existing machine files can be used, we recommend that users follow the "Gkeyll build instructions" linked above.  We recommend that the same MPI library is used when building SUNDIALS, GkeyllZero's dependencies, GkeyllZero, and this repository, so it may be necessary to rebuild SUNDIALS above using the same MPI compiler wrappers as are used in the Gkeyll machine files.

The remainder of this section assumes that GkeyllZero has not been built on this machine before, and summarize the minimal steps to install GkeyllZero and its dependencies into the `deps/gkyl-install` folder.  These closely follow the Gkeyll documentation steps for ["Installing from source manually"](https://gkeyll.readthedocs.io/en/latest/install.html#installing-from-source-manually), and so we omit explanation except where necessary.

We assume that SUNDIALS was already installed with MPI support, using the `mpicc` and `mpicxx` compiler wrappers that are already in the user's current `$PATH`.

To install GkeyllZero and its dependencies (without CUDA), from the top-level folder for this repository,

```bash
cd deps
export GKYLSOFT=$PWD/gkyl-install
cd gkylzero/install-deps
./mkdeps.sh CC=mpicc CXX=mpicxx FC=mpif90 MPICC=mpicc MPICXX=mpicxx --prefix=$GKYLSOFT --build-openblas=yes --build-superlu=yes
cd ..
./configure CC=mpicc --prefix=$GKYLSOFT
make -j install
```

#### PostGkeyll

Assuming that you downloaded all of the relevant submodules above, then we recommend that you set up a Python virtual environment to install PostGkeyll.  Similarly to the [posted installation instructions](https://github.com/ammarhakim/postgkyl), from the top-level folder in this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
cd deps/postgkyl
pip install -e .[adios,test]
```

After this installation is complete, you can "test" the installation by running

```bash
pytest [-v]
```

You may "deactivate" this Python environment from your current shell with the command

```bash
deactivate
```

and in the future you can "reactivate" the python environment in your shell by running from the top-level directory of this repository

```bash
source .venv/bin/activate
```


### Building the CMake-based tests (`diffusion_2D` and `adr_1D`)

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
  cmake -DSUNDIALS_ROOT=../deps/sundials-install ..
  make -j install
```

If SUNDIALS was installed with *hypre* support, then the configuration above should be changed slightly:

```bash
  mkdir ceda-demonstrations/build
  cd ceda-demonstrations/build
  cmake -DSUNDIALS_ROOT=../deps/sundials-install -DUSE_HYPRE=ON ..
  make -j install
```


### Building the Makefile-based tests (`gkeyll_diffusion`)

The examples that leverage GkeyllZero do not currently use CMake for compilation, and must be handled separately.

Assuming that both SUNDIALS and GkylZero were installed following the above instructions, then the following commands will build the `gkeyll_diffusion` example:

```bash
  cd ceda-demonstrations/gkeyll_diffusion
  export CC=mpicc
  make
```
