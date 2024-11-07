# Super-time-stepping Demonstration Codes

[Note: this project is in active development.]

This is a repository of [SUNDIALS](https://github.com/LLNL/sundials)-based applications to assess and demonstrate the parallel performance of new super-time-stepping (STS) method capabilities that have been added to SUNDIALS as part of the [CEDA SciDAC project](https://sites.google.com/pppl.gov/ceda-scidac-5?usp=sharing).


## Installation

The following steps describe how to build the demonstration code in a Linux or OS X environment.  **Note: these instructions are still under construction**


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

* [Gkeyll](https://github.com/ammarhakim/gkyl)

If these are not already available on your system, they may be cloned from GitHub as submodules.  After cloning this repository using the command above, you can retrieve these submodules via:

```bash
  cd ceda-demonstrations/deps
  git submodule init
  git submodule update
```

We note that a particular benefit of retrieving these dependencies using the submodules is that these point to specific revisions of both libraries that are known to work correctly with the codes in this repository.


### Building the Dependencies

We recommend that users follow the posted instructions for installing both SUNDIALS and Gkeyll:

* [SUNDIALS build instructions](https://sundials.readthedocs.io/en/latest/sundials/Install_link.html#building-and-installing-with-cmake).  Note that of the many SUNDIALS build options, this repository requires only a minimal SUNDIALS build with:

  * MPI

  * [HYPRE](https://github.com/hypre-space/hypre)

  Assuming that *hypre* has been installed in the folder `hypre_dir`, the following steps can be used to build SUNDIALS using this minimal configuration from the `deps/` folder:

  ```bash
  mkdir deps/sundials/build
  cd deps/sundials/build
  cmake -DCMAKE_INSTALL_PREFIX=../../sundials-install -DENABLE_MPI=ON -DSUNDIALS_INDEX_SIZE=32 -DENABLE_HYPRE -DHYPRE_DIR=hypre_dir ..
  make -j install
  ```

* [Gkeyll build instructions](https://gkeyll.readthedocs.io/en/latest/install.html).



### Configuration Options

Once the necessary dependencies are installed, the following CMake variables can be used to configure the demonstration code build:

* `CMAKE_INSTALL_PREFIX` - the path where executables and input files should be installed e.g., `my/install/path`. The executables will be installed in the `bin` directory and input files in the `tests` directory under the given path.

* `CMAKE_C_COMPILER` - the C compiler to use e.g., `mpicc`. If not set, CMake will attempt to automatically detect the C compiler.

* `CMAKE_C_FLAGS` - the C compiler flags to use e.g., `-g -O2`.

* `CMAKE_C_STANDARD` - the C standard to use, defaults to `99`.

* `CMAKE_CXX_COMPILER` - the C++ compiler to use e.g., `mpicxx`. If not set,
  CMake will attempt to automatically detect the C++ compiler.

* `CMAKE_CXX_FLAGS` - the C++ flags to use e.g., `-g -O2`.

* `CMAKE_CXX_STANDARD` - the C++ standard to use, defaults to `11`.

* `SUNDIALS_ROOT` - the root directory of the SUNDIALS installation, defaults to the value of the `SUNDIALS_ROOT` environment variable. If not set, CMake will attempt to automatically locate a SUNDIALS install on the system.

* `CMAKE_CUDA_COMPILER` - the CUDA compiler to use e.g., `nvcc`. If not set,
  CMake will attempt to automatically detect the CUDA compiler.

* `CMAKE_CUDA_FLAGS` - the CUDA compiler flags to use.

* `CMAKE_CUDA_ARCHITECTURES` - the CUDA architecture to target e.g., `70`.


### Building

Like most CMake-based projects, in-source builds are not permitted, so the code should be configured and built from a separate build directory, e.g.,

```bash
  mkdir ceda-demonstrations/build
  cd ceda-demonstrations/build
  cmake -DCMAKE_INSTALL_PREFIX="[install-path]" -DSUNDIALS_ROOT="[sundials-path] .."
  make -j install
```

where `[install-path]` is the path to where the binary and test input files should be installed and `[sundials-path]` is the path to the top-level folder containing the SUNDIALS installation.

If both SUNDIALS and Gkeyll were installed using the submodule-based instructions above, then the following commands should be sufficient to install into a new `ceda-demonstrations/install` directory:

```bash
  mkdir ceda-demonstrations/build
  cd ceda-demonstrations/build
  cmake -DCMAKE_INSTALL_PREFIX=../install -DSUNDIALS_ROOT=../deps/sundials-install -DGKYL_ROOT=../deps/gkyl-install .."
  make -j install
```
