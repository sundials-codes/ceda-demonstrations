# Super-time-stepping Demonstration Codes

[Note: this project is in active development.]

This is a repository of [SUNDIALS](https://github.com/LLNL/sundials)-based applications to assess and demonstrate the parallel performance of new super-time-stepping (STS) method capabilities that have been added to SUNDIALS as part of the [CEDA SciDAC project](https://sites.google.com/pppl.gov/ceda-scidac-5?usp=sharing). Namely:


## Installation

The following steps describe how to build the demonstration code in a Linux or OS X environment.  **Note: these instructions are currently outdated**

### Gettting the Code

To obtain the code, clone this repository with Git:

```bash
  git clone https://github.com/sundials-codes/ceda-demonstrations.git
```

### Requirements

To compile the code you will need:

* [CMake](https://cmake.org) 3.20 or newer

* modern C and C++ compilers

* the NVIDIA [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

* an MPI library e.g., [OpenMPI](https://www.open-mpi.org/), [MPICH](https://www.mpich.org/), etc.

* the [SUNDIALS](https://computing.llnl.gov/projects/sundials) library of time integrators and nonlinear solvers


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

In-source builds are not permitted, as such the code should be configured and built from a separate build directory e.g.,
```bash
  cd ceda-demonstration
  mkdir build
  cd build
  cmake ../. \
    -DCMAKE_INSTALL_PREFIX="[install-path]" \
    -DSUNDIALS_ROOT="[sundials-path]"
  make
  make install
```
where `[install-path]` is the path to where the binary and test input files
should be installed and `[sundials-path]` is the path to the top-levle folder containing the SUNDIALS installation.


## Authors

[Daniel R. Reynolds](https://people.smu.edu/dreynolds),
Mustafa Aggul,
Sylvia Amihere,
Manaure Francisquez, and
Ammar Hakim.
