cmake_minimum_required (VERSION 3.20)

project (CEDA-demonstrations C CXX)

# prohibit in-source build
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source build prohibited.")
endif()

# default build type
set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type")

# Show/hide compiler and flags based on the current build type
macro(show_language_flags lang)
  set(_build_types DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _cmake_build_type)
  foreach(build_type ${_build_types})
    if(${_cmake_build_type} MATCHES "${build_type}")
      message(STATUS "Appending ${lang} ${build_type} flags: ${CMAKE_${lang}_FLAGS_${build_type}}")
      mark_as_advanced(CLEAR CMAKE_${lang}_FLAGS_${build_type})
    else()
      mark_as_advanced(FORCE CMAKE_${lang}_FLAGS_${build_type})
    endif()
  endforeach()
  # show the language compiler and flags
  mark_as_advanced(CLEAR CMAKE_${lang}_COMPILER CMAKE_${lang}_FLAGS)
  # print implicit includes
  message(STATUS "Implicit ${lang} includes: ${CMAKE_${lang}_IMPLICIT_INCLUDE_DIRECTORIES}")
endmacro()

# Simple function to add a given prefix to a list of files
function(add_prefix var prefix)
  set(result)
  foreach(item ${${var}})
    list(APPEND result "${prefix}${item}")
  endforeach()
  set(${var} ${result} PARENT_SCOPE)
endfunction()

show_language_flags(CXX)
set(CMAKE_CXX_STANDARD "17" CACHE STRING "C++ standard to use")

# Append to the runtime search path (rpath) of installed binaries any
# directories outside the project that are in the linker search path or contain
# linked library files.
option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Set rpath" ON)
mark_as_advanced(FORCE CMAKE_INSTALL_RPATH_USE_LINK_PATH)

# set installation folders
set(demonstration_bin "${CMAKE_CURRENT_SOURCE_DIR}/bin")

# -------------
# Build options
# -------------

option(USE_CUDA "Use CUDA for GPU acceleration" OFF)

option(USE_HIP "Use HIP for GPU acceleration" OFF)

if(USE_CUDA)
  enable_language(CUDA)
  show_language_flags(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "CUDA Architecture")
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
endif()

# ----
# MPI
# ----

find_package(MPI REQUIRED)

# ---------
# SUNDIALS
# ---------

set(SUNDIALS_ROOT "$ENV{SUNDIALS_ROOT}"
  CACHE PATH "Root directory of SUNDIALS install")

find_package(SUNDIALS REQUIRED
  PATHS
  "${SUNDIALS_ROOT}/lib64/cmake/sundials"
  "${SUNDIALS_ROOT}/lib/cmake/sundials"
  "${SUNDIALS_ROOT}")

message(STATUS "Found SUNDIALS: ${SUNDIALS_DIR}")

# -----
# HYPRE
# -----

option(USE_HYPRE "Use HYPRE for preconditioning" OFF)
if(USE_HYPRE)
  # Forced to OFF for STS tetings, but left the USE_HYPRE option for the source code
  set(USE_HYPRE OFF CACHE BOOL "Use HYPRE for preconditioning" FORCE)
  message(STATUS "HYPRE support disabled for STS testing.")
endif()

# --------------------------------------------------------
# add subdirectories for specific demonstration test codes
# --------------------------------------------------------

add_subdirectory(diffusion_2D)
