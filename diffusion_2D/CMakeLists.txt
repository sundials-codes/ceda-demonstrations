# Shared sources
set(shared_sources main.cpp diffusion_2D.hpp diffusion_2D.cpp)

if (USE_HYPRE)
  list(APPEND shared_sources preconditioner_hypre.cpp)
else()
  list(APPEND shared_sources preconditioner_jacobi.cpp)
endif()

set(demonstration_prefix "${CMAKE_CURRENT_SOURCE_DIR}/")
add_prefix(shared_sources ${demonstration_prefix})

# MPI + Serial
add_subdirectory(mpi_serial)

# MPI + CUDA/HIP
if(USE_CUDA OR USE_HIP)
  add_subdirectory(mpi_gpu)
endif()
