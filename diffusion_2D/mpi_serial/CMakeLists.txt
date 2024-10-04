set(sources ${shared_sources}
  buffers.cpp diffusion.cpp solution.cpp utils.cpp)

# set the target name
set(target diffusion_2D_mpi)

# create executable
add_executable(${target} ${sources})

target_include_directories(${target} PRIVATE ${demonstration_prefix})

target_link_libraries(${target} PRIVATE
  SUNDIALS::arkode SUNDIALS::nvecparallel MPI::MPI_CXX)

if(USE_HYPRE)
  target_compile_definitions(${target} PRIVATE USE_HYPRE)
  target_link_libraries(${target} PRIVATE SUNDIALS::HYPRE)
endif()
