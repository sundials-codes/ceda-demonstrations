set(sources ${shared_sources}
  buffers.cpp diffusion.cpp initial.cpp main.cpp diffusion_2D.hpp diffusion_2D.cpp)

# set a target using the Jacobi preconditioner
set(standard_sources ${sources} preconditioner_jacobi.cpp)
set(target diffusion_2D_mpi)
add_executable(${target} ${standard_sources})
target_link_libraries(${target} PRIVATE
  SUNDIALS::arkode SUNDIALS::nvecparallel MPI::MPI_CXX)
install(TARGETS ${target} DESTINATION ${demonstration_bin})

# install plotting and testing scripts
install(FILES plot_data_2d.py runtests-diffusion2d.py DESTINATION ${demonstration_bin})

# if connected to HYPRE library, create a second target that uses HYPRE for preconditioning
if(USE_HYPRE)
  set(hypre_sources ${sources} preconditioner_hypre.cpp)
  set(target diffusion_2D_mpi_hypre)
  add_executable(${target} ${hypre_sources})
  target_compile_definitions(${target} PRIVATE USE_HYPRE)
  target_link_libraries(${target} PRIVATE
    SUNDIALS::arkode SUNDIALS::nvecparallel MPI::MPI_CXX SUNDIALS::HYPRE)
  install(TARGETS ${target} DESTINATION ${demonstration_bin})
endif()
