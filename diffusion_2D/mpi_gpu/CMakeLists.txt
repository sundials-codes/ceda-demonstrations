set(sources ${shared_sources}
  buffers.cpp diffusion.cpp solution.cpp utils.cpp)

if(USE_CUDA)
  foreach(file ${sources})
    set_source_files_properties(${file} PROPERTIES LANGUAGE CUDA)
  endforeach()
endif()

# set the target name
set(target diffusion_2D_mpigpu)

# create executable
add_executable(${target} ${sources})

target_include_directories(${target} PRIVATE ${demonstration_prefix})

target_link_libraries(${target} PRIVATE SUNDIALS::core
  SUNDIALS::arkode SUNDIALS::nvecmpiplusx MPI::MPI_CXX)

if (USE_CUDA)
  target_compile_definitions(${target} PRIVATE USE_CUDA)
  target_link_libraries(${target} PRIVATE SUNDIALS::nveccuda)
else()
  target_compile_definitions(${target} PRIVATE USE_HIP)
  target_link_libraries(${target} PRIVATE SUNDIALS::nvechip hip::device)
endif()
