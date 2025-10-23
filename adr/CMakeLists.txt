### Set up C++ executable for 1D advection_diffusion_reaction ###

set(sources ${shared_sources}
  advection_diffusion_reaction.cpp)

# set the target name
set(target advection_diffusion_reaction)

# create executable
add_executable(${target} ${sources})

target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/")

target_link_libraries(${target} PRIVATE
  SUNDIALS::arkode SUNDIALS::nvecserial)

#   request installation of the target into the overall "bin" folder
install(TARGETS ${target} DESTINATION ${demonstration_bin})


### Set up C++ executable for 2D advection_diffusion_reaction ###

set(sources_2d ${shared_sources}
  advection_diffusion_reaction_2d.cpp)

# set the target name
set(target_2d advection_diffusion_reaction_2d)

# create executable
add_executable(${target_2d} ${sources_2d})

target_include_directories(${target_2d} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/")

target_link_libraries(${target_2d} PRIVATE
  SUNDIALS::arkode SUNDIALS::nvecserial)

#   request installation of the target into the overall "bin" folder
install(TARGETS ${target_2d} DESTINATION ${demonstration_bin})


### Set up PIROCK executable ###

enable_language(Fortran)

set(sources_pirock dr_adr_1D.f)

# set the target name
set(target_pirock advection_diffusion_reaction_pirock)

# create executable
add_executable(${target_pirock} ${sources_pirock})

#set_target_properties(${target} PROPERTIES LINKER_LANGUAGE Fortran)

# Detect compiler and set legacy flags accordingly
get_filename_component(Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)

if(Fortran_COMPILER_NAME MATCHES "gfortran.*")
  # gfortran legacy flags, e.g., fixed form source
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy")
endif()

#   request installation of the target into the overall "bin" folder
install(TARGETS ${target_pirock} DESTINATION ${demonstration_bin})


### install testing scripts and input file ###
install(FILES runtests-adr1d.py plot-adr1d.py adr_1D_pirock_params.txt DESTINATION ${demonstration_bin})
