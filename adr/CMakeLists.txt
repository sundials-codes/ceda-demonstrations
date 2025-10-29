### Set up C++ executable for 1D advection_diffusion_reaction ###

set(sources ${shared_sources}
  advection_diffusion_reaction.cpp)

# set the target name
set(target advection_diffusion_reaction_1D)

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
set(target_2d advection_diffusion_reaction_2D)

# create executable
add_executable(${target_2d} ${sources_2d})

target_include_directories(${target_2d} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/")

target_link_libraries(${target_2d} PRIVATE
  SUNDIALS::arkode SUNDIALS::nvecserial)

#   request installation of the target into the overall "bin" folder
install(TARGETS ${target_2d} DESTINATION ${demonstration_bin})


### Set up PIROCK executable ###

enable_language(Fortran)

set(sources_pirock_adr1D dr_adr_1D.f)
set(target_pirock_adr1D advection_diffusion_reaction_1D_pirock)
add_executable(${target_pirock_adr1D} ${sources_pirock_adr1D})

set(sources_pirock_bruss2dadv dr_bruss2dadv.f)
set(target_pirock_bruss2dadv advection_diffusion_reaction_2D_pirock)
add_executable(${target_pirock_bruss2dadv} ${sources_pirock_bruss2dadv})

set(sources_pirock_bruss2dreac dr_bruss2dreac.f)
set(target_pirock_bruss2dreac reaction_diffusion_2D_pirock)
add_executable(${target_pirock_bruss2dreac} ${sources_pirock_bruss2dreac})

# Detect compiler and set legacy flags accordingly
get_filename_component(Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)

if(Fortran_COMPILER_NAME MATCHES "gfortran.*")
  # gfortran legacy flags, e.g., fixed form source
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy")
endif()

#   request installation of the target into the overall "bin" folder
install(TARGETS ${target_pirock_adr1D} DESTINATION ${demonstration_bin})


### install testing scripts and input file ###
install(FILES runtests-adr1d.py plot-adr1d.py adr_1D_pirock_params.txt DESTINATION ${demonstration_bin})
