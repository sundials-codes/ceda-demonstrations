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
