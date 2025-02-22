cmake_minimum_required (VERSION 3.10)
project (FDDiffusion)

find_package(SUNDIALS REQUIRED)

add_executable(fd_diffusion fd_diffusion.c)

target_link_libraries(fd_diffusion PUBLIC
  SUNDIALS::core SUNDIALS::arkode m)