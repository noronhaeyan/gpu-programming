cmake_minimum_required(VERSION 3.18)

# Default GPU architecture: 89 = Ada (RTX 6000 Ada, L4, RTX 40-series).
# Override at configure time, e.g.  cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90
# Must be set before project(), which otherwise picks the compiler's default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(cuda_kernels LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(SHOW_PTXAS "Print registers / shared memory / spills for every kernel" OFF)

# Shared headers (harness, checks, RAII buffers) used by every kernel.
add_library(common INTERFACE)
target_include_directories(common INTERFACE ${CMAKE_SOURCE_DIR}/common)
# -lineinfo maps profiler results (ncu) back to your source lines; it does not slow kernels down.
target_compile_options(common INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
if(SHOW_PTXAS)
  target_compile_options(common INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
endif()

# add_cuda_executable(<name> <sources...>): one executable per kernel project.
function(add_cuda_executable name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE common)
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endfunction()

add_cuda_executable(specs tools/specs.cu)
add_cuda_executable(vector_add 01_vector_add/vector_add.cu)
add_cuda_executable(transpose 02_transpose/transpose.cu)
# Add each new project here, e.g.:
# add_cuda_executable(transpose 02_transpose/transpose.cu)
