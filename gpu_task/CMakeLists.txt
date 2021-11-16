project(filtering CXX CUDA)
cmake_minimum_required(VERSION 3.8)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Set the default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

file(GLOB_RECURSE KER_INCS "kernel/*.cu" "kernel/*.h")
file(GLOB_RECURSE TOP_INCS "*.h")

source_group("kernel" FILES ${KER_INCS})

add_executable(app
        main.cc
        ${TOP_INCS}
        ${KER_INCS})

target_link_libraries(app ${CUDA_LIBRARIES})
set_target_properties(app PROPERTIES CUDA_ARCHITECTURES "75")
