cmake_minimum_required(VERSION 2.8)

project(ERL)

set(ERL_VERSION 1)

configure_file("${PROJECT_SOURCE_DIR}/source/ERL/ERLConfig.h.in" "${PROJECT_SOURCE_DIR}/source/ERL/ERLConfig.h")

include_directories("${PROJECT_SOURCE_DIR}/source")

# This is only required for the script to work in the version control
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
 
find_package(OpenCL REQUIRED)
 
include_directories(${OPENCL_INCLUDE_DIRS})
 
if(OPENCL_HAS_CPP_BINDINGS)
    message("OpenCL has CPP bindings. Full include is: " ${OPENCL_INCLUDE_DIRS})
else(OPENCL_HAS_CPP_BINDINGS)
    message("No OpenCL CPP bindings found")
endif(OPENCL_HAS_CPP_BINDINGS)

add_executable(ERL "${PROJECT_SOURCE_DIR}/source/Main.cpp")

target_link_libraries(ERL ${OPENCL_LIBRARIES})