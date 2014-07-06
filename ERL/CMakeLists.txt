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

find_package(Lua REQUIRED)
 
include_directories(${LUA_INCLUDE_DIR})

find_package(SFML 2 REQUIRED system window graphics network audio)
 
include_directories(${SFML_INCLUDE_DIR})

find_package(OpenGL REQUIRED)
 
include_directories(${OPENGL_INCLUDE_DIR})
 
add_executable(ERL "${PROJECT_SOURCE_DIR}/source/Main.cpp")

target_link_libraries(ERL ${OPENCL_LIBRARIES})
target_link_libraries(ERL ${LUA_LIBRARIES})
target_link_libraries(ERL ${SFML_LIBRARIES})
target_link_libraries(ERL ${OPENGL_LIBRARIES})