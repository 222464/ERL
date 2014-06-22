cmake_minimum_required(VERSION 2.8)

project(ERL)

set(ERL_VERSION 1)

configure_file("${PROJECT_SOURCE_DIR}/source/ERL/ERLConfig.h.in" "${PROJECT_SOURCE_DIR}/source/ERL/ERLConfig.h")

include_directories("${PROJECT_SOURCE_DIR}/source")

add_executable(ERL "${PROJECT_SOURCE_DIR}/source/Main.cpp")