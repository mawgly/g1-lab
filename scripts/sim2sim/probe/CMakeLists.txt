cmake_minimum_required(VERSION 3.16)
project(sim2sim_probe)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif ()

find_package(unitree_sdk2 REQUIRED)

add_executable(sim2sim_probe sim2sim_probe.cpp)
target_link_libraries(sim2sim_probe unitree_sdk2)
