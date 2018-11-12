cmake_minimum_required(VERSION 3.12)
project(mapgen_test)

find_package( OpenCV REQUIRED)

include_directories("/usr/local/include/eigen3/")

set(CMAKE_CXX_STANDARD 11)

add_executable(mapgen main.cpp)
add_executable(mk_map2 map2.cpp)
target_link_libraries(mapgen ${OpenCV_LIBS})
target_link_libraries(mk_map2 ${OpenCV_LIBS})