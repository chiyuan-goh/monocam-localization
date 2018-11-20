cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
    project(mapgen_test)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

include_directories("/usr/local/include/eigen3/")
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
link_directories(/usr/local/lib)


set(CMAKE_CXX_STANDARD 11)

add_definitions(${PCL_DEFINITIONS} )

add_executable(mapgen main.cpp)
add_executable(mk_map2 map2.cpp util.cpp)
add_executable(localize run_visualize.cpp util.cpp)
add_executable(test1 test.cpp util.cpp)

target_link_libraries(mapgen ${OpenCV_LIBS})
target_link_libraries(mk_map2 ${OpenCV_LIBS})
target_link_libraries(localize ${OpenCV_LIBS})
target_link_libraries(test1 ${PCL_LIBRARIES})



