cmake_minimum_required(VERSION 3.0.0)
project(openmp VERSION 0.1.0)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories( ${OpenCV_INCLUDE_DIRS})

add_executable(openmp openmp.cpp)

target_link_libraries(openmp ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)