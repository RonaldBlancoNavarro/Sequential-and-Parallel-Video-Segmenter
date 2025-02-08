cmake_minimum_required(VERSION 3.10)
project(MPI_OpenMP_example)

# Encuentra las bibliotecas de MPI
find_package(MPI REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# Establece las opciones de compilación para C y C++
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Agrega los directorios de inclusión de MPI
include_directories(${MPI_INCLUDE_PATH})
include_directories( ${OpenCV_INCLUDE_DIRS})

# Agrega el ejecutable
add_executable(paralelo paralelo.cpp)

# Enlaza las bibliotecas de MPI
target_link_libraries(paralelo ${MPI_C_LIBRARIES} ${OpenCV_LIBS} OpenMP::OpenMP_CXX )