cmake_minimum_required(VERSION 3.14)
project(ai_algorithms VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include_directories(include)

# Neural Network library
add_library(neural_network src/neural_network.cpp)
target_include_directories(neural_network PUBLIC include)

# Simulated Annealing library
add_library(simulated_annealing src/simulated_annealing.cpp)
target_include_directories(simulated_annealing PUBLIC include)

# Examples
add_executable(nn_example examples/nn_example.cpp)
target_link_libraries(nn_example PRIVATE neural_network)

add_executable(sa_example examples/sa_example.cpp)
target_link_libraries(sa_example PRIVATE simulated_annealing)
