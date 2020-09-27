cmake_minimum_required(VERSION 3.10)

project(Mini_Project VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

include_directories(SYSTEM src/glad/include)
include_directories(SYSTEM src/glfw/include)
include_directories(SYSTEM src/glm/glm)
include_directories(SYSTEM src/xtl/include)
include_directories(SYSTEM src/xsimd/include)
include_directories(SYSTEM src/xtensor/include)
    
add_subdirectory(src)
