set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)

add_subdirectory(glad)
add_subdirectory(glfw)
#add_subdirectory(glm)

#set(xtl_DIR ${CMAKE_BINARY_DIR}/src/xtl)
#set(xsimd_DIR ${CMAKE_BINARY_DIR}/src/xsimd)
#set(xtensor_DIR ${CMAKE_BINARY_DIR}/src/xtensor)

#add_subdirectory(xtl)
#add_subdirectory(xsimd)
#add_subdirectory(xtensor)

#find_package(xtl REQUIRED)
#find_package(xtensor REQUIRED)

add_compile_options(-O3 -mavx2 -ffast-math)

add_executable(main main.cc shader.cc obstacle.cc rectangle.cc car.cc rl.cc) 
target_compile_options(main PRIVATE -march=native)
target_link_libraries(main PUBLIC glad glfw ${BLAS_LIBRARIES})

set_target_properties(
    main 
    PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

configure_file(shaders/shape.vs ${CMAKE_BINARY_DIR}/shaders/shape.vs COPYONLY)
configure_file(shaders/shape.fs ${CMAKE_BINARY_DIR}/shaders/shape.fs COPYONLY)
