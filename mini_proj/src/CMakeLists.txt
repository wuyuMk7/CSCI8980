#add_executable(test test.cc)

add_subdirectory(glad)
add_subdirectory(glfw)
add_subdirectory(glm)

add_executable(main main.cc shader.cc rectangle.cc car.cc) 
target_link_libraries(main PUBLIC glad glfw)

set_target_properties(
    main 
    PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

configure_file(shaders/shape.vs ${CMAKE_BINARY_DIR}/shaders/shape.vs COPYONLY)
configure_file(shaders/shape.fs ${CMAKE_BINARY_DIR}/shaders/shape.fs COPYONLY)
