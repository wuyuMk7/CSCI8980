#add_executable(test test.cc)

add_subdirectory(glad)
add_subdirectory(glfw)
add_subdirectory(glm)

add_executable(main main.cc)
target_link_libraries(main PUBLIC glad glfw)

set_target_properties(
    main 
    PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
