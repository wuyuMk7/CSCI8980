#ifndef __MAIN_H__
#define __MAIN_H__

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <time.h>

#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "shader.h"
#include "car.h"

#include "gtc/matrix_transform.hpp"
#include "gtc/type_ptr.hpp"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

#endif
