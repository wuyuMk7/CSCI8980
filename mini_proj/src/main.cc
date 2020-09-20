#include "main.h"

int main(int argc, char* argv[])
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

    GLFWwindow* window = glfwCreateWindow(800, 600, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
       std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); 

    // Load shaders
    Shader car_shader;
    car_shader.loadByFile("./shaders/shape.vs", "./shaders/shape.fs");

    car_shader.use();
    glm::mat4 view(1.0f);
    // view = glm::translate(view, glm::vec3(0.0f, 0.0f, 3.0f));
    GLuint vi = glGetUniformLocation(car_shader.id(), "view");
    glUniformMatrix4fv(vi, 1, GL_FALSE, glm::value_ptr(view));

    glm::mat4 projection = glm::perspective(
        glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 10.0f
    );
    // glm::mat4 projection = glm::ortho(-100.0f, 800.0f, -100.0f, 600.0f, -10.0f, 50.0f);
    GLuint proj = glGetUniformLocation(car_shader.id(), "projection");
    glUniformMatrix4fv(proj, 1, GL_FALSE, glm::value_ptr(projection));

    // Create a rectangle
    Rectangle car_rec(car_shader, glm::vec3(-0.5f, 0.5f, 0.0f), glm::vec3(0.5f, -0.5f, 0.0f));
    Car car(car_shader, car_rec, 1.0f);
    car.color(glm::vec3(0.5f, 0.0f, 0.0f));
  
    while(!glfwWindowShouldClose(window))
    {
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        car.draw();

        glfwSwapBuffers(window);
        glfwPollEvents();    
    }

    glfwTerminate();

    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
} 

void processInput(GLFWwindow *window)
{
    // WSAD Control
    // if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    //     camera.moveForward(deltaFrameTime);
    // if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    //     camera.moveBackward(deltaFrameTime);
    // if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    //     camera.moveLeft(deltaFrameTime);
    // if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    //     camera.moveRight(deltaFrameTime);

    // if (glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS)
    //     camera.pitchAndYaw(1.0, 0.0);
    // if (glfwGetKey(window, GLFW_KEY_8) == GLFW_PRESS)
    //     camera.pitchAndYaw(-1.0, 0.0);
    // if (glfwGetKey(window, GLFW_KEY_9) == GLFW_PRESS)
    //     camera.pitchAndYaw(0.0, 1.0);
    // if (glfwGetKey(window, GLFW_KEY_0) == GLFW_PRESS)
    //     camera.pitchAndYaw(0.0, -1.0);
}