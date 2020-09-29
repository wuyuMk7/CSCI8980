#include "main.h"

std::string model_filename = "savedModel.csv";
bool flag_retrain = false, flag_save_model = true, flag_draw_obstacle = true;

glm::vec3 pt_goal(500.0f, 400.0f, 0.0f), pt_starting(60.0f, 80.0f, 0.0f);

int main(int argc, char* argv[])
{
    if (argc >= 2) flag_retrain = true;

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

    GLFWwindow* window = glfwCreateWindow(800, 600, 
        "Mini Project (Press W-A-S-D to control, Esc to exit)", NULL, NULL);
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

    // Load shaders & set view matrices for car shader
    Shader car_shader;
    car_shader.loadByFile("./shaders/shape.vs", "./shaders/shape.fs");
    car_shader.use();

    glm::mat4 view(1.0f);
    // view = glm::translate(view, glm::vec3(0.0f, 0.0f, 3.0f));
    GLuint vi = glGetUniformLocation(car_shader.id(), "view");
    glUniformMatrix4fv(vi, 1, GL_FALSE, glm::value_ptr(view));

    glm::mat4 projection = glm::ortho(0.0f, 800.0f, 0.0f, 600.0f, 0.0f, 1000.0f);
    GLuint proj = glGetUniformLocation(car_shader.id(), "projection");
    glUniformMatrix4fv(proj, 1, GL_FALSE, glm::value_ptr(projection));

    // Create a rectangle
    glm::vec3 car_rec_tl(pt_starting.x - 25.0f, pt_starting.y + 15.0f, pt_starting.z),
      car_rec_br(pt_starting.x + 25.0f, pt_starting.y - 15.0f, pt_starting.z);
    Rectangle car_rec(car_shader, car_rec_tl, car_rec_br);
    Car car(car_shader, car_rec, 0.0f);
    car.color(glm::vec3(0.5f, 0.0f, 0.0f));

    Obstacle ob(car_shader, (pt_starting + pt_goal) / 2.0f, 30.0f);
    if (glm::length(pt_goal - pt_starting) > 150.0f) {
      ob.color(glm::vec3(1.0f, 1.0f, 1.0f));
      ob.init();
      car.setObstacle(ob);
      flag_draw_obstacle = true;
    }

    // Load model or Train model
    if (!std::filesystem::exists(model_filename) || flag_retrain) {
      car.setGoal(glm::vec3(400.0f, 200.0f, 0.0f));
      car.trainRL(8.0f, 0.01f);
      //car.setGoal(glm::vec3(200.0f, 300.0f, 0.0f));
      //car.trainRL(8.0f, 0.01f);
      //car.setGoal(glm::vec3(600.0f, 500.0f, 0.0f));
      //car.trainRL(8.0f, 0.01f);

      if (flag_save_model) {
        car.saveModel(model_filename);
      }
    } else {
      car.loadModel(model_filename);
      car.setFLTime(8.0f, 0.01f);
    }

    car.setGoal(pt_goal);
    car.runRL();
    car.printActions();
    car.printStates();

    size_t cur_state_index = 0;
    while(!glfwWindowShouldClose(window))
    {
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        processInput(window);

        if (flag_draw_obstacle) ob.draw();

        // Or use framerate to calcuate dt
        //car.update(0.01f);
        car.draw(cur_state_index);
        car.processInput(window);

        if (cur_state_index < car.curStatesSize() - 1) ++cur_state_index;

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
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}
