#include "main.h"

std::string model_filename = "savedModel.csv";
bool flag_retrain = false, flag_save_model = true, flag_draw_obstacle = false;
bool allow_mouse_click = false, flag_next_goal = false;
bool flag_time_to_run = false;

glm::vec3 pt_goal(500.0f, 400.0f, 0.0f), pt_starting(60.0f, 80.0f, 0.0f), next_pt_goal;
float sim_time = 8.0f, dt = 0.05f;

int cur_state_index = -1;

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
  if (!allow_mouse_click) return;

  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
    double xPos, yPos;
    glfwGetCursorPos(window, &xPos, &yPos);
    //std::cout << xPos << " " << yPos << std::endl;
    next_pt_goal = glm::vec3((float)xPos, 600.0f - (float)yPos, 0.0f);
    flag_next_goal = true;
  }
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
    flag_time_to_run = true;
}

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
        "CSCI8980 Mini Project", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, keyCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
       std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

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
      //flag_draw_obstacle = true;
    }

    // Create array for goal point
    float *arr_goal_point = new float[6];
    arr_goal_point[0] = pt_goal.x;
    arr_goal_point[1] = pt_goal.y;
    arr_goal_point[2] = pt_goal.x-5.0f;
    arr_goal_point[3] = pt_goal.y-5.0f;
    arr_goal_point[4] = pt_goal.x+5.0f;
    arr_goal_point[5] = pt_goal.y-5.0f;

    GLuint goal_pt_vbo;
    glGenBuffers(1, &goal_pt_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, goal_pt_vbo);
    glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(float), arr_goal_point, GL_STATIC_DRAW);

    GLuint goal_pt_vao;
    glGenVertexArrays(1, &goal_pt_vao);
    glBindVertexArray(goal_pt_vao);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Load model or Train model
    clock_t start, end;
    if (!std::filesystem::exists(model_filename) || flag_retrain) {
      start = clock();
      //car.setGoal(glm::vec3(500.0f, 400.0f, 0.0f));
      car.setGoal(pt_goal);
      car.trainRL(sim_time, dt);
      end = clock();
      std::cout << "\nTraining costs: " << (double)(end-start) / CLOCKS_PER_SEC << " secs." << std::endl;
      /*
      car.setGoal(glm::vec3(200.0f, 300.0f, 0.0f));
      car.trainRL(sim_time, dt);
      car.setGoal(glm::vec3(400.0f, 200.0f, 0.0f));
      car.trainRL(sim_time, dt);
      car.setGoal(glm::vec3(530.0f, 210.0f, 0.0f));
      car.trainRL(sim_time, dt);
      car.setGoal(glm::vec3(300.0f, 320.0f, 0.0f));
      car.trainRL(sim_time, dt);
      */

      if (flag_save_model) {
        car.saveModel(model_filename);
      }
    } else {
      car.loadModel(model_filename);
      car.setFLTime(sim_time, dt);
    }

    car.setGoal(pt_goal);
    //car.setGoal(glm::vec3(400.0f, 200.0f, 0.0f));
    car.runRL();
    std::cout << "Current reward: " << car.scoreRL() << std::endl;
    car.printActions();
    car.printStates();

    // Renderer

    cur_state_index = 0;
    allow_mouse_click = true;
    while(!glfwWindowShouldClose(window))
    {
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        processInput(window);

        if (flag_time_to_run) {
          glm::mat4 goal_pt_model(1.0f);
          glUniformMatrix4fv(glGetUniformLocation(car_shader.id(), "model"), 1,
                             false, glm::value_ptr(goal_pt_model));
          glUniform3f(glGetUniformLocation(car_shader.id(), "in_color"), 1.0f,
                      1.0f, 1.0f);
          glBindVertexArray(goal_pt_vao);
          glDrawArrays(GL_TRIANGLES, 0, 3);
          glBindVertexArray(0);

          if (flag_draw_obstacle) ob.draw();

          // Or use framerate to calcuate dt
          //car.update(0.01f);
          car.draw(cur_state_index);
          car.processInput(window);

          if (cur_state_index < car.curStatesSize() - 1) ++cur_state_index;
          else if (flag_next_goal) {
            std::cout << "Next goal received!" << std::endl;
            flag_next_goal = false;

            //car.reset();
            //car.color(glm::vec3(0.5f, 0.0f, 0.0f));
            car.setGoal(next_pt_goal);
            car.runRL();
            std::cout << "Current reward: " << car.scoreRL() << std::endl;
            car.printActions();
            car.printStates();
            cur_state_index = 0;

            // Refill pt goal array
            arr_goal_point[0] = next_pt_goal.x;
            arr_goal_point[1] = next_pt_goal.y;
            arr_goal_point[2] = next_pt_goal.x-5.0f;
            arr_goal_point[3] = next_pt_goal.y-5.0f;
            arr_goal_point[4] = next_pt_goal.x+5.0f;
            arr_goal_point[5] = next_pt_goal.y-5.0f;

            glBindBuffer(GL_ARRAY_BUFFER, goal_pt_vbo);
            void *buf_ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
            memcpy(buf_ptr, arr_goal_point, 6 * sizeof(float));
            glUnmapBuffer(GL_ARRAY_BUFFER);
          }
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }


    delete[] arr_goal_point;
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
