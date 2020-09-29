#ifndef __CAR_H__
#define __CAR_H__

#include "rectangle.h"
#include "obstacle.h"
#include "runrl.h"
#include "rl.h"

#include "GLFW/glfw3.h"
#include "glm.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"

#include <cmath>
#include <unordered_map>
#include <string>
#include <algorithm>

const float car_vel_max = 200.0f;
const float car_omega_max = 3.14f;

class Car : public Drawable, public Movable, public RLRunnable
{
public:
    Car(Shader &shader, Rectangle &rec) : 
      _shader(shader), _rec(rec), _vel(0.0f), _rl(this) { this->init(); };
    Car(Shader &shader, Rectangle &rec, float vel): 
      _shader(shader), _rec(rec), _vel(vel), _rl(this) { this->init(); };

    void init();
    void update(float dt);
    void draw();
    void draw(size_t cur_state_index);
    void recalcSubRecPos();
    void move(const glm::vec3 &to);
    void runRL();
    double scoreRL();

    void color(const glm::vec3 &new_color);

    void accelerate(float scale = 1.0f);
    void decelerate(float scale = 1.0f);
    void moveForward(float dt);
    void turnLeft(float scale = 1.0f);
    void turnRight(float scale = 1.0f);

    void processInput(GLFWwindow *window);

    void setGoal(const glm::vec3 &goal) { this->_goal = goal; };
    void setObstacle(Obstacle &ob) { this->_ob = ob; this->_has_ob = true; };
    void trainRL(float sim_time, float dt);
  //    std::unordered_map<std::string, std::vector<std::vector<double>>> runRL(float sim_time, float dt);
    void printStates();
    void printActions();
    size_t curStatesSize() { return _rl_state_vec.size(); };
    size_t curActionsSize() { return _rl_action_vec.size(); };
    void forceToMoveTo(const glm::vec3 &to) { this->move(to - this->_rec.tl()); };
    void forceToRotateTo(const float theta) { this->_theta = theta; };
    void saveModel(const std::string &outfile) { this->_rl.save(outfile); };
    void loadModel(const std::string &infile) { this->_rl.load(infile); };
    void setFLTime(float sim_time, float dt)
    {
      this->_rl_sim_time = sim_time;
      this->_rl_dt = dt;
    }
private:
    Shader _shader;
    Rectangle& _rec, _rec_sub;
    float _vel = 1.0f;
    float _theta = 0.0f;
    float _omega = 0.0f;
    glm::vec3 _color = glm::vec3(0.0f);
    glm::vec3 _main_sub_center_vec = glm::vec3(0.0f);

    glm::vec3 _goal = glm::vec3(0.0f);

    RL _rl;
    Obstacle _ob;
    bool _has_ob = false;
    float _rl_sim_time, _rl_dt;
    std::vector<std::vector<double>> _rl_state_vec, _rl_action_vec;
};

#endif
