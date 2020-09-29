#include "car.h"

void Car::init()
{
    // TODO: resize the smaller rectangle to overlap the bigger one
    // this->recalcSubRecPos();
    glm::vec3 tl = _rec.tl(), tr = _rec.tr(), bl = _rec.bl(), br = _rec.br();
    glm::vec3 r_diff = tr - br, b_diff = br - bl, b_diff_n = glm::normalize(b_diff);
    float sub_b_len = glm::length(b_diff) * 0.25f;
    glm::vec3 sub_tl = 0.75f * r_diff + br, 
              sub_bl = 0.25f * r_diff + br,
              sub_tr = sub_tl + sub_b_len * b_diff_n, 
              sub_br = sub_bl + sub_b_len * b_diff_n;
    this->_rec_sub = Rectangle(this->_shader, sub_tl, sub_tr, sub_bl, sub_br);
    glm::vec3 rec_sub_cen = this->_rec_sub.bl() + this->_rec_sub.initialCenter(),
              rec_cen = this->_rec.bl() + this->_rec.initialCenter();
    this->_main_sub_center_vec = rec_sub_cen - rec_cen;
}

void Car::update(float dt)
{
    float angle_diff = this->_omega * dt;
    this->_theta += angle_diff;
    this->_main_sub_center_vec = glm::rotateZ(this->_main_sub_center_vec, glm::radians(angle_diff));
    this->moveForward(dt);
}

void Car::draw()
{
    this->_rec.draw();
    this->_rec_sub.draw();
}

void Car::recalcSubRecPos() 
{
    glm::vec3 tl = _rec.tl(), tr = _rec.tr(), bl = _rec.bl(), br = _rec.br();
    glm::vec3 r_diff = tr - br, b_diff = br - bl, b_diff_n = glm::normalize(b_diff);
    float sub_b_len = glm::length(b_diff) * 0.25f;
    glm::vec3 sub_tl = 0.75f * r_diff + br, 
              sub_bl = 0.25f * r_diff + br,
              sub_tr = sub_tl + sub_b_len * b_diff_n, 
              sub_br = sub_bl + sub_b_len * b_diff_n;
    this->_rec_sub.tl(sub_tl);
    this->_rec_sub.bl(sub_bl);
    this->_rec_sub.tr(sub_tr);
    this->_rec_sub.br(sub_br);
}

void Car::move(const glm::vec3& to)
{
    glm::vec3 cur_main_cen = (this->_rec.bl() + this->_rec.tr()) * 0.5f,
              cur_sub_cen = (this->_rec_sub.bl() + this->_rec_sub.tr()) * 0.5f,
              cur_main_sub_cen_vec = cur_sub_cen - cur_main_cen;

    this->_rec.rotateTo(this->_theta);
    this->_rec.move(to);

    //this->recalcSubRecPos();
    // std::cout << glm::to_string(this->_main_sub_center_vec) << std::endl;
    // std::cout << glm::to_string(cur_main_sub_cen_vec) << std::endl; 
    // std::cout << glm::to_string(this->_main_sub_center_vec - cur_main_sub_cen_vec) << std::endl << std::endl;
    this->_rec_sub.move(this->_main_sub_center_vec - cur_main_sub_cen_vec);
    this->_rec_sub.rotateTo(this->_theta);
    this->_rec_sub.move(to);
}

void Car::color(const glm::vec3& new_color)
{
    this->_color = new_color;
    this->_rec.color(new_color);
    this->_rec_sub.color(new_color);
}

void Car::accelerate(float scale)
{
    this->_vel = std::min(this->_vel + scale * 2.0f, car_vel_max);
    // std::cout << "a: " << _vel << std::endl;
}

void Car::decelerate(float scale)
{
    this->_vel = std::max(this->_vel - scale * 2.0f, -car_vel_max / 2);
    // std::cout << "d: " << _vel << std::endl;
}

void Car::moveForward(float dt)
{
    // update the rotation of the car here?
    // this->_theta += dt * this->_omega;
    glm::vec3 dir_n = glm::normalize(this->_rec.tr() - this->_rec.tl());
    dir_n = glm::rotate(dir_n, glm::radians(this->_theta), glm::vec3(0.0f, 0.0f, 1.0f));
    //std::cout << this->_theta << std::endl;
    this->move(_vel * dt * dir_n);
}

void Car::turnLeft(float scale)
{
    this->_omega = std::min(this->_omega + scale, car_omega_max);
}

void Car::turnRight(float scale)
{
    this->_omega = std::max(this->_omega - scale, -car_omega_max);
}

void Car::trainRL(float sim_time, float dt)
{
  this->_rl_sim_time = sim_time;
  this->_rl_dt = dt;

  this->_rl.train();
}

void Car::runRL()
{
  double dou_times;
  modf(_rl_sim_time / _rl_dt, &dou_times);

  double c_x = _rec.tl().x, c_y = _rec.tl().y, c_theta = _theta;
  std::vector<std::vector<double>> state_list, action_list;
  state_list.emplace_back(std::vector<double> {c_x, c_y, c_theta, _goal.x, _goal.y});

  int times = (int)dou_times;
  for (size_t i = 0;i < times; ++i) {
    xt::xarray<double> cur_state{ c_x, c_y, c_theta,  _goal.x, _goal.y }, next_state;
    xt::xarray<double> action = this->_rl.run(cur_state);

    // Clamp
    double new_vel = action[0], new_omega = action[1], new_vel_x, new_vel_y;
    if (new_vel < 0 && new_vel < -car_vel_max/2) new_vel = -car_vel_max/2;
    if (new_vel > 0 && new_vel > car_vel_max) new_vel = car_vel_max;
    if (new_omega < 0 && new_omega < -car_omega_max) new_omega = -car_omega_max;
    if (new_omega > 0 && new_omega > car_omega_max) new_omega = car_omega_max;

    new_vel_x = new_vel * cos(c_theta);
    new_vel_y = new_vel * sin(c_theta);

    c_x += new_vel_x * _rl_dt;
    c_y += new_vel_y * _rl_dt;
    c_theta += new_omega * _rl_dt;

    state_list.emplace_back(std::vector<double>{ c_x, c_y, c_theta, _goal.x, _goal.y });
    action_list.emplace_back(std::vector<double>{ action[0], action[1] });
  }

  _rl_state_vec = state_list;
  _rl_action_vec = action_list;
}

double Car::scoreRL()
{
  double task_reward = 0;

  std::vector<double> cur_action, cur_state;
  double dist;
  for (int i = 0;i < _rl_action_vec.size(); ++i) {
    cur_action = _rl_action_vec[i];
    cur_state = _rl_state_vec[i+1];

    //double dx = cur_state[3] - cur_state[0], dy = cur_state[4] - cur_state[1];
    double dx = this->_goal.x - cur_state[0], dy = this->_goal.y - cur_state[1];
    dist = sqrt(dx * dx + dy * dy);

    task_reward -= dist;
    task_reward -= 1 * abs(cur_action[0]);
    task_reward -= 1 * abs(cur_action[1]);

    // Check borders
    if (cur_state[0] < 0 || cur_state[1] < 0 || cur_state[0] > 800 || cur_state[0] > 600)
      task_reward -= 1000;
  }

  if (_rl_action_vec.size() > 0) {
    if (dist < 100) task_reward += 3000;
    if (dist < 50 && abs(cur_action[0]) < 3.0) task_reward += 10000;
    //std::cout << cur_state[0] << ", " << cur_state[1] << "," << dist << std::endl;
  }

  return task_reward;
}

/*
std::unordered_map<std::string, std::vector<std::vector<double>>>
Car::runRL(float sim_time, float dt)
{
  double dou_times;
  modf(sim_time / dt, &dou_times);

  double c_x = _rec.tl().x, c_y = _rec.tl().y, c_theta = _theta;
  std::vector<std::vector<double>> state_list, action_list;
  int times = (int)dou_times;
  for (size_t i = 0;i < times; ++i) {
    xt::xarray<double> cur_state{ c_x, c_y, c_theta,  _goal.x, _goal.y }, next_state;
    xt::xarray<double> action = this->_rl.run(cur_state);

    // Clamp
    double new_vel = action[0], new_omega = action[1], new_vel_x, new_vel_y;
    if (new_vel < 0 && new_vel < -car_vel_max/2) new_vel = -car_vel_max/2;
    if (new_vel > 0 && new_vel > car_vel_max) new_vel = car_vel_max;
    if (new_omega < 0 && new_omega < -car_omega_max) new_omega = -car_omega_max;
    if (new_omega > 0 && new_omega > car_omega_max) new_omega = car_omega_max;

    new_vel_x = new_vel * cos(new_omega);
    new_vel_y = new_vel * sin(new_omega);

    c_x += new_vel_x * dt;
    c_y += new_vel_y * dt;
    c_theta += new_omega * dt;

    state_list.emplace_back(std::vector<double>{ c_x, c_y, c_theta, _goal.x, _goal.y });
    action_list.emplace_back(std::vector<double>{ action[0], action[1] });
  }

  std::unordered_map<std::string, std::vector<std::vector<double>>> ret_data;
  ret_data["state"] = state_list;
  ret_data["action"] = action_list;

  return ret_data;
}
*/

void Car::processInput(GLFWwindow *window)
{
    // WSAD Control
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        this->accelerate();
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        this->decelerate();
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        this->turnLeft();
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        this->turnRight();
}
