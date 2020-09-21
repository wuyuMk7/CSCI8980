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
    this->_vel = std::min(this->_vel + scale * 2.0f, 60.0f);
    // std::cout << "a: " << _vel << std::endl;
}

void Car::decelerate(float scale)
{
    this->_vel = std::max(this->_vel - scale * 2.0f, -30.0f);
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
    this->_omega = std::min(this->_omega + scale, 30.0f);
}

void Car::turnRight(float scale)
{
    this->_omega = std::max(this->_omega - scale, -30.0f);
}

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