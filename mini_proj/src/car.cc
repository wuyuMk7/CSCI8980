#include "car.h"

void Car::init()
{
    // TODO: resize the smaller rectangle to overlap the bigger one
    glm::vec3 tl = _rec.tl(), tr = _rec.tr(), bl = _rec.bl(), br = _rec.br();
    glm::vec3 r_diff = tr - br, b_diff = br - bl, b_diff_n = glm::normalize(b_diff);
    float sub_b_len = glm::length(b_diff) * 0.25f;
    glm::vec3 sub_tl = 0.75f * r_diff + br, 
              sub_bl = 0.25f * r_diff + br,
              sub_tr = sub_tl + sub_b_len * b_diff_n, 
              sub_br = sub_bl + sub_b_len * b_diff_n;
    this->_rec_sub = Rectangle(this->_shader, sub_tl, sub_tr, sub_bl, sub_br);
}

void Car::update(float dt)
{
    this->moveForward(dt);
}

void Car::draw()
{
    this->_rec.draw();
    this->_rec_sub.draw();
}

void Car::move(const glm::vec3& to)
{
    this->_rec.move(to);
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
    glm::vec3 dir_n = glm::normalize(this->_rec.tr() - this->_rec.tl());
    this->move(_vel * dt * dir_n);
}

void Car::turnLeft()
{

}

void Car::turnRight()
{

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