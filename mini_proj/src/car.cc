#include "car.h"

void Car::init()
{
    glm::vec3 tl = _rec.tl(), tr = _rec.tr(), bl = _rec.bl(), br = _rec.br();
    glm::vec3 r_diff = tr - br, b_diff = br - bl, b_diff_n = glm::normalize(b_diff);
    float sub_b_len = glm::length(b_diff) * 0.125f;
    glm::vec3 sub_tl = 0.75f * r_diff + br, sub_bl = 0.25f * r_diff + br,
              sub_tr = sub_tl + sub_b_len * b_diff_n, 
              sub_br = sub_bl + sub_b_len * b_diff_n;
    this->_rec_sub = Rectangle(this->_shader, sub_tl, sub_tr, sub_bl, sub_br);
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