#ifndef __CAR_H__
#define __CAR_H__

#include "rectangle.h"

#include "glm.hpp"

class Car : public Drawable, public Movable
{
public:
    Car(Shader &shader, Rectangle &rec) : 
        _shader(shader), _rec(rec), _vel(0.0f) { this->init(); };
    Car(Shader &shader, Rectangle &rec, float vel): 
        _shader(shader), _rec(rec), _vel(vel) { this->init(); };

    void init();
    void draw();
    void move(const glm::vec3 &to);

    void color(const glm::vec3 &new_color);
private:
    Shader _shader;
    Rectangle& _rec, _rec_sub;
    float _vel = 0.0f;
    glm::vec3 _color = glm::vec3(0.0f);
};

#endif