#ifndef __CAR_H__
#define __CAR_H__

#include "rectangle.h"

#include "GLFW/glfw3.h"
#include "glm.hpp"

#include <algorithm>

class Car : public Drawable, public Movable
{
public:
    Car(Shader &shader, Rectangle &rec) : 
        _shader(shader), _rec(rec), _vel(0.0f) { this->init(); };
    Car(Shader &shader, Rectangle &rec, float vel): 
        _shader(shader), _rec(rec), _vel(vel) { this->init(); };

    void init();
    void update();
    void draw();
    void move(const glm::vec3 &to);

    void color(const glm::vec3 &new_color);
    
    void accelerate(float scale = 1.0f);
    void decelerate(float scale = 1.0f);
    void moveForward();
    void turnLeft();
    void turnRight();

    void processInput(GLFWwindow *window);
private:
    Shader _shader;
    Rectangle& _rec, _rec_sub;
    float _vel = 1.0f;
    glm::vec3 _color = glm::vec3(0.0f);
};

#endif