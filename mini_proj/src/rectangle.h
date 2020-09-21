#ifndef __RECTANGLE_H__
#define __RECTANGLE_H__

#include "shader.h"
#include "shape.h"

#include "gtc/matrix_transform.hpp"
#include "gtc/type_ptr.hpp"
#include "gtx/string_cast.hpp"
#include "gtx/rotate_vector.hpp"

#include <iostream>

class Rectangle : public Shape
{
public:
    Rectangle () {};
    Rectangle (Shader &shader) : _shader(shader) {};
    Rectangle (Shader &shader, glm::vec3 tl, glm::vec3 br) : 
        _shader(shader), _tl(tl), _br(br) 
    {
        this->_bl = glm::vec3(tl.x, br.y, 0.0f);
        this->_tr = glm::vec3(br.x, tl.y, 0.0f);
        this->init();        
    };
    Rectangle (Shader &shader, glm::vec3 tl, glm::vec3 tr, glm::vec3 bl, glm::vec3 br) : 
        _shader(shader), _tl(tl), _tr(tr), _br(br), _bl(bl)
    {
        this->init();
    };
    ~Rectangle () {};

    // Implement pure virtual functions
    void init();
    void draw();
    void move(const glm::vec3 &to);

    // Getters
    const glm::vec3& tl() const { return this->_tl; };
    const glm::vec3& tr() const { return this->_tr; };
    const glm::vec3& bl() const { return this->_bl; };
    const glm::vec3& br() const { return this->_br; };
    // Setters
    void tl(const glm::vec3 new_tl) { _tl = new_tl; };
    void tr(const glm::vec3 new_tr) { _tr = new_tr; };
    void bl(const glm::vec3 new_bl) { _bl = new_bl; };
    void br(const glm::vec3 new_br) { _br = new_br; };
    void rotate(const double angle) { _rotate += angle; };
    void rotateTo(const double angle) { _rotate = angle; };
    void scaleTo(const double new_scale) { _scale = new_scale; };
    void color(const glm::vec3 new_color) { _color = new_color; };
private:
    // Shader program
    Shader _shader;

    // Four vertices of a rectangle
    glm::vec3 _tl, _tr, _br, _bl;
    // Color of the rectangle
    glm::vec3 _color = glm::vec3(0.0f);
    // Angle
    float _rotate = 0.0f;
    // Scale
    float _scale = 1.0f;
    // Central point
    glm::vec3 _center;
};

#endif