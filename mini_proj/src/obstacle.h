#ifndef __OBSTACLE_H__
#define __OBSTACLE_H__

#include "shader.h"
#include "shape.h"

#include "gtc/matrix_transform.hpp"
#include "gtc/type_ptr.hpp"
#include "gtx/string_cast.hpp"
#include "gtx/rotate_vector.hpp"

#include <vector>
#include <iostream>
#include <cmath>

class Obstacle : public Shape
{
public:
  Obstacle () {};
  Obstacle (Shader &shader) : _shader(shader) {};
  Obstacle (Shader &shader, glm::vec3 center, double radius) : 
    _shader(shader), _center(center), _radius(radius) {};

  // Implement pure virtual functions
  void init();
  void draw();
  void move(const glm::vec3 &to);

  // Getters
  const glm::vec3& center() { return this->_center; };
  const double radius() { return this->_radius; };
    // Setters
  void center(const glm::vec3 center) { _center = center; };
  void radius(const double radius) { _radius = radius; };
  void scaleTo(const double new_scale) { _scale = new_scale; };
  void color(const glm::vec3 new_color) { _color = new_color; };
private:
  // Shader program
  Shader _shader;

  glm::vec3 _center;
  double _radius;
  // Color of the rectangle
  glm::vec3 _color = glm::vec3(0.0f);
  // Scale
  float _scale = 1.0f;
};

#endif
