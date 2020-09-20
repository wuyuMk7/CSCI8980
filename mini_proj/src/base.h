#ifndef __BASE_H__
#define __BASE_H__

#include "glm.hpp"
#include <vector>
#include <unordered_map>

class Drawable
{
public:
    virtual void draw() = 0;    
};

class Movable
{
public:
    virtual void move(const glm::vec3 &) = 0;
};

#endif