#ifndef __SHAPE_H__
#define __SHAPE_H__

#include "glad/glad.h"
#include "glm.hpp"
#include "base.h"

#include <vector>
#include <unordered_map>

class Shape : public Drawable
{
public:
    Shape() {};
    // Shape() : _verts(std::vector<glm::vec3>{}) {};
    // Copy constructor
    // Reload assignment 

    // Function to initialize the shape
    virtual void init() = 0;

    // const std::vector<glm::vec3>& verts() const { return _verts; };
    const GLuint vao() const { return _vao; };
    const GLuint vbo() const { return _vbo; };

    // virtual void draw() const;
    // virtual void move(const std::unordered_map<unsigned int, glm::vec3> &);
    // virtual void moveTo(const std::unordered_map<unsigned int, glm::vec3> &);
    
protected:
    GLuint _vao;
    GLuint _vbo;
};

#endif