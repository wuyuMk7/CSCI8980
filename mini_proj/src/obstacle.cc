#include "obstacle.h"

void Obstacle::init()
{
  size_t total_len = 360 * 3 * 2;
  float *base_verts = new float[total_len];
  for (size_t i = 0;i < 360; ++i) {
    base_verts[i*6] = _center.x;
    base_verts[i*6+1] = _center.y;
    base_verts[i * 6 + 2] = _center.x + _radius * glm::cos(glm::radians((float)i));
    base_verts[i * 6 + 3] = _center.y + _radius * glm::sin(glm::radians((float)i));
    base_verts[i * 6 + 4] = _center.x + _radius * glm::cos(glm::radians((float)i+1.0f));
    base_verts[i * 6 + 5] = _center.y + _radius * glm::sin(glm::radians((float)i+1.0f));
    //std::cout << (float)i << " " << (float)i+1.0f << " " << glm::sin(glm::radians((float)i+1.0f)) << std::endl;
    //std::cout << base_verts[i * 6] << " " << base_verts[i * 6 + 1] << std::endl;
    //std::cout << base_verts[i * 6+2] << " " << base_verts[i * 6 + 3] << std::endl;
    //std::cout << base_verts[i * 6+4] << " " << base_verts[i * 6 + 5] << std::endl;
  }

  glGenVertexArrays(1, &this->_vao);
  glGenBuffers(1, &this->_vbo);

  glBindBuffer(GL_ARRAY_BUFFER, this->_vbo);
  glBufferData(GL_ARRAY_BUFFER, total_len * sizeof(float), base_verts, GL_STATIC_DRAW);

  glBindVertexArray(this->_vao);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  delete[] base_verts;
}

void Obstacle::draw()
{
    this->_shader.use();

    // Calculate model matrix
    glm::mat4 model = glm::mat4(1.0f);

    //model = glm::translate(model, cur_center - _initial_center);

    glUniformMatrix4fv(
        glGetUniformLocation(this->_shader.id(), "model"),
        1, 
        false,
        glm::value_ptr(model)   
    );

    glUniform3f(
        glGetUniformLocation(this->_shader.id(), "in_color"),
        this->_color.x, this->_color.y, this->_color.z
    );

    glBindVertexArray(this->_vao);
    glDrawArrays(GL_TRIANGLES, 0, 360 * 3);
    glBindVertexArray(0);
}

void Obstacle::move(const glm::vec3 &to)
{
  _center += to;
}
