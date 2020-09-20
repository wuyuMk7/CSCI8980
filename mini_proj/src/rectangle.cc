#include "rectangle.h"

void Rectangle::init()
{
    glm::vec3 ori(0.0f), trans = this->_tl - ori;
    glm::vec3 b_tl = ori, b_tr = this->_tr - trans,
              b_bl = this->_bl - trans, b_br = this->_br - trans;

    float base_verts[] = {
        b_tl.x, b_tl.y,
        b_tr.x, b_tr.y,
        b_br.x, b_br.y,
        
        b_tl.x, b_tl.y,
        b_bl.x, b_bl.y,
        b_br.x, b_br.y 
    };

    // std::cout << glm::to_string(b_tl) << std::endl
    //           << glm::to_string(b_tr) << std::endl
    //           << glm::to_string(b_br) << std::endl
    //           << glm::to_string(b_bl) << std::endl;

    glGenVertexArrays(1, &this->_vao);
    glGenBuffers(1, &this->_vbo);

    glBindBuffer(GL_ARRAY_BUFFER, this->_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(base_verts), base_verts, GL_STATIC_DRAW);

    glBindVertexArray(this->_vao);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Rectangle::draw()
{
    this->_shader.use();

    // Calculate model matrix
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, this->_tl);

    glm::vec3 center = (this->_br - this->_tl) * 0.5f;
    model = glm::translate(model, center);
    model = glm::rotate(model, glm::radians(this->_rotate), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::translate(model, -center);

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
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void Rectangle::move(const glm::vec3 &to)
{
    _tl += to;
    _tr += to;
    _br += to;
    _bl += to;
}