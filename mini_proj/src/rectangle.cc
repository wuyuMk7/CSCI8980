#include "rectangle.h"

void Rectangle::init()
{
    glm::vec3 ori(0.0f), trans = this->_bl - ori;
    glm::vec3 b_tl = this->_tl - trans, b_tr = this->_tr - trans,
              b_bl = ori, b_br = this->_br - trans;

    _initial_center = (b_bl + b_tr) * 0.5f;

    float base_verts[] = {
        b_tl.x, b_tl.y,
        b_tr.x, b_tr.y,
        b_br.x, b_br.y,
        
        b_tl.x, b_tl.y,
        b_bl.x, b_bl.y,
        b_br.x, b_br.y 
    };
    // float base_verts[] = {
    //     0.0f, 400.0f,
    //     200.0f, 0.0f,
    //     400.0f, 400.0f,

    //     0.0f, 400.0f,
    //     200.0f, 0.0f,
    //     400.0f, 400.0f
    // };

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

    glm::vec3 cur_center = (this->_bl + this->_tr) * 0.5f;
    model = glm::translate(model, cur_center - _initial_center);

    model = glm::translate(model, _initial_center);
    model = glm::rotate(model, glm::radians(this->_rotate), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::translate(model, -_initial_center);
    // this->_bl = glm::vec3(model * glm::vec4(this->_bl.x, this->_bl.y, this->_bl.z, 1.0f));
    // this->_tl = glm::vec3(model * glm::vec4(this->_tl.x, this->_tl.y, this->_tl.z, 1.0f));
    // this->_br = glm::vec3(model * glm::vec4(this->_br.x, this->_br.y, this->_br.z, 1.0f));
    // this->_tr = glm::vec3(model * glm::vec4(this->_tr.x, this->_tr.y, this->_tr.z, 1.0f));
    //model = glm::translate(model, this->_bl);

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