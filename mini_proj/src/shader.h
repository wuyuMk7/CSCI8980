#ifndef __SHADER_H__
#define __SHADER_H__

#include "glad/glad.h"
#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

class Shader 
{
public:
    const Shader use() { 
        glUseProgram(this->_id);
        return *this;
    };

    void loadByFile(const std::string &vs_src, const std::string &frag_src);
    void loadByCode(const std::string &vs_src, const std::string &frag_src);

    // Getter
    const GLuint id() { return _id; };
private:
    GLuint _id;

    void _loadShader(const std::string &vs_src, const std::string &frag_src);
    void _checkCompileStatus(const GLuint shader);
};

#endif