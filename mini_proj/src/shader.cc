#include "shader.h"

void Shader::loadByFile(
    const std::string &vs_src, 
    const std::string &frag_src
) {
    std::ifstream vs_fstream(vs_src), frag_fstream(frag_src);
    std::stringstream vs_sstream, frag_sstream;

    vs_sstream << vs_fstream.rdbuf();
    frag_sstream << frag_fstream.rdbuf();

    this->_loadShader(vs_sstream.str(), frag_sstream.str());
}

void Shader::loadByCode(
    const std::string &vs_src,
    const std::string &frag_src
) {
    this->_loadShader(vs_src, frag_src);
}

void Shader::_loadShader(
    const std::string &vs_src,
    const std::string &frag_src
) {
    GLuint vs_shader, frag_shader;
    const char *vs_shader_src = vs_src.c_str(), *frag_shader_src = frag_src.c_str();

    vs_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs_shader, 1, &vs_shader_src, NULL);
    glCompileShader(vs_shader);
    this->_checkCompileStatus(vs_shader);

    frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag_shader, 1, &frag_shader_src, NULL);
    glCompileShader(frag_shader);
    this->_checkCompileStatus(frag_shader);

    GLuint final_shader = glCreateProgram();
    glAttachShader(final_shader, vs_shader);
    glAttachShader(final_shader, frag_shader);
    glLinkProgram(final_shader);
    this->_id = final_shader;

    glDeleteShader(vs_shader);
    glDeleteShader(frag_shader);
}

void Shader::_checkCompileStatus(const GLuint shader)
{
    GLint com_status;

    glGetShaderiv(shader, GL_COMPILE_STATUS, &com_status);
    if (!com_status) {
        char buf[512];
        glGetShaderInfoLog(shader, 512, NULL, buf);
        printf("Vertex shader compilation failed. Error msg: %s \n", buf);

        exit(0);
    }
}