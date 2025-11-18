#version 330 core

// Skybox Vertex Shader for HDRI Environment
// Mountain Studio Pro

layout (location = 0) in vec3 aPosition;

uniform mat4 uProjection;
uniform mat4 uView;

out vec3 vTexCoords;

void main() {
    vTexCoords = aPosition;

    // Remove translation from view matrix
    mat4 rotView = mat4(mat3(uView));

    vec4 pos = uProjection * rotView * vec4(aPosition, 1.0);

    // Set z = w for maximum depth
    gl_Position = pos.xyww;
}
