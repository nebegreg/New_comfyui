#version 330 core

// Shadow Depth Vertex Shader
// Renders scene from light's perspective to generate shadow map
// Mountain Studio Pro

layout (location = 0) in vec3 aPosition;

uniform mat4 uLightSpaceMatrix;
uniform mat4 uModel;

void main() {
    gl_Position = uLightSpaceMatrix * uModel * vec4(aPosition, 1.0);
}
