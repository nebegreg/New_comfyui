#version 330 core

// Terrain Vertex Shader with Shadow Mapping
// Mountain Studio Pro

layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;

// Uniforms
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat4 uLightSpaceMatrix;

// Outputs to fragment shader
out vec3 vFragPos;
out vec3 vNormal;
out vec3 vColor;
out vec4 vFragPosLightSpace;
out float vElevation;

void main() {
    // Transform position to world space
    vec4 worldPos = uModel * vec4(aPosition, 1.0);
    vFragPos = worldPos.xyz;

    // Transform normal to world space
    vNormal = mat3(transpose(inverse(uModel))) * aNormal;

    // Pass through color
    vColor = aColor;

    // Calculate position in light space for shadow mapping
    vFragPosLightSpace = uLightSpaceMatrix * worldPos;

    // Pass elevation for effects
    vElevation = aPosition.y;

    // Transform to clip space
    gl_Position = uProjection * uView * worldPos;
}
