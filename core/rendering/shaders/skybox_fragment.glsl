#version 330 core

// Skybox Fragment Shader for HDRI Environment
// Samples cubemap texture for panoramic environment
// Mountain Studio Pro

in vec3 vTexCoords;

uniform samplerCube uSkybox;
uniform float uExposure;

out vec4 FragColor;

// Simple tone mapping
vec3 ToneMapReinhard(vec3 hdrColor) {
    vec3 mapped = hdrColor / (hdrColor + vec3(1.0));
    return mapped;
}

// ACES tone mapping (more filmic)
vec3 ToneMapACES(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    // Sample cubemap
    vec3 envColor = texture(uSkybox, vTexCoords).rgb;

    // Apply exposure
    envColor = envColor * uExposure;

    // Tone mapping
    envColor = ToneMapACES(envColor);

    // Gamma correction
    envColor = pow(envColor, vec3(1.0/2.2));

    FragColor = vec4(envColor, 1.0);
}
