#version 330 core

// Shadow Depth Fragment Shader
// Outputs depth to shadow map texture
// Mountain Studio Pro

void main() {
    // gl_FragDepth is automatically written
    // We don't need to write anything explicitly for depth-only pass
    // But we can write it explicitly if needed:
    // gl_FragDepth = gl_FragCoord.z;
}
