#version 330 core

// Terrain Fragment Shader with Shadow Mapping and Fog
// Mountain Studio Pro

// Inputs from vertex shader
in vec3 vFragPos;
in vec3 vNormal;
in vec3 vColor;
in vec4 vFragPosLightSpace;
in float vElevation;

// Uniforms
uniform vec3 uLightDir;           // Directional light direction (normalized)
uniform vec3 uLightColor;         // Light color
uniform vec3 uViewPos;            // Camera position
uniform sampler2D uShadowMap;     // Shadow depth texture
uniform float uAmbientStrength;   // Ambient light strength
uniform float uShadowBias;        // Shadow acne prevention bias
uniform bool uShadowsEnabled;     // Toggle shadows
uniform bool uFogEnabled;         // Toggle fog
uniform vec3 uFogColor;           // Fog color
uniform float uFogDensity;        // Fog density
uniform float uFogStart;          // Fog start distance
uniform float uFogEnd;            // Fog end distance

// Output
out vec4 FragColor;

// PCF (Percentage Closer Filtering) shadow calculation
float CalculateShadow(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir) {
    if (!uShadowsEnabled) {
        return 0.0;
    }

    // Perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;

    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;

    // Outside shadow map frustum = no shadow
    if (projCoords.z > 1.0) {
        return 0.0;
    }

    // Get depth from shadow map
    float currentDepth = projCoords.z;

    // Adaptive bias based on slope
    float bias = max(uShadowBias * (1.0 - dot(normal, -lightDir)), uShadowBias * 0.1);

    // PCF (3x3 kernel)
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(uShadowMap, 0);

    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(uShadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }

    shadow /= 9.0;  // Average over 9 samples

    return shadow;
}

// Calculate fog factor (linear fog)
float CalculateFog(float distance) {
    if (!uFogEnabled) {
        return 0.0;
    }

    // Linear fog
    float fogFactor = (uFogEnd - distance) / (uFogEnd - uFogStart);
    fogFactor = clamp(fogFactor, 0.0, 1.0);

    return 1.0 - fogFactor;
}

// Calculate fog factor (exponential squared - more realistic)
float CalculateFogExp(float distance) {
    if (!uFogEnabled) {
        return 0.0;
    }

    float fogFactor = exp(-pow(uFogDensity * distance, 2.0));
    fogFactor = clamp(fogFactor, 0.0, 1.0);

    return 1.0 - fogFactor;
}

void main() {
    // Normalize inputs
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(-uLightDir);

    // --- Ambient lighting ---
    vec3 ambient = uAmbientStrength * vColor * uLightColor;

    // --- Diffuse lighting (Lambertian) ---
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * vColor * uLightColor;

    // --- Specular lighting (Blinn-Phong) ---
    vec3 viewDir = normalize(uViewPos - vFragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    vec3 specular = spec * uLightColor * 0.3;  // Reduced for terrain

    // --- Shadow calculation ---
    float shadow = CalculateShadow(vFragPosLightSpace, normal, lightDir);

    // Apply shadow to diffuse and specular only
    vec3 lighting = ambient + (1.0 - shadow) * (diffuse + specular);

    // --- Fog ---
    float distance = length(uViewPos - vFragPos);
    float fogAmount = CalculateFogExp(distance);

    vec3 finalColor = mix(lighting, uFogColor, fogAmount);

    // Output
    FragColor = vec4(finalColor, 1.0);
}
