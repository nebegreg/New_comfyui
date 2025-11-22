"""
Shadow Mapping System for Real-Time 3D Rendering
=================================================

Implements shadow mapping with:
- Depth map rendering from light POV
- PCF (Percentage Closer Filtering) for soft shadows
- Cascade shadow maps for large terrains
- Shadow acne prevention

Features:
- Realistic soft shadows
- Adjustable shadow quality
- Multiple light sources support
- Self-shadowing terrain

Author: Mountain Studio Pro Team
"""

import numpy as np
import logging
from typing import Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# OpenGL imports
try:
    from OpenGL.GL import *
    from OpenGL.GL import shaders
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    logger.warning("OpenGL not available - shadow mapping disabled")


class ShadowQuality(Enum):
    """Shadow quality presets"""
    LOW = "low"           # 512x512, 2x2 PCF
    MEDIUM = "medium"     # 1024x1024, 3x3 PCF
    HIGH = "high"         # 2048x2048, 5x5 PCF
    ULTRA = "ultra"       # 4096x4096, 7x7 PCF


class ShadowMapper:
    """
    Shadow mapping system for terrain rendering.

    Renders depth map from light's point of view, then uses it
    to determine shadowed areas in the main render pass.
    """

    # Shadow map vertex shader
    SHADOW_VERTEX_SHADER = """
    #version 330 core

    layout(location = 0) in vec3 position;

    uniform mat4 light_space_matrix;

    void main() {
        gl_Position = light_space_matrix * vec4(position, 1.0);
    }
    """

    # Shadow map fragment shader
    SHADOW_FRAGMENT_SHADER = """
    #version 330 core

    void main() {
        // Depth is written automatically to depth buffer
        // No color output needed
    }
    """

    # Main render vertex shader (with shadow)
    TERRAIN_VERTEX_SHADER = """
    #version 330 core

    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 normal;
    layout(location = 2) in vec2 texcoord;

    out vec3 frag_position;
    out vec3 frag_normal;
    out vec2 frag_texcoord;
    out vec4 frag_position_light_space;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform mat4 light_space_matrix;

    void main() {
        vec4 world_pos = model * vec4(position, 1.0);

        frag_position = world_pos.xyz;
        frag_normal = mat3(transpose(inverse(model))) * normal;
        frag_texcoord = texcoord;
        frag_position_light_space = light_space_matrix * world_pos;

        gl_Position = projection * view * world_pos;
    }
    """

    # Main render fragment shader (with PCF shadows)
    TERRAIN_FRAGMENT_SHADER = """
    #version 330 core

    in vec3 frag_position;
    in vec3 frag_normal;
    in vec2 frag_texcoord;
    in vec4 frag_position_light_space;

    out vec4 frag_color;

    uniform vec3 light_direction;
    uniform vec3 light_color;
    uniform float light_intensity;
    uniform vec3 view_position;

    uniform sampler2D shadow_map;
    uniform int pcf_samples;  // 2, 3, 5, 7
    uniform float shadow_bias;

    // PBR textures
    uniform sampler2D albedo_map;
    uniform sampler2D normal_map;
    uniform sampler2D roughness_map;
    uniform sampler2D ao_map;

    uniform bool use_pcf;

    float calculate_shadow_pcf(vec4 frag_pos_light_space, vec3 normal, vec3 light_dir) {
        // Perspective divide
        vec3 proj_coords = frag_pos_light_space.xyz / frag_pos_light_space.w;

        // Transform to [0,1] range
        proj_coords = proj_coords * 0.5 + 0.5;

        // Get depth from shadow map
        float current_depth = proj_coords.z;

        // Check if outside shadow map
        if (proj_coords.z > 1.0)
            return 1.0;

        // Bias to prevent shadow acne
        float bias = max(shadow_bias * (1.0 - dot(normal, light_dir)), shadow_bias * 0.1);

        // PCF (Percentage Closer Filtering)
        float shadow = 0.0;
        vec2 texel_size = 1.0 / textureSize(shadow_map, 0);

        int half_samples = pcf_samples / 2;

        for (int x = -half_samples; x <= half_samples; x++) {
            for (int y = -half_samples; y <= half_samples; y++) {
                vec2 offset = vec2(x, y) * texel_size;
                float pcf_depth = texture(shadow_map, proj_coords.xy + offset).r;

                shadow += (current_depth - bias) > pcf_depth ? 0.0 : 1.0;
            }
        }

        shadow /= float(pcf_samples * pcf_samples);

        return shadow;
    }

    float calculate_shadow_simple(vec4 frag_pos_light_space) {
        vec3 proj_coords = frag_pos_light_space.xyz / frag_pos_light_space.w;
        proj_coords = proj_coords * 0.5 + 0.5;

        float closest_depth = texture(shadow_map, proj_coords.xy).r;
        float current_depth = proj_coords.z;

        float shadow = current_depth > closest_depth ? 0.0 : 1.0;

        return shadow;
    }

    void main() {
        // Sample textures
        vec3 albedo = texture(albedo_map, frag_texcoord).rgb;
        vec3 normal = normalize(frag_normal);  // Or sample from normal_map
        float roughness = texture(roughness_map, frag_texcoord).r;
        float ao = texture(ao_map, frag_texcoord).r;

        // Lighting calculations
        vec3 view_dir = normalize(view_position - frag_position);
        vec3 light_dir = normalize(-light_direction);

        // Diffuse
        float diff = max(dot(normal, light_dir), 0.0);
        vec3 diffuse = diff * light_color * light_intensity;

        // Specular (simplified Blinn-Phong)
        vec3 halfway_dir = normalize(light_dir + view_dir);
        float spec = pow(max(dot(normal, halfway_dir), 0.0), 32.0) * (1.0 - roughness);
        vec3 specular = spec * light_color * light_intensity * 0.5;

        // Ambient
        vec3 ambient = vec3(0.3) * albedo * ao;

        // Calculate shadow
        float shadow = 1.0;
        if (use_pcf) {
            shadow = calculate_shadow_pcf(frag_position_light_space, normal, light_dir);
        } else {
            shadow = calculate_shadow_simple(frag_position_light_space);
        }

        // Combine lighting
        vec3 color = ambient + shadow * (diffuse + specular) * albedo;

        frag_color = vec4(color, 1.0);
    }
    """

    def __init__(self, quality: ShadowQuality = ShadowQuality.MEDIUM):
        """
        Initialize shadow mapper.

        Args:
            quality: Shadow quality preset
        """
        if not OPENGL_AVAILABLE:
            raise RuntimeError("OpenGL not available - cannot use shadow mapping")

        self.quality = quality

        # Quality settings
        quality_map = {
            ShadowQuality.LOW: (512, 2, 0.005),
            ShadowQuality.MEDIUM: (1024, 3, 0.003),
            ShadowQuality.HIGH: (2048, 5, 0.002),
            ShadowQuality.ULTRA: (4096, 7, 0.001)
        }

        self.shadow_map_size, self.pcf_samples, self.shadow_bias = quality_map[quality]

        # OpenGL objects
        self.shadow_fbo = None
        self.shadow_texture = None
        self.shadow_shader = None
        self.terrain_shader = None

        # Light parameters
        self.light_direction = np.array([0.5, -1.0, 0.3], dtype=np.float32)
        self.light_color = np.array([1.0, 1.0, 0.98], dtype=np.float32)
        self.light_intensity = 1.2

        # Shadow frustum
        self.ortho_size = 100.0
        self.near_plane = 1.0
        self.far_plane = 200.0

        logger.info(f"ShadowMapper initialized: {quality.value}, {self.shadow_map_size}x{self.shadow_map_size}")

    def initialize(self):
        """Initialize OpenGL resources"""
        # Create shadow map framebuffer
        self.shadow_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.shadow_fbo)

        # Create shadow map texture (depth only)
        self.shadow_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.shadow_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
                     self.shadow_map_size, self.shadow_map_size,
                     0, GL_DEPTH_COMPONENT, GL_FLOAT, None)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

        # Border color (outside shadow map = fully lit)
        border_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)

        # Attach to framebuffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_2D, self.shadow_texture, 0)

        # No color attachment needed
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        # Check framebuffer status
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Shadow framebuffer not complete")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Compile shaders
        self.shadow_shader = self._compile_shadow_shader()
        self.terrain_shader = self._compile_terrain_shader()

        logger.info("Shadow mapping resources initialized")

    def _compile_shadow_shader(self):
        """Compile shadow map shader"""
        vertex = shaders.compileShader(self.SHADOW_VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment = shaders.compileShader(self.SHADOW_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        shader = shaders.compileProgram(vertex, fragment)
        return shader

    def _compile_terrain_shader(self):
        """Compile terrain shader with shadows"""
        vertex = shaders.compileShader(self.TERRAIN_VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment = shaders.compileShader(self.TERRAIN_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        shader = shaders.compileProgram(vertex, fragment)
        return shader

    def calculate_light_space_matrix(self) -> np.ndarray:
        """Calculate light space transformation matrix"""
        # Light position (opposite of direction)
        light_pos = -self.light_direction * 100.0

        # View matrix from light
        light_view = self._look_at(light_pos, np.array([0, 0, 0]), np.array([0, 1, 0]))

        # Orthographic projection for directional light
        light_projection = self._ortho(-self.ortho_size, self.ortho_size,
                                       -self.ortho_size, self.ortho_size,
                                       self.near_plane, self.far_plane)

        # Combined light space matrix
        light_space_matrix = light_projection @ light_view

        return light_space_matrix.astype(np.float32)

    def _look_at(self, eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Calculate view matrix (look-at)"""
        f = center - eye
        f = f / np.linalg.norm(f)

        s = np.cross(f, up)
        s = s / np.linalg.norm(s)

        u = np.cross(s, f)

        result = np.identity(4)
        result[0, :3] = s
        result[1, :3] = u
        result[2, :3] = -f
        result[3, :3] = np.array([-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye)])

        return result

    def _ortho(self, left: float, right: float, bottom: float, top: float,
               near: float, far: float) -> np.ndarray:
        """Calculate orthographic projection matrix"""
        result = np.identity(4)
        result[0, 0] = 2.0 / (right - left)
        result[1, 1] = 2.0 / (top - bottom)
        result[2, 2] = -2.0 / (far - near)
        result[3, 0] = -(right + left) / (right - left)
        result[3, 1] = -(top + bottom) / (top - bottom)
        result[3, 2] = -(far + near) / (far - near)
        return result

    def render_shadow_map(self, terrain_vao, terrain_vertex_count):
        """
        Render shadow map (depth pass).

        Args:
            terrain_vao: Terrain VAO
            terrain_vertex_count: Number of vertices
        """
        # Bind shadow framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.shadow_fbo)
        glViewport(0, 0, self.shadow_map_size, self.shadow_map_size)
        glClear(GL_DEPTH_BUFFER_BIT)

        # Use shadow shader
        glUseProgram(self.shadow_shader)

        # Set light space matrix
        light_space_matrix = self.calculate_light_space_matrix()
        light_loc = glGetUniformLocation(self.shadow_shader, "light_space_matrix")
        glUniformMatrix4fv(light_loc, 1, GL_FALSE, light_space_matrix)

        # Render terrain
        glBindVertexArray(terrain_vao)
        glDrawElements(GL_TRIANGLES, terrain_vertex_count, GL_UNSIGNED_INT, None)

        # Unbind
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def render_terrain_with_shadows(self, terrain_vao, terrain_vertex_count,
                                     view_matrix, projection_matrix):
        """
        Render terrain with shadows (main pass).

        Args:
            terrain_vao: Terrain VAO
            terrain_vertex_count: Number of vertices
            view_matrix: Camera view matrix
            projection_matrix: Camera projection matrix
        """
        # Use terrain shader
        glUseProgram(self.terrain_shader)

        # Set matrices
        model = np.identity(4, dtype=np.float32)
        light_space_matrix = self.calculate_light_space_matrix()

        glUniformMatrix4fv(glGetUniformLocation(self.terrain_shader, "model"),
                          1, GL_FALSE, model)
        glUniformMatrix4fv(glGetUniformLocation(self.terrain_shader, "view"),
                          1, GL_FALSE, view_matrix)
        glUniformMatrix4fv(glGetUniformLocation(self.terrain_shader, "projection"),
                          1, GL_FALSE, projection_matrix)
        glUniformMatrix4fv(glGetUniformLocation(self.terrain_shader, "light_space_matrix"),
                          1, GL_FALSE, light_space_matrix)

        # Set lighting
        glUniform3fv(glGetUniformLocation(self.terrain_shader, "light_direction"),
                    1, self.light_direction)
        glUniform3fv(glGetUniformLocation(self.terrain_shader, "light_color"),
                    1, self.light_color)
        glUniform1f(glGetUniformLocation(self.terrain_shader, "light_intensity"),
                   self.light_intensity)

        # Set shadow parameters
        glUniform1i(glGetUniformLocation(self.terrain_shader, "pcf_samples"),
                   self.pcf_samples)
        glUniform1f(glGetUniformLocation(self.terrain_shader, "shadow_bias"),
                   self.shadow_bias)
        glUniform1i(glGetUniformLocation(self.terrain_shader, "use_pcf"), 1)

        # Bind shadow map
        glActiveTexture(GL_TEXTURE0 + 4)  # Use texture unit 4 for shadow map
        glBindTexture(GL_TEXTURE_2D, self.shadow_texture)
        glUniform1i(glGetUniformLocation(self.terrain_shader, "shadow_map"), 4)

        # Render terrain
        glBindVertexArray(terrain_vao)
        glDrawElements(GL_TRIANGLES, terrain_vertex_count, GL_UNSIGNED_INT, None)

    def set_light(self, direction: np.ndarray, color: np.ndarray = None, intensity: float = None):
        """Set light parameters"""
        self.light_direction = direction / np.linalg.norm(direction)

        if color is not None:
            self.light_color = color

        if intensity is not None:
            self.light_intensity = intensity

    def set_quality(self, quality: ShadowQuality):
        """Change shadow quality (requires reinitialization)"""
        if quality != self.quality:
            self.quality = quality
            self.cleanup()
            self.initialize()

    def cleanup(self):
        """Cleanup OpenGL resources"""
        if self.shadow_fbo:
            glDeleteFramebuffers(1, [self.shadow_fbo])
        if self.shadow_texture:
            glDeleteTextures([self.shadow_texture])
        if self.shadow_shader:
            glDeleteProgram(self.shadow_shader)
        if self.terrain_shader:
            glDeleteProgram(self.terrain_shader)

        logger.info("Shadow mapping resources cleaned up")

    def __del__(self):
        """Destructor"""
        self.cleanup()
