"""
Advanced Terrain Viewer with Real-Time Shadows and FPS Camera
Mountain Studio Pro

Features:
- Modern OpenGL (3.3+) with custom shaders
- Real-time shadow mapping with PCF
- FPS camera controls (WASD + mouse)
- Phong lighting model
- Atmospheric fog
- HDRI skybox support (optional)
- Performance optimized with LOD

Controls:
    WASD - Move camera
    Space - Move up
    Shift - Move down
    Mouse - Look around
    R - Reset camera
    C - Toggle collision
    1-3 - Quality presets
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
import time

from PySide6.QtWidgets import QWidget, QOpenGLWidget, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QSurfaceFormat

try:
    from OpenGL.GL import *
    from OpenGL.GL import shaders
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

from core.camera.fps_camera import FPSCamera

logger = logging.getLogger(__name__)


class AdvancedTerrainViewer(QOpenGLWidget):
    """
    Advanced OpenGL terrain viewer with shadow mapping and FPS camera.

    Requires OpenGL 3.3+ for modern shader support.
    """

    # Shadow map resolutions
    SHADOW_LOW = 1024
    SHADOW_MEDIUM = 2048
    SHADOW_HIGH = 4096

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize advanced terrain viewer."""
        if not OPENGL_AVAILABLE:
            raise ImportError("PyOpenGL is required for AdvancedTerrainViewer")

        super().__init__(parent)

        # Request OpenGL 3.3 Core Profile
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setSamples(4)  # 4x MSAA
        self.setFormat(fmt)

        # Terrain data
        self._heightmap = None
        self._terrain_scale = 1.0
        self._height_scale = 1.0
        self._mesh_lod = 1  # LOD: 1, 2, 4
        self._terrain_vao = None
        self._terrain_vbo = None
        self._terrain_ebo = None
        self._vertex_count = 0

        # Camera
        self._camera = FPSCamera()
        self._camera_mode = 'fps'  # 'fps' or 'orbit'

        # Rendering settings
        self._shadows_enabled = True
        self._shadow_quality = self.SHADOW_MEDIUM
        self._fog_enabled = True
        self._fog_density = 0.0001
        self._wireframe = False

        # Lighting
        self._light_dir = np.array([0.3, -0.7, 0.5], dtype=np.float32)  # Directional light
        self._light_dir = self._light_dir / np.linalg.norm(self._light_dir)
        self._light_color = np.array([1.0, 1.0, 0.95], dtype=np.float32)
        self._ambient_strength = 0.3
        self._shadow_bias = 0.005

        # Shadow mapping
        self._shadow_fbo = None
        self._shadow_texture = None

        # Shaders
        self._terrain_shader = None
        self._shadow_shader = None
        self._skybox_shader = None

        # Skybox (optional)
        self._skybox_enabled = False
        self._skybox_texture = None
        self._skybox_vao = None

        # Input state
        self._keys_pressed = set()
        self._mouse_grabbed = True
        self._last_mouse_pos = None
        self._last_frame_time = time.time()

        # FPS counter
        self._fps = 0
        self._frame_count = 0
        self._fps_timer = time.time()

        # Mouse capture
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        # Update timer (60 FPS target)
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._on_update)
        self._update_timer.start(16)  # ~60 FPS

        logger.info("AdvancedTerrainViewer initialized")

    # ========== OpenGL Initialization ==========

    def initializeGL(self):
        """Initialize OpenGL context and resources."""
        # Check OpenGL version
        gl_version = glGetString(GL_VERSION).decode('utf-8')
        logger.info(f"OpenGL Version: {gl_version}")

        # Enable features
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_MULTISAMPLE)  # MSAA

        # Clear color (sky blue)
        glClearColor(0.5, 0.7, 0.9, 1.0)

        # Load shaders
        self._load_shaders()

        # Create shadow FBO
        self._create_shadow_framebuffer()

        logger.info("OpenGL initialized successfully")

    def _load_shaders(self):
        """Load and compile all GLSL shaders."""
        shader_dir = Path(__file__).parent.parent.parent / "core" / "rendering" / "shaders"

        try:
            # Terrain shader
            terrain_vert = self._load_shader_file(shader_dir / "terrain_vertex.glsl")
            terrain_frag = self._load_shader_file(shader_dir / "terrain_fragment.glsl")
            self._terrain_shader = self._compile_shader_program(terrain_vert, terrain_frag)

            # Shadow shader
            shadow_vert = self._load_shader_file(shader_dir / "shadow_depth.vert")
            shadow_frag = self._load_shader_file(shader_dir / "shadow_depth.frag")
            self._shadow_shader = self._compile_shader_program(shadow_vert, shadow_frag)

            # Skybox shader (optional)
            if (shader_dir / "skybox_vertex.glsl").exists():
                skybox_vert = self._load_shader_file(shader_dir / "skybox_vertex.glsl")
                skybox_frag = self._load_shader_file(shader_dir / "skybox_fragment.glsl")
                self._skybox_shader = self._compile_shader_program(skybox_vert, skybox_frag)

            logger.info("Shaders loaded successfully")

        except Exception as e:
            logger.error(f"Shader loading failed: {e}")
            raise

    @staticmethod
    def _load_shader_file(path: Path) -> str:
        """Load shader source from file."""
        with open(path, 'r') as f:
            return f.read()

    @staticmethod
    def _compile_shader_program(vertex_src: str, fragment_src: str) -> int:
        """Compile vertex and fragment shaders into a program."""
        vertex_shader = shaders.compileShader(vertex_src, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER)

        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)

        # Check linking status
        if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
            info = glGetProgramInfoLog(program).decode('utf-8')
            raise RuntimeError(f"Shader program linking failed:\n{info}")

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        return program

    def _create_shadow_framebuffer(self):
        """Create framebuffer for shadow mapping."""
        # Create FBO
        self._shadow_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._shadow_fbo)

        # Create depth texture
        self._shadow_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._shadow_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
                     self._shadow_quality, self._shadow_quality, 0,
                     GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        border_color = [1.0, 1.0, 1.0, 1.0]
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)

        # Attach depth texture to FBO
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_2D, self._shadow_texture, 0)

        # No color attachment needed
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        # Check FBO status
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Shadow framebuffer is not complete")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        logger.info(f"Shadow framebuffer created: {self._shadow_quality}x{self._shadow_quality}")

    # ========== Terrain Setup ==========

    def set_terrain(
        self,
        heightmap: np.ndarray,
        terrain_scale: float = 100.0,
        height_scale: float = 20.0,
        lod: int = 1
    ):
        """
        Set terrain heightmap and generate mesh.

        Args:
            heightmap: 2D array of heights [0-1]
            terrain_scale: Size in world units
            height_scale: Height multiplier
            lod: Level of detail (1, 2, 4)
        """
        self._heightmap = heightmap
        self._terrain_scale = terrain_scale
        self._height_scale = height_scale
        self._mesh_lod = lod

        # Update camera heightmap for collision
        self._camera.set_heightmap(heightmap, terrain_scale, height_scale)

        # Generate mesh
        self._generate_terrain_mesh()

        # Center camera above terrain
        terrain_center_height = heightmap[heightmap.shape[0]//2, heightmap.shape[1]//2] * height_scale
        self._camera.position = np.array([0.0, terrain_center_height + 10.0, 0.0], dtype=np.float32)

        logger.info(f"Terrain set: {heightmap.shape}, scale={terrain_scale}, height={height_scale}, LOD={lod}")

        self.update()

    def _generate_terrain_mesh(self):
        """Generate terrain mesh from heightmap."""
        if self._heightmap is None:
            return

        makeCurrent_result = self.makeCurrent()
        if not makeCurrent_result:
            logger.warning("Could not make OpenGL context current for mesh generation")

        height, width = self._heightmap.shape

        # Apply LOD
        step = self._mesh_lod
        sub_height = height // step
        sub_width = width // step

        # Generate vertices
        vertices = []
        normals = []
        colors = []

        for z in range(0, height, step):
            for x in range(0, width, step):
                # Position
                px = (x / (width - 1) - 0.5) * self._terrain_scale
                py = self._heightmap[z, x] * self._height_scale
                pz = (z / (height - 1) - 0.5) * self._terrain_scale

                vertices.extend([px, py, pz])

                # Normal (calculate from gradient)
                if x > 0 and x < width - 1 and z > 0 and z < height - 1:
                    dhdx = (self._heightmap[z, x + 1] - self._heightmap[z, x - 1]) / 2.0
                    dhdz = (self._heightmap[z + 1, x] - self._heightmap[z - 1, x]) / 2.0

                    nx = -dhdx * self._height_scale / (self._terrain_scale / width)
                    ny = 1.0
                    nz = -dhdz * self._height_scale / (self._terrain_scale / height)

                    n_len = np.sqrt(nx**2 + ny**2 + nz**2)
                    nx, ny, nz = nx / n_len, ny / n_len, nz / n_len
                else:
                    nx, ny, nz = 0.0, 1.0, 0.0

                normals.extend([nx, ny, nz])

                # Color (elevation-based)
                h_norm = self._heightmap[z, x]
                color = self._get_elevation_color(h_norm)
                colors.extend(color)

        # Generate indices
        indices = []
        for z in range(sub_height - 1):
            for x in range(sub_width - 1):
                i0 = z * sub_width + x
                i1 = i0 + 1
                i2 = i0 + sub_width
                i3 = i2 + 1

                # Two triangles per quad
                indices.extend([i0, i2, i1])
                indices.extend([i1, i2, i3])

        # Convert to numpy
        vertices = np.array(vertices, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)

        self._vertex_count = len(indices)

        # Interleave vertex data: position(3) + normal(3) + color(3)
        vertex_data = np.zeros(len(vertices) // 3 * 9, dtype=np.float32)
        for i in range(len(vertices) // 3):
            vertex_data[i*9:i*9+3] = vertices[i*3:i*3+3]
            vertex_data[i*9+3:i*9+6] = normals[i*3:i*3+3]
            vertex_data[i*9+6:i*9+9] = colors[i*3:i*3+3]

        # Create VAO/VBO/EBO
        if self._terrain_vao is None:
            self._terrain_vao = glGenVertexArrays(1)
            self._terrain_vbo = glGenBuffers(1)
            self._terrain_ebo = glGenBuffers(1)

        glBindVertexArray(self._terrain_vao)

        # Upload vertex data
        glBindBuffer(GL_ARRAY_BUFFER, self._terrain_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        # Upload indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._terrain_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Vertex attributes
        stride = 9 * 4  # 9 floats * 4 bytes

        # Position (location 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

        # Normal (location 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

        # Color (location 2)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))

        glBindVertexArray(0)

        logger.info(f"Terrain mesh generated: {self._vertex_count} indices")

    @staticmethod
    def _get_elevation_color(height_normalized: float) -> Tuple[float, float, float]:
        """Get color based on elevation."""
        if height_normalized < 0.15:
            # Water/valleys (blue-green)
            return (0.15, 0.35, 0.55)
        elif height_normalized < 0.4:
            # Forests (dark green)
            return (0.2, 0.5, 0.2)
        elif height_normalized < 0.6:
            # Meadows (green-brown)
            return (0.4, 0.5, 0.3)
        elif height_normalized < 0.75:
            # Rock (gray-brown)
            return (0.5, 0.45, 0.4)
        else:
            # Snow (white-blue)
            return (0.9, 0.9, 0.95)

    # ========== Rendering ==========

    def paintGL(self):
        """Render the scene."""
        if self._terrain_vao is None:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            return

        # --- Shadow Pass ---
        if self._shadows_enabled:
            self._render_shadow_pass()

        # --- Main Render Pass ---
        self._render_main_pass()

        # FPS counter
        self._update_fps()

    def _render_shadow_pass(self):
        """Render scene from light's perspective to generate shadow map."""
        glBindFramebuffer(GL_FRAMEBUFFER, self._shadow_fbo)
        glViewport(0, 0, self._shadow_quality, self._shadow_quality)
        glClear(GL_DEPTH_BUFFER_BIT)

        glUseProgram(self._shadow_shader)

        # Light space matrix (orthographic projection from light's view)
        light_view = self._calculate_light_view_matrix()
        light_projection = self._calculate_light_projection_matrix()
        light_space_matrix = light_projection @ light_view

        model = np.eye(4, dtype=np.float32)

        # Set uniforms
        glUniformMatrix4fv(glGetUniformLocation(self._shadow_shader, "uLightSpaceMatrix"),
                          1, GL_TRUE, light_space_matrix)
        glUniformMatrix4fv(glGetUniformLocation(self._shadow_shader, "uModel"),
                          1, GL_TRUE, model)

        # Render terrain
        glBindVertexArray(self._terrain_vao)
        glDrawElements(GL_TRIANGLES, self._vertex_count, GL_UNSIGNED_INT, None)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _render_main_pass(self):
        """Render scene from camera's perspective with shadows and lighting."""
        glViewport(0, 0, self.width(), self.height())
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Wireframe mode
        if self._wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glUseProgram(self._terrain_shader)

        # Matrices
        model = np.eye(4, dtype=np.float32)
        view = self._camera.get_view_matrix()
        projection = self._camera.get_projection_matrix(self.width() / max(self.height(), 1))

        light_view = self._calculate_light_view_matrix()
        light_projection = self._calculate_light_projection_matrix()
        light_space_matrix = light_projection @ light_view

        # Set uniforms
        glUniformMatrix4fv(glGetUniformLocation(self._terrain_shader, "uModel"),
                          1, GL_TRUE, model)
        glUniformMatrix4fv(glGetUniformLocation(self._terrain_shader, "uView"),
                          1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(self._terrain_shader, "uProjection"),
                          1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(self._terrain_shader, "uLightSpaceMatrix"),
                          1, GL_TRUE, light_space_matrix)

        glUniform3fv(glGetUniformLocation(self._terrain_shader, "uLightDir"),
                     1, self._light_dir)
        glUniform3fv(glGetUniformLocation(self._terrain_shader, "uLightColor"),
                     1, self._light_color)
        glUniform3fv(glGetUniformLocation(self._terrain_shader, "uViewPos"),
                     1, self._camera.position)

        glUniform1f(glGetUniformLocation(self._terrain_shader, "uAmbientStrength"),
                    self._ambient_strength)
        glUniform1f(glGetUniformLocation(self._terrain_shader, "uShadowBias"),
                    self._shadow_bias)

        glUniform1i(glGetUniformLocation(self._terrain_shader, "uShadowsEnabled"),
                    self._shadows_enabled)
        glUniform1i(glGetUniformLocation(self._terrain_shader, "uFogEnabled"),
                    self._fog_enabled)

        glUniform3f(glGetUniformLocation(self._terrain_shader, "uFogColor"),
                    0.7, 0.8, 0.9)
        glUniform1f(glGetUniformLocation(self._terrain_shader, "uFogDensity"),
                    self._fog_density)
        glUniform1f(glGetUniformLocation(self._terrain_shader, "uFogStart"),
                    50.0)
        glUniform1f(glGetUniformLocation(self._terrain_shader, "uFogEnd"),
                    500.0)

        # Bind shadow map
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._shadow_texture)
        glUniform1i(glGetUniformLocation(self._terrain_shader, "uShadowMap"), 0)

        # Render terrain
        glBindVertexArray(self._terrain_vao)
        glDrawElements(GL_TRIANGLES, self._vertex_count, GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

    def _calculate_light_view_matrix(self) -> np.ndarray:
        """Calculate view matrix from light's perspective."""
        # Light position (far away in light direction)
        light_pos = -self._light_dir * self._terrain_scale * 2.0

        # Look at center
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        return FPSCamera._look_at(light_pos, target, up)

    def _calculate_light_projection_matrix(self) -> np.ndarray:
        """Calculate orthographic projection for directional light."""
        size = self._terrain_scale * 0.7
        near = 1.0
        far = self._terrain_scale * 3.0

        result = np.zeros((4, 4), dtype=np.float32)
        result[0, 0] = 1.0 / size
        result[1, 1] = 1.0 / size
        result[2, 2] = -2.0 / (far - near)
        result[2, 3] = -(far + near) / (far - near)
        result[3, 3] = 1.0

        return result

    def resizeGL(self, w: int, h: int):
        """Handle window resize."""
        glViewport(0, 0, w, h)

    # ========== Input Handling ==========

    def _on_update(self):
        """Update loop (60 FPS)."""
        current_time = time.time()
        delta_time = current_time - self._last_frame_time
        self._last_frame_time = current_time

        # Update camera based on input
        self._camera.process_keyboard(delta_time)

        # Trigger repaint
        self.update()

    def keyPressEvent(self, event):
        """Handle key press."""
        key = event.key()

        # Camera movement
        if key == Qt.Key_W:
            self._camera.set_move_forward(True)
        elif key == Qt.Key_S:
            self._camera.set_move_backward(True)
        elif key == Qt.Key_A:
            self._camera.set_move_left(True)
        elif key == Qt.Key_D:
            self._camera.set_move_right(True)
        elif key == Qt.Key_Space:
            self._camera.set_move_up(True)
        elif key == Qt.Key_Shift:
            self._camera.set_move_down(True)

        # Controls
        elif key == Qt.Key_R:
            self._camera.reset()
        elif key == Qt.Key_C:
            self._camera.collision_enabled = not self._camera.collision_enabled
            logger.info(f"Collision: {self._camera.collision_enabled}")

        event.accept()

    def keyReleaseEvent(self, event):
        """Handle key release."""
        key = event.key()

        if key == Qt.Key_W:
            self._camera.set_move_forward(False)
        elif key == Qt.Key_S:
            self._camera.set_move_backward(False)
        elif key == Qt.Key_A:
            self._camera.set_move_left(False)
        elif key == Qt.Key_D:
            self._camera.set_move_right(False)
        elif key == Qt.Key_Space:
            self._camera.set_move_up(False)
        elif key == Qt.Key_Shift:
            self._camera.set_move_down(False)

        event.accept()

    def mouseMoveEvent(self, event):
        """Handle mouse movement for camera rotation."""
        if not self._mouse_grabbed:
            return

        pos = event.pos()

        if self._last_mouse_pos is None:
            self._last_mouse_pos = pos
            return

        dx = pos.x() - self._last_mouse_pos.x()
        dy = self._last_mouse_pos.y() - pos.y()  # Inverted Y

        self._camera.process_mouse_movement(dx, dy)

        self._last_mouse_pos = pos
        event.accept()

    def mousePressEvent(self, event):
        """Capture mouse on click."""
        if event.button() == Qt.LeftButton:
            self._mouse_grabbed = True
            self.setCursor(Qt.BlankCursor)
            self._last_mouse_pos = event.pos()

        event.accept()

    def mouseReleaseEvent(self, event):
        """Release mouse."""
        if event.button() == Qt.LeftButton:
            self._mouse_grabbed = False
            self.setCursor(Qt.ArrowCursor)
            self._last_mouse_pos = None

        event.accept()

    # ========== Utilities ==========

    def _update_fps(self):
        """Update FPS counter."""
        self._frame_count += 1
        current_time = time.time()

        if current_time - self._fps_timer >= 1.0:
            self._fps = self._frame_count
            self._frame_count = 0
            self._fps_timer = current_time

    def get_fps(self) -> int:
        """Get current FPS."""
        return self._fps

    def set_shadows_enabled(self, enabled: bool):
        """Toggle shadows."""
        self._shadows_enabled = enabled
        logger.info(f"Shadows: {enabled}")
        self.update()

    def set_fog_enabled(self, enabled: bool):
        """Toggle fog."""
        self._fog_enabled = enabled
        logger.info(f"Fog: {enabled}")
        self.update()

    def set_wireframe(self, enabled: bool):
        """Toggle wireframe mode."""
        self._wireframe = enabled
        self.update()

    def set_shadow_quality(self, quality: int):
        """
        Set shadow map resolution.

        Args:
            quality: SHADOW_LOW, SHADOW_MEDIUM, or SHADOW_HIGH
        """
        if quality == self._shadow_quality:
            return

        self._shadow_quality = quality

        # Recreate shadow framebuffer
        if self._shadow_fbo is not None:
            self.makeCurrent()
            glDeleteFramebuffers(1, [self._shadow_fbo])
            glDeleteTextures(1, [self._shadow_texture])
            self._create_shadow_framebuffer()

        logger.info(f"Shadow quality: {quality}x{quality}")
        self.update()

    def cleanup(self):
        """Clean up OpenGL resources."""
        if not self.isValid():
            return

        self.makeCurrent()

        # Delete buffers
        if self._terrain_vao is not None:
            glDeleteVertexArrays(1, [self._terrain_vao])
            glDeleteBuffers(1, [self._terrain_vbo])
            glDeleteBuffers(1, [self._terrain_ebo])

        # Delete FBO
        if self._shadow_fbo is not None:
            glDeleteFramebuffers(1, [self._shadow_fbo])
            glDeleteTextures(1, [self._shadow_texture])

        # Delete shaders
        if self._terrain_shader is not None:
            glDeleteProgram(self._terrain_shader)
        if self._shadow_shader is not None:
            glDeleteProgram(self._shadow_shader)
        if self._skybox_shader is not None:
            glDeleteProgram(self._skybox_shader)

        logger.info("OpenGL resources cleaned up")

    def __del__(self):
        """Destructor."""
        self.cleanup()
