"""
FPS Camera System for Mountain Studio Pro

Implements a complete First-Person camera with:
- WASD movement
- Mouse look (yaw/pitch)
- Terrain collision
- Smooth interpolation
- Speed control

Author: Mountain Studio Pro
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FPSCamera:
    """
    First-Person camera with full movement and collision detection.

    Controls:
        WASD - Horizontal movement
        Space - Move up
        Shift - Move down
        Mouse - Look around
        Scroll - Change speed

    Attributes:
        position (np.ndarray): Camera position [x, y, z]
        yaw (float): Horizontal rotation in degrees
        pitch (float): Vertical rotation in degrees (clamped to -89/+89)
        speed (float): Movement speed in units/second
        sensitivity (float): Mouse sensitivity
        collision_enabled (bool): Enable terrain collision
        min_height_above_terrain (float): Minimum height above terrain
    """

    # Class constants
    DEFAULT_SPEED = 10.0
    DEFAULT_SENSITIVITY = 0.1
    DEFAULT_FOV = 60.0
    MIN_PITCH = -89.0
    MAX_PITCH = 89.0
    MIN_HEIGHT_OFFSET = 2.0  # meters above terrain

    def __init__(
        self,
        position: Optional[np.ndarray] = None,
        yaw: float = -90.0,  # Looking along -Z initially
        pitch: float = 0.0,
        speed: float = DEFAULT_SPEED,
        sensitivity: float = DEFAULT_SENSITIVITY
    ):
        """
        Initialize FPS camera.

        Args:
            position: Initial position [x, y, z]. Defaults to [0, 10, 0]
            yaw: Initial yaw angle in degrees. -90 = looking along -Z
            pitch: Initial pitch angle in degrees. 0 = level
            speed: Movement speed in units/second
            sensitivity: Mouse sensitivity multiplier
        """
        self.position = position if position is not None else np.array([0.0, 10.0, 0.0], dtype=np.float32)
        self.yaw = yaw
        self.pitch = np.clip(pitch, self.MIN_PITCH, self.MAX_PITCH)
        self.speed = speed
        self.sensitivity = sensitivity
        self.fov = self.DEFAULT_FOV

        # Collision
        self.collision_enabled = True
        self.min_height_above_terrain = self.MIN_HEIGHT_OFFSET
        self._heightmap = None
        self._terrain_scale = 1.0
        self._terrain_height_scale = 1.0

        # Movement state
        self._move_forward = False
        self._move_backward = False
        self._move_left = False
        self._move_right = False
        self._move_up = False
        self._move_down = False

        # Cached vectors
        self._front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self._update_camera_vectors()

        logger.info(f"FPS Camera initialized at {self.position}")

    def set_heightmap(
        self,
        heightmap: np.ndarray,
        terrain_scale: float = 1.0,
        height_scale: float = 1.0
    ):
        """
        Set heightmap for collision detection.

        Args:
            heightmap: 2D array of terrain heights [0-1]
            terrain_scale: Scale of terrain in world units
            height_scale: Multiplier for height values
        """
        self._heightmap = heightmap
        self._terrain_scale = terrain_scale
        self._terrain_height_scale = height_scale
        logger.info(f"Heightmap set: {heightmap.shape}, scale={terrain_scale}, height_scale={height_scale}")

    def _update_camera_vectors(self):
        """Update front, right, and up vectors based on yaw and pitch."""
        # Calculate new front vector
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)

        front_x = np.cos(yaw_rad) * np.cos(pitch_rad)
        front_y = np.sin(pitch_rad)
        front_z = np.sin(yaw_rad) * np.cos(pitch_rad)

        self._front = np.array([front_x, front_y, front_z], dtype=np.float32)
        self._front = self._front / np.linalg.norm(self._front)

        # Calculate right and up vectors
        self._right = np.cross(self._front, self._world_up)
        self._right = self._right / np.linalg.norm(self._right)

        self._up = np.cross(self._right, self._front)
        self._up = self._up / np.linalg.norm(self._up)

    def process_mouse_movement(self, x_offset: float, y_offset: float, constrain_pitch: bool = True):
        """
        Process mouse movement for camera rotation.

        Args:
            x_offset: Mouse movement in X (pixels)
            y_offset: Mouse movement in Y (pixels)
            constrain_pitch: Clamp pitch to prevent camera flip
        """
        x_offset *= self.sensitivity
        y_offset *= self.sensitivity

        self.yaw += x_offset
        self.pitch += y_offset

        if constrain_pitch:
            self.pitch = np.clip(self.pitch, self.MIN_PITCH, self.MAX_PITCH)

        # Normalize yaw to -180/+180
        if self.yaw > 180.0:
            self.yaw -= 360.0
        elif self.yaw < -180.0:
            self.yaw += 360.0

        self._update_camera_vectors()

    def process_keyboard(self, delta_time: float):
        """
        Update position based on current keyboard state.

        Args:
            delta_time: Time since last frame in seconds
        """
        velocity = self.speed * delta_time

        # Calculate movement on horizontal plane (ignore Y component of front)
        front_horizontal = np.array([self._front[0], 0.0, self._front[2]], dtype=np.float32)
        if np.linalg.norm(front_horizontal) > 0:
            front_horizontal = front_horizontal / np.linalg.norm(front_horizontal)

        right_horizontal = np.array([self._right[0], 0.0, self._right[2]], dtype=np.float32)
        if np.linalg.norm(right_horizontal) > 0:
            right_horizontal = right_horizontal / np.linalg.norm(right_horizontal)

        # Apply movement
        if self._move_forward:
            self.position += front_horizontal * velocity
        if self._move_backward:
            self.position -= front_horizontal * velocity
        if self._move_right:
            self.position += right_horizontal * velocity
        if self._move_left:
            self.position -= right_horizontal * velocity
        if self._move_up:
            self.position += self._world_up * velocity
        if self._move_down:
            self.position -= self._world_up * velocity

        # Apply collision
        if self.collision_enabled and self._heightmap is not None:
            self._apply_terrain_collision()

    def _apply_terrain_collision(self):
        """Keep camera above terrain using bilinear interpolation."""
        terrain_height = self._get_terrain_height_at(self.position[0], self.position[2])
        min_y = terrain_height + self.min_height_above_terrain

        if self.position[1] < min_y:
            # Smooth interpolation instead of hard clamp
            self.position[1] = self._lerp(self.position[1], min_y, 0.2)

    def _get_terrain_height_at(self, world_x: float, world_z: float) -> float:
        """
        Get interpolated terrain height at world position.

        Args:
            world_x: World X coordinate
            world_z: World Z coordinate

        Returns:
            Interpolated height at position
        """
        if self._heightmap is None:
            return 0.0

        # Convert world coords to heightmap coords
        # Heightmap is centered at origin, scaled by terrain_scale
        height, width = self._heightmap.shape

        # Map world coordinates to heightmap indices
        # Assuming terrain is centered: world [-scale/2, scale/2] maps to heightmap [0, size]
        half_scale = self._terrain_scale / 2.0

        x_norm = (world_x + half_scale) / self._terrain_scale  # [0, 1]
        z_norm = (world_z + half_scale) / self._terrain_scale  # [0, 1]

        x_grid = x_norm * (width - 1)
        z_grid = z_norm * (height - 1)

        # Clamp to valid range
        x_grid = np.clip(x_grid, 0, width - 1)
        z_grid = np.clip(z_grid, 0, height - 1)

        # Bilinear interpolation
        x0 = int(np.floor(x_grid))
        x1 = min(x0 + 1, width - 1)
        z0 = int(np.floor(z_grid))
        z1 = min(z0 + 1, height - 1)

        fx = x_grid - x0
        fz = z_grid - z0

        # Get four corner heights
        h00 = self._heightmap[z0, x0]
        h10 = self._heightmap[z0, x1]
        h01 = self._heightmap[z1, x0]
        h11 = self._heightmap[z1, x1]

        # Bilinear interpolation
        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx
        height_normalized = h0 * (1 - fz) + h1 * fz

        # Scale to world height
        return height_normalized * self._terrain_height_scale

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation."""
        return a + (b - a) * t

    def get_view_matrix(self) -> np.ndarray:
        """
        Calculate view matrix for rendering.

        Returns:
            4x4 view matrix
        """
        return self._look_at(self.position, self.position + self._front, self._up)

    @staticmethod
    def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
        """
        Calculate look-at view matrix.

        Args:
            eye: Camera position
            center: Target position
            up: Up vector

        Returns:
            4x4 view matrix
        """
        f = center - eye
        f = f / np.linalg.norm(f)

        s = np.cross(f, up)
        s = s / np.linalg.norm(s)

        u = np.cross(s, f)

        result = np.eye(4, dtype=np.float32)
        result[0, 0:3] = s
        result[1, 0:3] = u
        result[2, 0:3] = -f
        result[0, 3] = -np.dot(s, eye)
        result[1, 3] = -np.dot(u, eye)
        result[2, 3] = np.dot(f, eye)

        return result

    def get_projection_matrix(self, aspect_ratio: float, near: float = 0.1, far: float = 1000.0) -> np.ndarray:
        """
        Calculate perspective projection matrix.

        Args:
            aspect_ratio: Viewport width / height
            near: Near clipping plane
            far: Far clipping plane

        Returns:
            4x4 projection matrix
        """
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)

        result = np.zeros((4, 4), dtype=np.float32)
        result[0, 0] = f / aspect_ratio
        result[1, 1] = f
        result[2, 2] = (far + near) / (near - far)
        result[2, 3] = (2.0 * far * near) / (near - far)
        result[3, 2] = -1.0

        return result

    # Movement state setters
    def set_move_forward(self, value: bool):
        self._move_forward = value

    def set_move_backward(self, value: bool):
        self._move_backward = value

    def set_move_left(self, value: bool):
        self._move_left = value

    def set_move_right(self, value: bool):
        self._move_right = value

    def set_move_up(self, value: bool):
        self._move_up = value

    def set_move_down(self, value: bool):
        self._move_down = value

    def reset(self):
        """Reset camera to initial position and rotation."""
        self.position = np.array([0.0, 10.0, 0.0], dtype=np.float32)
        self.yaw = -90.0
        self.pitch = 0.0
        self._update_camera_vectors()
        logger.info("Camera reset")

    def get_state(self) -> dict:
        """
        Get current camera state for serialization.

        Returns:
            Dictionary with camera parameters
        """
        return {
            'position': self.position.tolist(),
            'yaw': self.yaw,
            'pitch': self.pitch,
            'speed': self.speed,
            'sensitivity': self.sensitivity,
            'fov': self.fov,
            'collision_enabled': self.collision_enabled
        }

    def set_state(self, state: dict):
        """
        Restore camera from saved state.

        Args:
            state: Dictionary from get_state()
        """
        self.position = np.array(state['position'], dtype=np.float32)
        self.yaw = state['yaw']
        self.pitch = state['pitch']
        self.speed = state.get('speed', self.DEFAULT_SPEED)
        self.sensitivity = state.get('sensitivity', self.DEFAULT_SENSITIVITY)
        self.fov = state.get('fov', self.DEFAULT_FOV)
        self.collision_enabled = state.get('collision_enabled', True)
        self._update_camera_vectors()
        logger.info(f"Camera state restored: {self.position}")

    def __repr__(self) -> str:
        return (f"FPSCamera(pos={self.position}, yaw={self.yaw:.1f}°, "
                f"pitch={self.pitch:.1f}°, speed={self.speed})")
