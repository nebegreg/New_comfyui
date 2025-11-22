"""
Animation System - Camera Paths & Time-lapse
=============================================

Create and export animated sequences:
- Keyframe animation system
- Camera path with spline interpolation
- Time-lapse rendering
- Video export (MP4)
- Turntable animations

Author: Mountain Studio Pro Team
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


@dataclass
class Keyframe:
    """Animation keyframe"""
    time: float  # Time in seconds
    camera_position: np.ndarray
    camera_target: np.ndarray
    camera_up: np.ndarray
    hdri_time: str
    weather: Optional[str] = None


class Timeline:
    """Animation timeline with keyframes"""
    
    def __init__(self, duration: float = 10.0):
        self.duration = duration
        self.keyframes: List[Keyframe] = []
        self.current_time = 0.0
        
    def add_keyframe(self, keyframe: Keyframe):
        """Add keyframe to timeline"""
        self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda k: k.time)
        
    def interpolate(self, time: float) -> Dict:
        """Interpolate parameters at given time"""
        if not self.keyframes:
            return {}
            
        # Find surrounding keyframes
        before = None
        after = None
        
        for kf in self.keyframes:
            if kf.time <= time:
                before = kf
            if kf.time >= time and after is None:
                after = kf
                
        if before is None:
            return self._keyframe_to_dict(self.keyframes[0])
        if after is None:
            return self._keyframe_to_dict(self.keyframes[-1])
        if before == after:
            return self._keyframe_to_dict(before)
            
        # Linear interpolation
        t = (time - before.time) / (after.time - before.time)
        
        return {
            'camera_position': before.camera_position * (1-t) + after.camera_position * t,
            'camera_target': before.camera_target * (1-t) + after.camera_target * t,
            'camera_up': before.camera_up * (1-t) + after.camera_up * t,
            'hdri_time': before.hdri_time if t < 0.5 else after.hdri_time,
            'weather': before.weather if t < 0.5 else after.weather
        }
        
    def _keyframe_to_dict(self, kf: Keyframe) -> Dict:
        """Convert keyframe to dictionary"""
        return {
            'camera_position': kf.camera_position,
            'camera_target': kf.camera_target,
            'camera_up': kf.camera_up,
            'hdri_time': kf.hdri_time,
            'weather': kf.weather
        }


class CameraPath:
    """Smooth camera path using spline interpolation"""
    
    def __init__(self):
        self.control_points: List[np.ndarray] = []
        self.times: List[float] = []
        self.spline = None
        
    def add_point(self, position: np.ndarray, time: float):
        """Add control point"""
        self.control_points.append(position)
        self.times.append(time)
        self._update_spline()
        
    def _update_spline(self):
        """Rebuild spline from control points"""
        if len(self.control_points) >= 2:
            points = np.array(self.control_points)
            self.spline = CubicSpline(self.times, points)
            
    def evaluate(self, time: float) -> np.ndarray:
        """Get position at time"""
        if self.spline is None:
            return self.control_points[0] if self.control_points else np.zeros(3)
        return self.spline(time)


class AnimationExporter:
    """Export animations to video"""
    
    def __init__(self, fps: int = 30):
        self.fps = fps
        
    def export_video(self, frames: List[np.ndarray], output_path: str,
                     codec: str = 'mp4v'):
        """Export frames to video file"""
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not available - cannot export video")
            return False
            
        if not frames:
            logger.error("No frames to export")
            return False
            
        h, w = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
        
        for frame in frames:
            # Convert float [0-1] to uint8 [0-255]
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                
            # OpenCV uses BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
        out.release()
        logger.info(f"Video exported: {output_path}")
        return True


class AnimationSystem:
    """Complete animation system"""
    
    def __init__(self):
        self.timeline = Timeline()
        self.camera_path = CameraPath()
        self.exporter = AnimationExporter()
        
    def create_turntable(self, center: np.ndarray, radius: float,
                          duration: float = 10.0, rotations: int = 1):
        """Create turntable animation"""
        num_keyframes = 36
        
        for i in range(num_keyframes):
            angle = (i / num_keyframes) * 2 * np.pi * rotations
            time = (i / num_keyframes) * duration
            
            x = center[0] + radius * np.cos(angle)
            z = center[2] + radius * np.sin(angle)
            y = center[1] + radius * 0.3  # Slight elevation
            
            kf = Keyframe(
                time=time,
                camera_position=np.array([x, y, z]),
                camera_target=center,
                camera_up=np.array([0, 1, 0]),
                hdri_time='midday'
            )
            
            self.timeline.add_keyframe(kf)
            
        logger.info(f"Created turntable animation: {num_keyframes} keyframes")
        
    def save(self, filepath: str):
        """Save animation to file"""
        data = {
            'duration': self.timeline.duration,
            'keyframes': [
                {
                    'time': kf.time,
                    'camera_position': kf.camera_position.tolist(),
                    'camera_target': kf.camera_target.tolist(),
                    'camera_up': kf.camera_up.tolist(),
                    'hdri_time': kf.hdri_time,
                    'weather': kf.weather
                }
                for kf in self.timeline.keyframes
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Animation saved: {filepath}")
        
    def __repr__(self):
        return f"AnimationSystem({len(self.timeline.keyframes)} keyframes, {self.timeline.duration}s)"
