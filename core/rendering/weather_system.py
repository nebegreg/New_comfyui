"""
Dynamic Weather System
======================

Realistic weather simulation and rendering:
- Snow (particles, accumulation)
- Rain (drops, wetness)
- Fog (volumetric, distance-based)
- Clouds (dynamic 3D)
- Wind (affects particles and vegetation)
- Time-lapse (day/night cycle with weather)

Author: Mountain Studio Pro Team
"""

import numpy as np
import logging
from typing import Tuple, Optional
from enum import Enum
import time

logger = logging.getLogger(__name__)


class WeatherType(Enum):
    """Weather types"""
    CLEAR = "clear"
    LIGHT_SNOW = "light_snow"
    HEAVY_SNOW = "heavy_snow"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    FOG = "fog"
    BLIZZARD = "blizzard"
    STORM = "storm"


class WeatherSystem:
    """
    Dynamic weather simulation and rendering.
    """

    def __init__(self):
        self.current_weather = WeatherType.CLEAR
        self.transition_progress = 0.0
        self.target_weather = None
        
        # Particle systems
        self.snow_particles = []
        self.rain_particles = []
        
        # Weather parameters
        self.wind_direction = np.array([1.0, 0.0, 0.3])
        self.wind_strength = 0.5
        self.fog_density = 0.0
        self.cloud_coverage = 0.0
        
        # Time of day
        self.time_of_day = 12.0  # Hours (0-24)
        self.day_cycle_speed = 0.1  # Hours per second
        
        logger.info("WeatherSystem initialized")

    def set_weather(self, weather: WeatherType, transition_time: float = 5.0):
        """Change weather with smooth transition"""
        self.target_weather = weather
        self.transition_time = transition_time
        self.transition_progress = 0.0
        logger.info(f"Transitioning to {weather.value}")

    def update(self, dt: float):
        """Update weather simulation (call each frame)"""
        # Update time of day
        self.time_of_day += self.day_cycle_speed * dt
        if self.time_of_day >= 24.0:
            self.time_of_day -= 24.0
        
        # Update weather transition
        if self.target_weather:
            self.transition_progress += dt / self.transition_time
            if self.transition_progress >= 1.0:
                self.current_weather = self.target_weather
                self.target_weather = None
                
        # Update particles
        self._update_particles(dt)
        
        # Update atmospheric parameters
        self._update_atmosphere()

    def _update_particles(self, dt: float):
        """Update particle systems"""
        # Update snow particles
        for particle in self.snow_particles:
            particle['position'] += particle['velocity'] * dt
            particle['position'] += self.wind_direction * self.wind_strength * dt
            
        # Update rain particles
        for particle in self.rain_particles:
            particle['position'] += particle['velocity'] * dt

    def _update_atmosphere(self):
        """Update atmospheric parameters based on weather"""
        weather_params = {
            WeatherType.CLEAR: {'fog': 0.0, 'clouds': 0.1},
            WeatherType.LIGHT_SNOW: {'fog': 0.1, 'clouds': 0.6},
            WeatherType.HEAVY_SNOW: {'fog': 0.3, 'clouds': 0.9},
            WeatherType.LIGHT_RAIN: {'fog': 0.15, 'clouds': 0.7},
            WeatherType.HEAVY_RAIN: {'fog': 0.25, 'clouds': 0.95},
            WeatherType.FOG: {'fog': 0.8, 'clouds': 0.5},
            WeatherType.BLIZZARD: {'fog': 0.5, 'clouds': 1.0},
            WeatherType.STORM: {'fog': 0.4, 'clouds': 1.0}
        }
        
        params = weather_params.get(self.current_weather, {'fog': 0.0, 'clouds': 0.0})
        self.fog_density = params['fog']
        self.cloud_coverage = params['clouds']

    def generate_snow_particles(self, count: int, bounds: Tuple):
        """Generate snow particles"""
        self.snow_particles = []
        for _ in range(count):
            self.snow_particles.append({
                'position': np.random.rand(3) * bounds,
                'velocity': np.array([0, -1.0, 0]) + np.random.rand(3) * 0.5,
                'size': np.random.rand() * 0.5 + 0.2
            })

    def get_sun_position(self) -> Tuple[float, float]:
        """Get sun position based on time of day"""
        # Sun elevation: high at noon, low at sunrise/sunset
        hour_angle = (self.time_of_day - 12.0) / 12.0 * np.pi
        elevation = 60.0 * np.cos(hour_angle)
        azimuth = self.time_of_day / 24.0 * 360.0
        
        return azimuth, elevation

    def __repr__(self):
        return f"WeatherSystem({self.current_weather.value}, time={self.time_of_day:.1f}h)"
