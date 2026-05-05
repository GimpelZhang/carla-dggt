# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Integration Module

Provides classes for integrating DGGT data with CARLA simulation.

Classes:
- DggtActor: CARLA actor wrapper with DGGT track reference

Adapted from NuRec nurec_integration.py (NurecActor class).
"""

import math

import numpy as np
import carla
from typing import Optional

from .dggt_track import DggtTrack


class DggtActor:
    """
    CARLA actor wrapper with DGGT track reference.

    Adapted from NuRec NurecActor (nurec_integration.py:421-451).

    This class wraps a CARLA actor instance and associates it with a DggtTrack
    for pose interpolation and physics management.

    Attributes:
        actor_inst: The CARLA actor instance
        track: DggtTrack containing pose data for this actor
        physics: Whether physics simulation is enabled
        alive: Whether the actor is still active in simulation
        blueprint_id: CARLA blueprint identifier (e.g., "vehicle.tesla.model3")
        object_id: DGGT object ID for tracking
    """

    def __init__(
        self,
        actor_inst: carla.Actor,
        track: DggtTrack,
        physics: bool = False,
        blueprint_id: Optional[str] = None,
        object_id: Optional[int] = None,
    ):
        """
        Initialize DggtActor.

        Args:
            actor_inst: CARLA actor instance
            track: DggtTrack containing pose data
            physics: Initial physics state (default False)
            blueprint_id: CARLA blueprint identifier
            object_id: DGGT object ID for tracking
        """
        self.actor_inst = actor_inst
        self.track = track
        self.physics = physics
        self.alive = True
        self.blueprint_id = blueprint_id
        self.object_id = object_id

    def destroy(self) -> None:
        """
        Destroy the CARLA actor and mark as not alive.

        This method should be called when the actor's track lifetime ends
        or when the simulation is shutting down.
        """
        self.alive = False
        self.actor_inst.destroy()

    def set_physics(self, physics: bool, current_frame: int) -> None:
        """
        Enable or disable physics simulation with velocity calculation.

        When enabling physics, calculates velocity from track pose interpolation
        to maintain smooth motion. Velocity is computed from position difference
        between current and previous frames.

        Args:
            physics: True to enable physics, False to disable
            current_frame: Current frame index for velocity calculation
        """
        # Skip if physics state unchanged
        if self.physics == physics:
            return

        self.physics = physics
        self.actor_inst.set_simulate_physics(physics)

        # When enabling physics, calculate and set velocity from track
        if physics:
            # Get timestamps for velocity calculation
            # Convert frame indices to timestamps (10Hz: frame * 100000 us)
            us_per_frame = 100_000  # Default 10Hz

            min_time = self.track.start_time()
            current_time = current_frame * us_per_frame

            # Look back 1 frame (100ms) for velocity calculation
            before_time = max(min_time, current_time - us_per_frame)

            # Interpolate poses
            pose_before = self.track.interpolate_pose_matrix(before_time)
            current_pose = self.track.interpolate_pose_matrix(current_time)

            if pose_before is None or current_pose is None:
                # Cannot calculate velocity without poses
                return

            # Calculate velocity vector (m/s)
            # Position difference in meters, time difference in microseconds
            # velocity = delta_pos / delta_time (in seconds)
            # delta_time_us = current_time - before_time (in microseconds)
            # velocity_mps = delta_pos_meters * 1_000_000 / delta_time_us
            delta_time_us = current_time - before_time
            if delta_time_us <= 0:
                return

            velocity_vector = (
                (current_pose[:3, 3] - pose_before[:3, 3])
                * 1_000_000
                / delta_time_us
            )

            # Flip Y coordinate for CARLA (CARLA uses left-handed coordinate system)
            # DGGT/NuRec poses are in right-handed coordinates
            velocity_vector[1] = -velocity_vector[1]

            # Set target velocity on CARLA actor
            self.actor_inst.set_target_velocity(
                carla.Vector3D(
                    float(velocity_vector[0]),
                    float(velocity_vector[1]),
                    float(velocity_vector[2])
                )
            )

    def apply_control(self, control: carla.VehicleControl) -> None:
        """Apply vehicle control directly."""
        self.actor_inst.apply_control(control)

    def get_speed(self) -> float:
        """Get current speed in m/s."""
        v = self.actor_inst.get_velocity()
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def get_velocity_vector(self) -> carla.Vector3D:
        """Get current velocity as carla.Vector3D."""
        return self.actor_inst.get_velocity()

    def is_alive(self) -> bool:
        """Check if actor is still active."""
        return self.alive

    def get_track_id(self) -> str:
        """Get the track_id of associated DggtTrack."""
        return self.track.track_id

    def get_current_pose(self, current_frame: int) -> Optional[np.ndarray]:
        """
        Get interpolated pose for current frame.

        Args:
            current_frame: Frame index for pose interpolation

        Returns:
            4x4 pose matrix, or None if out of track range
        """
        us_per_frame = 100_000  # Default 10Hz
        timestamp = current_frame * us_per_frame
        return self.track.interpolate_pose_matrix(timestamp)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DggtActor(track_id='{self.track.track_id}', "
            f"blueprint='{self.blueprint_id}', "
            f"physics={self.physics}, alive={self.alive})"
        )