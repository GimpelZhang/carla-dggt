# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Scenario Module

Provides DggtScenario class for Traffic Manager integration with DGGT data
with OpenDRIVE world generation support.

Adapted from NuRec NurecScenario (nurec_integration.py:1087-1129).

Key adaptations:
- Uses DggtActor and DggtTrack instead of NurecActor and Track
- Frame-based timing (timestamp_us = frame_idx * 100000 for 10Hz)
- Path generation from DggtTrack.get_path() method
- OpenDRIVE world generation from scene map.xodr files
- Coordinate transform setup using geoReference data

Classes:
- DggtScenario: Traffic Manager integration for DGGT frame-based data
"""

import logging
import numpy as np
from typing import Dict, List, Optional

import carla

from .dggt_integration import DggtActor
from .dggt_track import DggtTrack
from .dggt_opendrive import DggtOpendriveWorld
from .dggt_projection import get_t_rig_enu_from_ecef

# Reuse constants from NuRec
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import EGO_TRACK_ID

logger = logging.getLogger(__name__)


class DggtScenario:
    """
    Traffic Manager integration for DGGT frame-based scenarios with OpenDRIVE support.

    Adapted from NuRec NurecScenario (nurec_integration.py:1087-1129).

    This class provides Traffic Manager integration for DGGT scenarios,
    enabling path-following behavior for actors using CARLA's Traffic Manager.

    OpenDRIVE world generation is supported via the DggtOpendriveWorld utility.
    If a map.xodr file is found in the scene directory, a CARLA world is generated
    from it. Otherwise, falls back to an existing CARLA map (default Town01).

    Attributes:
        client: CARLA client instance
        scene_path: Path to DGGT scene directory
        traffic_manager: CARLA TrafficManager instance (lazy initialization)
        actor_mapping: Dictionary mapping track_id to DggtActor instances
        _fps: Frame rate (default 10Hz for Waymo)
        _opendrive_world: DggtOpendriveWorld instance for OpenDRIVE handling
        _t_scenario_carla: Coordinate transform matrix (ECEF to CARLA ENU)
        _world: CARLA world instance (either OpenDRIVE-generated or default)
    """

    # Default CARLA map to use if no OpenDRIVE file found
    DEFAULT_MAP = "Town01"

    def __init__(
        self,
        client: carla.Client,
        scene_path: str,
        fps: float = 10.0,
        t_world_base: Optional[np.ndarray] = None,
    ):
        """
        Initialize DggtScenario.

        Args:
            client: CARLA client instance
            scene_path: Path to DGGT scene directory (e.g., scene/0328/001/)
            fps: Frame rate (default 10Hz for Waymo)
            t_world_base: Optional 4x4 ECEF transformation matrix for coordinate
                          alignment. If provided and OpenDRIVE map found, used to
                          compute t_scenario_carla transform.
        """
        self.client = client
        self.scene_path = scene_path
        self._fps = fps

        # Traffic Manager (lazy initialization)
        self.traffic_manager: Optional[carla.TrafficManager] = None

        # Actor mapping: track_id -> DggtActor
        self.actor_mapping: Dict[str, DggtActor] = {}

        # OpenDRIVE world support
        self._opendrive_world: Optional[DggtOpendriveWorld] = None
        self._t_scenario_carla: Optional[np.ndarray] = None
        self._world: Optional[carla.World] = None
        self._t_world_base = t_world_base

        logger.info(f"Initialized DggtScenario for scene: {scene_path}")

    def _enable_traffic_manager(self) -> None:
        """
        Enable Traffic Manager in synchronous mode.

        Lazy initialization - only creates Traffic Manager when first needed.
        Sets synchronous mode to match CARLA's synchronous simulation.
        """
        if self.traffic_manager is None:
            logger.info("Enabling traffic manager")
            self.traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.set_synchronous_mode(True)

    def get_sim_time(self) -> int:
        """
        Get current simulation time in microseconds.

        Note: This method should be overridden by a TimeKeeper implementation
        or called with a frame index parameter in actual usage.

        Default implementation returns 0 (start of scenario).

        Returns:
            Current simulation time in microseconds
        """
        # Default: return 0 (start time)
        # In actual usage, this should track elapsed frames
        return 0

    def get_sim_time_from_frame(self, frame_idx: int) -> int:
        """
        Convert frame index to simulation time in microseconds.

        Args:
            frame_idx: Frame index (0-based)

        Returns:
            Simulation time in microseconds
        """
        us_per_frame = int(1_000_000 / self._fps)
        return frame_idx * us_per_frame

    def set_follow_path(
        self,
        track_id: str,
        path: Optional[List[carla.Location]] = None,
        spacing_us: int = 1_000_000,
    ) -> None:
        """
        Set path for actor to follow using Traffic Manager.

        Adapted from NuRec NurecScenario.set_follow_path (nurec_integration.py:1087-1109).

        Enables physics and autopilot for the actor, then sets a custom path
        via Traffic Manager. The path is either provided or generated from
        the actor's DggtTrack.

        Args:
            track_id: Track identifier (e.g., EGO_TRACK_ID or "dggt_obj_0001")
            path: Optional list of carla.Location waypoints. If None, path is
                  generated from the actor's DggtTrack.
            spacing_us: Spacing between path points in microseconds (default 1s)
        """
        self._enable_traffic_manager()
        if self.traffic_manager is None:
            raise RuntimeError("Traffic manager not initialized")

        # Get actor from mapping
        actor = self.actor_mapping.get(track_id)
        if actor is None:
            raise KeyError(f"Actor with track_id '{track_id}' not found in actor_mapping")

        # Enable physics for the actor
        current_sim_time = self.get_sim_time()
        actor.set_physics(True, current_sim_time)

        # Generate path from track if not provided
        if path is None:
            # Use DggtTrack.get_path() to generate waypoints
            # DggtTrack returns 4x4 pose matrices, need to convert to carla.Location
            pose_matrices = actor.track.get_path(spacing_us, start_time=current_sim_time)
            path = self._convert_poses_to_locations(pose_matrices)

        # Enable autopilot and configure Traffic Manager
        actor.actor_inst.set_autopilot(True)

        # Set custom path via Traffic Manager
        self.traffic_manager.set_path(actor.actor_inst, path)

        # Configure Traffic Manager behavior (same as NuRec)
        self.traffic_manager.update_vehicle_lights(actor.actor_inst, True)
        self.traffic_manager.random_left_lanechange_percentage(actor.actor_inst, 0)
        self.traffic_manager.random_right_lanechange_percentage(actor.actor_inst, 0)
        self.traffic_manager.auto_lane_change(actor.actor_inst, False)
        self.traffic_manager.distance_to_leading_vehicle(actor.actor_inst, 0)
        self.traffic_manager.ignore_lights_percentage(actor.actor_inst, 100)
        self.traffic_manager.ignore_vehicles_percentage(actor.actor_inst, 100)

        logger.info(
            f"Set follow path for actor '{track_id}' with {len(path)} waypoints"
        )

    def set_ego_follow_path(
        self,
        path: Optional[List[carla.Location]] = None,
        spacing_us: int = 1_000_000,
    ) -> None:
        """
        Convenience method to set path for ego vehicle.

        Calls set_follow_path with EGO_TRACK_ID.

        Args:
            path: Optional list of carla.Location waypoints. If None, path is
                  generated from ego vehicle's DggtTrack.
            spacing_us: Spacing between path points in microseconds (default 1s)
        """
        self.set_follow_path(EGO_TRACK_ID, path, spacing_us)

    def _convert_poses_to_locations(
        self,
        pose_matrices: List,
    ) -> List[carla.Location]:
        """
        Convert 4x4 pose matrices to CARLA Location list.

        Applies CARLA coordinate transform (Y-axis flip) to convert
        from DGGT/NuRec right-handed coordinates to CARLA left-handed.

        Args:
            pose_matrices: List of 4x4 pose matrices from DggtTrack.get_path()

        Returns:
            List of carla.Location objects
        """
        import numpy as np

        locations = []
        for pose in pose_matrices:
            # Extract translation from pose matrix
            # pose[:3, 3] gives [x, y, z] in DGGT world coordinates
            x = float(pose[0, 3])
            y = float(pose[1, 3])
            z = float(pose[2, 3])

            # Apply CARLA coordinate transform (Y-axis flip)
            # CARLA uses left-handed coordinate system
            y_carla = -y

            locations.append(carla.Location(x=x, y=y_carla, z=z))

        return locations

    def add_actor(self, track_id: str, dggt_actor: DggtActor) -> None:
        """
        Add a DggtActor to the actor mapping.

        Args:
            track_id: Track identifier
            dggt_actor: DggtActor instance to add
        """
        self.actor_mapping[track_id] = dggt_actor
        logger.debug(f"Added actor '{track_id}' to mapping")

    def get_actor(self, track_id: str) -> Optional[DggtActor]:
        """
        Get DggtActor by track_id.

        Args:
            track_id: Track identifier

        Returns:
            DggtActor instance or None if not found
        """
        return self.actor_mapping.get(track_id)

    def get_ego_actor(self) -> Optional[DggtActor]:
        """
        Get ego vehicle DggtActor.

        Returns:
            DggtActor for ego vehicle or None if not found
        """
        return self.actor_mapping.get(EGO_TRACK_ID)

    def get_world(self) -> carla.World:
        """
        Get CARLA world instance.

        Returns:
            carla.World from client
        """
        return self.client.get_world()

    def destroy_all_actors(self) -> None:
        """
        Destroy all actors in actor_mapping.

        Called during scenario cleanup.
        """
        for track_id, actor in self.actor_mapping.items():
            if actor.is_alive():
                actor.destroy()
                logger.debug(f"Destroyed actor '{track_id}'")

        self.actor_mapping.clear()
        logger.info("All actors destroyed")

    # === OpenDRIVE World Generation Methods ===

    def _detect_xodr_file(self) -> Optional[str]:
        """
        Detect map.xodr file in the scene directory.

        Creates a DggtOpendriveWorld instance and attempts to detect the
        OpenDRIVE map file in the scene path.

        Returns:
            str: Path to map.xodr if found, None otherwise
        """
        if self._opendrive_world is None:
            self._opendrive_world = DggtOpendriveWorld(self.client, self.scene_path)

        xodr_path = self._opendrive_world.detect_xodr_file()
        if xodr_path:
            logger.info(f"Detected OpenDRIVE file: {xodr_path}")
        else:
            logger.warning(
                f"No map.xodr found in {self.scene_path}, will use default map"
            )
        return xodr_path

    def _load_xodr_content(self) -> Optional[str]:
        """
        Load XML content from the detected map.xodr file.

        Requires _detect_xodr_file() to have been called first.

        Returns:
            str: XML content if loaded successfully, None otherwise
        """
        if self._opendrive_world is None:
            logger.error("OpenDRIVE world not initialized, call _detect_xodr_file first")
            return None

        content = self._opendrive_world.load_xodr_content()
        if content:
            logger.debug(f"Loaded OpenDRIVE content ({len(content)} bytes)")
        return content

    def _generate_opendrive_world(
        self,
        params: Optional[carla.OpendriveGenerationParameters] = None
    ) -> Optional[carla.World]:
        """
        Generate CARLA world from OpenDRIVE data.

        Uses DggtOpendriveWorld to generate a CARLA world from the loaded
        OpenDRIVE XML content. Falls back to default map if generation fails.

        Args:
            params: Optional OpenDRIVE generation parameters. If not provided,
                    uses default parameters optimized for DGGT scenes.

        Returns:
            carla.World: Generated world if successful, None otherwise
        """
        if self._opendrive_world is None:
            self._detect_xodr_file()

        if self._opendrive_world is None or self._opendrive_world.xodr_path is None:
            logger.warning("No OpenDRIVE file available for world generation")
            return None

        # Load content if not already done
        if self._opendrive_world.xodr_content is None:
            self._load_xodr_content()

        world = self._opendrive_world.generate_world(params)
        if world:
            self._world = world
            logger.info(f"Generated OpenDRIVE world for scene: {self.scene_path}")
        else:
            logger.warning("Failed to generate OpenDRIVE world")

        return world

    def _setup_coordinate_transform(self) -> Optional[np.ndarray]:
        """
        Setup coordinate transform using geoReference from OpenDRIVE.

        Computes the transformation matrix from the scene's ECEF coordinates
        to CARLA's ENU (East-North-Up) coordinate frame using the geoReference
        data from the OpenDRIVE map.

        Requires _t_world_base (ECEF transformation) and OpenDRIVE content
        to be available.

        Returns:
            np.ndarray: 4x4 transformation matrix if computed successfully,
                        None if geoReference not found or t_world_base not set
        """
        if self._opendrive_world is None or self._opendrive_world.xodr_content is None:
            logger.warning("OpenDRIVE content not loaded for coordinate transform")
            return None

        if self._t_world_base is None:
            logger.warning("t_world_base not set, cannot compute coordinate transform")
            return None

        # Use the utility function from dggt_projection
        self._t_scenario_carla = get_t_rig_enu_from_ecef(
            self._t_world_base,
            self._opendrive_world.xodr_content
        )

        if self._t_scenario_carla is not None:
            logger.info("Computed coordinate transform from geoReference")
            logger.debug(f"t_scenario_carla: {self._t_scenario_carla}")

        return self._t_scenario_carla

    def setup_opendrive_world(
        self,
        t_world_base: Optional[np.ndarray] = None,
        params: Optional[carla.OpendriveGenerationParameters] = None
    ) -> carla.World:
        """
        Full OpenDRIVE world setup: detect, load, generate, and transform.

        Convenience method that performs the complete OpenDRIVE workflow:
        1. Detect map.xodr file
        2. Load OpenDRIVE content
        3. Generate CARLA world
        4. Setup coordinate transform (if t_world_base provided)

        Falls back to default CARLA map if no OpenDRIVE file found.

        Args:
            t_world_base: Optional ECEF transformation matrix for coordinate
                          alignment
            params: Optional OpenDRIVE generation parameters

        Returns:
            carla.World: Generated OpenDRIVE world or default map world
        """
        # Update t_world_base if provided
        if t_world_base is not None:
            self._t_world_base = t_world_base

        # Detect and attempt OpenDRIVE world generation
        xodr_path = self._detect_xodr_file()

        if xodr_path:
            world = self._generate_opendrive_world(params)
            if world:
                # Setup coordinate transform if ECEF base provided
                if self._t_world_base is not None:
                    self._setup_coordinate_transform()
                return world

        # Fallback: load default CARLA map
        logger.info(f"Loading default CARLA map: {self.DEFAULT_MAP}")
        self._world = self.client.load_world(self.DEFAULT_MAP)
        return self._world

    def get_opendrive_geo_reference(self) -> Optional[Dict]:
        """
        Get parsed geoReference from OpenDRIVE map.

        Returns:
            Dict: Dictionary with 'lat', 'lon', 'alt' keys if available,
                  None if not parsed or no OpenDRIVE file
        """
        if self._opendrive_world is None:
            return None
        return self._opendrive_world.get_geo_reference()

    @property
    def t_scenario_carla(self) -> Optional[np.ndarray]:
        """Get the coordinate transformation matrix."""
        return self._t_scenario_carla

    @property
    def world(self) -> Optional[carla.World]:
        """Get the CARLA world instance."""
        return self._world

    def __repr__(self) -> str:
        """String representation for debugging."""
        opendrive_status = "none"
        if self._opendrive_world is not None:
            if self._world is not None and self._opendrive_world.xodr_path is not None:
                opendrive_status = "generated"
            elif self._opendrive_world.xodr_path is not None:
                opendrive_status = "detected"
            else:
                opendrive_status = "not_found"

        return (
            f"DggtScenario(scene_path='{self.scene_path}', "
            f"actors={len(self.actor_mapping)}, "
            f"tm_enabled={self.traffic_manager is not None}, "
            f"opendrive={opendrive_status})"
        )

    def __enter__(self) -> 'DggtScenario':
        """
        Context manager entry.

        Sets up the OpenDRIVE world if not already done, enabling
        the scenario to be used in a 'with' statement for automatic
        cleanup.

        Returns:
            DggtScenario: Self reference for context manager
        """
        # Setup OpenDRIVE world if not already initialized
        if self._world is None:
            self.setup_opendrive_world()

        logger.info(f"Entering DggtScenario context for {self.scene_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Context manager exit.

        Destroys all actors and cleans up resources.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self.destroy_all_actors()
        logger.info(f"Exited DggtScenario context for {self.scene_path}")