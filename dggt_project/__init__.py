# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Project Package

Phase 1-4 implementation: Actor Management, Track System, and OpenDRIVE for DGGT integration.

Modules:
- dggt_track: Pose interpolation and track management
  - DggtInterpolatedPoses: Base class for pose interpolation with lazy loading
  - DggtTrack: Single object trajectory with metadata
  - DggtTracks: Collection manager for track lifecycle
  - DggtTrackIndex: Scanner for building track continuity from per-frame data
- dggt_integration: CARLA actor wrapper (DggtActor)
- dggt_blueprint_library: Blueprint matching (BlueprintLibrary, Blueprint)
- dggt_trajectory_follower: Trajectory following with PID control (DggtTrajectoryFollower)
- dggt_scenario: Traffic Manager integration (DggtScenario)
- dggt_projection: Coordinate transformations for OpenDRIVE alignment
  - lat_lng_alt_2_ECEF_elipsoidal: GPS to ECEF conversion
  - ecef_2_ENU: ECEF to ENU transformation matrix
  - get_t_rig_enu_from_ecef: Scene-to-map coordinate alignment
- dggt_opendrive: OpenDRIVE world generation
  - DggtOpendriveWorld: Detects and loads map.xodr, generates CARLA world
"""

from .dggt_track import (
    DggtInterpolatedPoses,
    DggtTrack,
    DggtTracks,
    DggtTrackIndex,
    build_ego_track,
    scan_scene_for_tracks,
)
from .dggt_integration import DggtActor
from .dggt_blueprint_library import BlueprintLibrary, Blueprint
from .dggt_trajectory_follower import DggtTrajectoryFollower
from .dggt_scenario import DggtScenario
from .dggt_projection import (
    lat_lng_alt_2_ECEF_elipsoidal,
    ecef_2_ENU,
    get_t_rig_enu_from_ecef,
)
from .dggt_opendrive import DggtOpendriveWorld

__all__ = [
    # Track classes
    "DggtInterpolatedPoses",
    "DggtTrack",
    "DggtTracks",
    "DggtTrackIndex",
    "build_ego_track",
    "scan_scene_for_tracks",
    # Actor classes
    "DggtActor",
    # Blueprint classes
    "BlueprintLibrary",
    "Blueprint",
    # Trajectory follower
    "DggtTrajectoryFollower",
    # Scenario classes
    "DggtScenario",
    # Projection functions
    "lat_lng_alt_2_ECEF_elipsoidal",
    "ecef_2_ENU",
    "get_t_rig_enu_from_ecef",
    # OpenDRIVE classes
    "DggtOpendriveWorld",
]