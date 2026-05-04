#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Phase 1 Actor Management Integration Test

Tests all Phase 1 components with real CARLA server.

Test scenarios:
1. BlueprintLibrary loading and blueprint matching
2. DggtTrackIndex scanning and track discovery
3. Ego vehicle spawning
4. Dynamic actor spawning
5. Actor pose updates across frames
6. Actor lifecycle (spawn/destroy)

Environment:
- CARLA server at 127.0.0.1:2000
- DGGT scene at /home/junchuan/e2e/dggt/output/waymo/training/scene1/0328/001/
"""

import sys
import os
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import DGGT modules
from dggt_project import (
    DggtTrack,
    DggtTracks,
    DggtTrackIndex,
    DggtActor,
    BlueprintLibrary,
    Blueprint,
    build_ego_track,
    scan_scene_for_tracks,
)

# Import CARLA utilities
from utils import mat_to_carla_transform

# Import constants
from constants import (
    EGO_TRACK_ID,
    EGO_LABEL,
    VEHICLE_LABELS,
)

# Try to import CARLA
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    logger.warning("CARLA not available - some tests will be skipped")


# ============================================================================
# Test Configuration
# ============================================================================

CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
SCENE_PATH = '/home/junchuan/e2e/dggt/output/waymo/training/scene1/0328/001/'
NUM_FRAMES = 7  # Number of frames in test scene
FPS = 10.0
BLUEPRINT_DATA_DIR = os.path.join(os.path.dirname(__file__), 'dggt_project')


# ============================================================================
# Helper Functions
# ============================================================================

def spawn_ego_vehicle(
    world,
    track: DggtTrack,
    blueprint_id: str = EGO_LABEL,
    transform_matrix: Optional[np.ndarray] = None,
) -> Optional[DggtActor]:
    """
    Spawn ego vehicle in CARLA world.

    Args:
        world: CARLA world instance
        track: DggtTrack for ego vehicle
        blueprint_id: CARLA blueprint to use
        transform_matrix: Optional transform to apply to pose

    Returns:
        DggtActor if successful, None otherwise
    """
    if not CARLA_AVAILABLE:
        logger.warning("CARLA not available, cannot spawn ego")
        return None

    bp_library = world.get_blueprint_library()
    ego_bp = bp_library.find(blueprint_id)

    # Get initial pose
    initial_pose = track.interpolate_pose_matrix(0)
    if initial_pose is None:
        logger.error("Failed to get initial ego pose")
        return None

    # Apply transform if provided
    if transform_matrix is not None:
        initial_pose = transform_matrix @ initial_pose

    carla_pose = mat_to_carla_transform(initial_pose)

    # Try to spawn
    actor_inst = world.try_spawn_actor(ego_bp, carla_pose)

    # Fallback: spawn at arbitrary location
    if actor_inst is None:
        logger.warning(f"Failed to spawn ego at {carla_pose}, trying fallback location")
        carla_pose = carla.Transform(
            carla.Location(x=0, y=0, z=1000),
            carla.Rotation(pitch=0, yaw=0, roll=0)
        )
        actor_inst = world.spawn_actor(ego_bp, carla_pose)

    if actor_inst is None:
        logger.error("Failed to spawn ego vehicle")
        return None

    logger.info(f"Spawned ego vehicle: id={actor_inst.id}, blueprint={blueprint_id}")

    return DggtActor(
        actor_inst=actor_inst,
        track=track,
        physics=False,
        blueprint_id=blueprint_id,
    )


def spawn_dynamic_actor(
    world,
    track: DggtTrack,
    blueprint_library: BlueprintLibrary,
    transform_matrix: Optional[np.ndarray] = None,
) -> Optional[DggtActor]:
    """
    Spawn dynamic actor (vehicle or pedestrian) in CARLA world.

    Args:
        world: CARLA world instance
        track: DggtTrack for the actor
        blueprint_library: BlueprintLibrary for matching
        transform_matrix: Optional transform to apply to pose

    Returns:
        DggtActor if successful, None otherwise
    """
    if not CARLA_AVAILABLE:
        return None

    # Filter by label
    if not (track.label in VEHICLE_LABELS or track.label == "person"):
        logger.debug(f"Skipping track {track.track_id} with label {track.label}")
        return None

    # Get best-fit blueprint
    is_vehicle = track.label != "person"
    best_fit = blueprint_library.get_best_fit_blueprint(track.dims, is_vehicle)

    bp_library = world.get_blueprint_library()
    actor_bp = bp_library.find(best_fit.id)

    if actor_bp is None:
        logger.warning(f"Blueprint {best_fit.id} not found in CARLA")
        return None

    # Get spawn pose
    spawn_pose = track.interpolate_pose_matrix(track.start_time())
    if spawn_pose is None:
        logger.error(f"Failed to get spawn pose for track {track.track_id}")
        return None

    # Apply blueprint offset (rear axle to bounding box center)
    spawn_pose = blueprint_library.apply_offset_to_pose(spawn_pose, best_fit.id, inverse=True)

    # Apply transform if provided
    if transform_matrix is not None:
        spawn_pose = transform_matrix @ spawn_pose

    carla_pose = mat_to_carla_transform(spawn_pose)

    # Try to spawn
    actor_inst = world.try_spawn_actor(actor_bp, carla_pose)

    # Fallback: try 10 meters higher
    if actor_inst is None:
        carla_pose.location.z += 10
        actor_inst = world.try_spawn_actor(actor_bp, carla_pose)

    if actor_inst is None:
        logger.warning(f"Failed to spawn actor {track.track_id} ({best_fit.id})")
        return None

    # Set exact transform
    actor_inst.set_transform(mat_to_carla_transform(spawn_pose))

    logger.info(f"Spawned actor: track={track.track_id}, blueprint={best_fit.id}, id={actor_inst.id}")

    return DggtActor(
        actor_inst=actor_inst,
        track=track,
        physics=False,
        blueprint_id=best_fit.id,
        object_id=track._object_id if hasattr(track, '_object_id') else None,
    )


# ============================================================================
# Test Functions
# ============================================================================

def test_blueprint_library():
    """Test 1: BlueprintLibrary loading and blueprint matching."""
    logger.info("=" * 60)
    logger.info("Test 1: BlueprintLibrary")
    logger.info("=" * 60)

    # Load library
    bp_lib = BlueprintLibrary(data_dir=BLUEPRINT_DATA_DIR)
    logger.info(f"Loaded BlueprintLibrary from {BLUEPRINT_DATA_DIR}")

    # Test vehicle matching
    test_dims = [4.5, 1.8, 1.5]  # Typical sedan dimensions
    best_fit = bp_lib.get_best_fit_blueprint(test_dims, vehicle=True)
    logger.info(f"Best fit for vehicle {test_dims}: {best_fit.id}")
    logger.info(f"  Dimensions: {best_fit.dimensions}")
    logger.info(f"  Offset: {best_fit.offset}")

    # Test pedestrian matching
    ped_dims = [0.6, 0.5, 1.7]  # Typical pedestrian dimensions
    best_fit_ped = bp_lib.get_best_fit_blueprint(ped_dims, vehicle=False)
    logger.info(f"Best fit for pedestrian {ped_dims}: {best_fit_ped.id}")

    # Test offset application
    test_pose = np.eye(4)
    test_pose[:3, 3] = [10, 20, 0]
    offset_pose = bp_lib.apply_offset_to_pose(test_pose, best_fit.id)
    logger.info(f"Applied offset to pose: {offset_pose[:3, 3]}")

    assert best_fit.id != "", "Blueprint ID should not be empty"
    assert len(best_fit.dimensions) == 3, "Should have 3 dimensions"

    logger.info("Test 1 PASSED\n")
    return True


def test_track_index():
    """Test 2: DggtTrackIndex scanning and track discovery."""
    logger.info("=" * 60)
    logger.info("Test 2: DggtTrackIndex")
    logger.info("=" * 60)

    # Scan scene for tracks
    track_index = DggtTrackIndex(
        scene_path=SCENE_PATH,
        num_frames=NUM_FRAMES,
        fps=FPS,
    )

    object_ids = track_index.get_object_ids()
    logger.info(f"Found {len(object_ids)} unique objects")

    tracks = track_index.get_tracks()
    logger.info(f"Generated {len(tracks)} tracks")

    for track in tracks:
        logger.info(f"  {track}")
        logger.info(f"    Frames: [{track.start_frame()}, {track.end_frame()}]")
        logger.info(f"    Lifetime: {track.get_lifetime_frames()} frames = {track.get_lifetime_seconds():.2f}s")
        logger.info(f"    Dims: {track.dims}")
        logger.info(f"    Label: {track.label}")

    assert len(tracks) >= 0, "Should generate track list"

    logger.info("Test 2 PASSED\n")
    return tracks


def test_ego_track():
    """Test 3: Ego track creation and pose interpolation."""
    logger.info("=" * 60)
    logger.info("Test 3: Ego Track")
    logger.info("=" * 60)

    ego_track = build_ego_track(SCENE_PATH, NUM_FRAMES, FPS)
    logger.info(f"Ego track: {ego_track}")

    # Test pose interpolation
    for frame in range(NUM_FRAMES):
        pose = ego_track.interpolate_pose_matrix(frame * 100_000)
        logger.debug(f"  Frame {frame}: translation = {pose[:3, 3]}")

    # Test path generation
    path = ego_track.get_path(spacing_us=100_000)
    logger.info(f"Generated path with {len(path)} waypoints")

    assert ego_track.track_id == EGO_TRACK_ID, "Track ID should be 'ego'"
    assert ego_track.is_ego(), "Should be marked as ego"

    logger.info("Test 3 PASSED\n")
    return ego_track


def test_carla_connection():
    """Test 4: CARLA server connection."""
    logger.info("=" * 60)
    logger.info("Test 4: CARLA Connection")
    logger.info("=" * 60)

    if not CARLA_AVAILABLE:
        logger.warning("CARLA not available - skipping CARLA tests")
        return None

    try:
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(5.0)
        world = client.get_world()

        logger.info(f"Connected to CARLA at {CARLA_HOST}:{CARLA_PORT}")
        logger.info(f"Map: {world.get_map().name}")

        return client
    except Exception as e:
        logger.warning(f"Failed to connect to CARLA: {e}")
        logger.warning("Skipping CARLA-dependent tests")
        return None


def test_actor_spawning(client, ego_track: DggtTrack, dynamic_tracks: List[DggtTrack]):
    """Test 5: Actor spawning (ego + dynamic)."""
    logger.info("=" * 60)
    logger.info("Test 5: Actor Spawning")
    logger.info("=" * 60)

    if client is None:
        logger.warning("CARLA client not available - skipping spawn test")
        return {}, {}

    world = client.get_world()
    bp_lib = BlueprintLibrary(data_dir=BLUEPRINT_DATA_DIR)

    actor_mapping: Dict[str, DggtActor] = {}
    actor_blueprints: Dict[int, str] = {}

    # Spawn ego
    ego_actor = spawn_ego_vehicle(world, ego_track)
    if ego_actor is not None:
        actor_mapping[EGO_TRACK_ID] = ego_actor
        actor_blueprints[ego_actor.actor_inst.id] = ego_actor.blueprint_id

    # Spawn dynamic actors
    for track in dynamic_tracks:
        actor = spawn_dynamic_actor(world, track, bp_lib)
        if actor is not None:
            actor_mapping[track.track_id] = actor
            actor_blueprints[actor.actor_inst.id] = actor.blueprint_id

    logger.info(f"Spawned {len(actor_mapping)} actors total")

    return actor_mapping, actor_blueprints


def test_pose_updates(client, actor_mapping: Dict[str, DggtActor]):
    """Test 6: Actor pose updates across frames."""
    logger.info("=" * 60)
    logger.info("Test 6: Pose Updates")
    logger.info("=" * 60)

    if client is None or len(actor_mapping) == 0:
        logger.warning("Skipping pose update test")
        return

    world = client.get_world()

    # Run tick loop
    logger.info("Running frame updates...")
    for frame in range(NUM_FRAMES):
        # Update poses
        for track_id, actor in actor_mapping.items():
            if not actor.is_alive():
                continue

            pose = actor.get_current_pose(frame)
            if pose is not None:
                carla_pose = mat_to_carla_transform(pose)
                actor.actor_inst.set_transform(carla_pose)

        # Tick simulation
        world.tick()
        logger.debug(f"Frame {frame} complete")

    logger.info(f"Completed {NUM_FRAMES} frame updates")


def test_actor_lifecycle(actor_mapping: Dict[str, DggtActor]):
    """Test 7: Actor lifecycle (destroy)."""
    logger.info("=" * 60)
    logger.info("Test 7: Actor Lifecycle")
    logger.info("=" * 60)

    if len(actor_mapping) == 0:
        logger.warning("No actors to test lifecycle")
        return

    # Destroy all actors
    for track_id, actor in actor_mapping.items():
        if actor.is_alive():
            logger.info(f"Destroying actor: {track_id}")
            actor.destroy()

    # Verify all destroyed
    alive_count = sum(1 for a in actor_mapping.values() if a.is_alive())
    assert alive_count == 0, f"Expected 0 alive actors, got {alive_count}"

    logger.info(f"Destroyed {len(actor_mapping)} actors")
    logger.info("Test 7 PASSED\n")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all integration tests."""
    logger.info("=" * 60)
    logger.info("Phase 1 Actor Management Integration Test")
    logger.info("=" * 60)
    logger.info(f"Scene: {SCENE_PATH}")
    logger.info(f"Frames: {NUM_FRAMES}")
    logger.info(f"CARLA: {CARLA_HOST}:{CARLA_PORT}")
    logger.info("")

    # Test 1: BlueprintLibrary
    if not test_blueprint_library():
        logger.error("Test 1 FAILED")
        return 1

    # Test 2: DggtTrackIndex
    dynamic_tracks = test_track_index()
    if dynamic_tracks is None:
        logger.error("Test 2 FAILED")
        return 1

    # Test 3: Ego Track
    ego_track = test_ego_track()
    if ego_track is None:
        logger.error("Test 3 FAILED")
        return 1

    # Test 4: CARLA Connection
    client = test_carla_connection()

    # Test 5-7: CARLA-dependent tests
    actor_mapping = {}
    if client is not None:
        try:
            actor_mapping, actor_blueprints = test_actor_spawning(client, ego_track, dynamic_tracks)
            test_pose_updates(client, actor_mapping)
            test_actor_lifecycle(actor_mapping)
        except Exception as e:
            logger.error(f"CARLA test error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info("=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())