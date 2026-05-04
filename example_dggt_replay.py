#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Rendering Example Script

This script demonstrates DGGT (Gaussian Splatting) rendering integration with CARLA.
It is similar to example_nurec_replay_save_images.py but uses DGGT server rendering.

Usage:
    python example_dggt_replay.py --config configs/dggt_config.yaml
    python example_dggt_replay.py --config configs/dggt_config.yaml --num-frames 50 --output-dir ./output

Features:
    - YAML configuration loading with environment variable override
    - CARLA synchronous mode simulation
    - DGGT neural rendering via gRPC
    - Image saving to output directory
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Callable, List
import imageio

# Add module path for imports
sys.path.insert(0, str(Path(__file__).parent))

import carla
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore", message=".*Gimbal lock detected.*")

from dggt_config import DggtConfigLoader
from dggt_project import (
    DggtScenario as DggtOpendriveScenario,  # Phase 4: OpenDRIVE world generation
    DggtTracks,
    DggtTrack,
    DggtActor,
    DggtTrajectoryFollower,
    BlueprintLibrary,
    scan_scene_for_tracks,
    build_ego_track,
)
from dggt_integration import DggtScenario, DggtRenderer, DggtSensor  # Rendering infrastructure
from pygame_display import PygameDisplay  # Phase 2: Pygame display integration
from utils import mat_to_carla_transform
from constants import EGO_TRACK_ID, EGO_LABEL, VEHICLE_LABELS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Phase 2: Camera Callback Functions for Pygame Display
# ============================================================================

def make_dggt_camera_callback(
    display: PygameDisplay,
    camera_name: str,
    pygame_pos: Tuple[int, int],
    saveimages: bool = False,
    output_dir: str = "output"
) -> Callable:
    """
    Create callback for DGGT rendered images.

    Args:
        display: PygameDisplay instance for rendering
        camera_name: Camera identifier for naming saved images
        pygame_pos: Grid position (row, col) for pygame display
        saveimages: Whether to save images to disk
        output_dir: Directory for saved images

    Returns:
        Callback function that handles DGGT rendered images
    """
    name_to_index = {}

    def callback(image: np.ndarray):
        display.setImage(image, (2, 2), pygame_pos)
        if saveimages:
            next_index = name_to_index.get(camera_name, 0)
            name_to_index[camera_name] = next_index + 1
            os.makedirs(f"{output_dir}/{camera_name}", exist_ok=True)
            imageio.imwrite(
                f"{output_dir}/{camera_name}/{next_index:05d}.jpg",
                image.astype(np.uint8)
            )
    return callback


def process_carla_image(
    display: PygameDisplay,
    pygame_dims: Tuple[int, int],
    image_pos: Tuple[int, int],
    image: carla.Image,
) -> None:
    """
    Callback for CARLA camera sensor images.

    Args:
        display: PygameDisplay instance for rendering
        pygame_dims: Grid dimensions (rows, cols)
        image_pos: Grid position (row, col) for this image
        image: CARLA Image from camera sensor
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]  # Remove alpha
    array = array[:, :, ::-1]  # BGR→RGB
    display.setImage(array, pygame_dims, image_pos)


# ============================================================================
# Phase 3: Camera Configuration System
# ============================================================================

def load_camera_config(config_path: str = "configs/dggt_camera_config.yaml") -> list:
    """
    Load camera configuration from YAML file.

    Args:
        config_path: Path to camera config YAML

    Returns:
        List of camera configuration dicts
    """
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================================
# Phase 4: Camera Initialization with add_cameras Function
# ============================================================================

def add_cameras(
    scenario: DggtScenario,
    renderer: DggtRenderer,
    world: carla.World,
    ego_actor: carla.Actor,
    pygame_display: PygameDisplay,
    output_dir: str,
    saveimages: bool = False,
    resolution_ratio: float = 0.125,
    camera_config_path: str = "configs/dggt_camera_config.yaml",
) -> Tuple[List[DggtSensor], List[carla.Actor]]:
    """
    Setup cameras with pygame display integration.

    Mirrors example_nurec_replay_save_images.py:add_cameras() pattern.

    Args:
        scenario: DggtScenario instance
        renderer: DggtRenderer instance
        world: CARLA world
        ego_actor: Ego vehicle actor for camera attachment
        pygame_display: PygameDisplay instance
        output_dir: Output directory for saved images
        saveimages: Whether to save images to disk
        resolution_ratio: Resolution scaling factor
        camera_config_path: Path to camera configuration YAML

    Returns:
        Tuple of (dggt_sensors, carla_cameras) for cleanup
    """
    camera_configs = load_camera_config(camera_config_path)
    grid_size = (2, 2)

    bp_library = world.get_blueprint_library()

    # Track sensors for proper cleanup
    dggt_sensors: List[DggtSensor] = []
    carla_cameras: List[carla.Actor] = []

    for cam_cfg in camera_configs:
        # Case 1: DGGT rendered camera
        if "dggt_camera" in cam_cfg:
            dggt_cfg = cam_cfg["dggt_camera"]
            logical_id = dggt_cfg["logical_id"]
            grid_pos = tuple(dggt_cfg["grid_pos"])

            # Create DggtSensor with callback
            sensor = DggtSensor(
                renderer=renderer,
                camera_id=logical_id,
                resolution_ratio=dggt_cfg.get("resolution_ratio", resolution_ratio),
                output_dir=output_dir,
                scenario=scenario,
            )

            # Register callback for pygame display
            camera_name = f"dggt_{logical_id}"
            callback = make_dggt_camera_callback(
                pygame_display, camera_name, grid_pos, saveimages, output_dir
            )
            sensor.set_image_callback(callback)
            sensor.register_tick_callback(world)
            dggt_sensors.append(sensor)

        # Case 2: CARLA standard RGB camera
        elif "sensor" in cam_cfg:
            sensor_type = cam_cfg["sensor"]
            camera_bp = bp_library.find(f"sensor.camera.{sensor_type}")

            # Apply attributes
            for attr, value in cam_cfg.get("attributes", {}).items():
                camera_bp.set_attribute(attr, str(value))

            # Build transform
            loc = cam_cfg["transform"]["location"]
            rot = cam_cfg["transform"]["rotation"]
            camera_transform = carla.Transform(
                carla.Location(x=loc.get("x", 0), y=loc.get("y", 0), z=loc.get("z", 0)),
                carla.Rotation(pitch=rot.get("pitch", 0), yaw=rot.get("yaw", 0), roll=rot.get("roll", 0)),
            )

            # Spawn and attach to ego
            camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_actor)
            grid_pos = tuple(cam_cfg.get("grid_pos", (0, 0)))

            # Register callback
            camera.listen(
                lambda image, pos=grid_pos: process_carla_image(
                    pygame_display, grid_size, pos, image
                )
            )
            carla_cameras.append(camera)

        else:
            logger.warning(f"Unknown camera config format: {cam_cfg}")

    return dggt_sensors, carla_cameras


# ============================================================================
# Dynamic Actor Spawning and Lifecycle Management
# ============================================================================

# Default frame rate for DGGT (Waymo data is 10Hz)
DEFAULT_FPS = 10.0
US_PER_FRAME = 100_000  # 1/10 second = 100000 microseconds


def spawn_dynamic_actor(
    world: carla.World,
    track: DggtTrack,
    blueprint_library: BlueprintLibrary,
) -> DggtActor:
    """
    Spawn a dynamic actor (vehicle or pedestrian) from a DggtTrack.

    Args:
        world: CARLA world instance
        track: DggtTrack containing pose data
        blueprint_library: BlueprintLibrary for matching dimensions

    Returns:
        DggtActor if successful, None otherwise
    """
    bp_library = world.get_blueprint_library()

    # Determine if vehicle or pedestrian
    is_vehicle = track.label in VEHICLE_LABELS
    is_pedestrian = track.label == "person"

    if not (is_vehicle or is_pedestrian):
        logger.debug(f"Skipping track {track.track_id}: unsupported label '{track.label}'")
        return None

    # Get best-fit blueprint
    best_fit_blueprint = blueprint_library.get_best_fit_blueprint(
        [track.dims[2], track.dims[0], track.dims[1]], vehicle=is_vehicle
    )
    actor_bp = bp_library.find(best_fit_blueprint.id)
    if actor_bp is None:
        logger.warning(f"Blueprint {best_fit_blueprint.id} not found for track {track.track_id}")
        return None

    # Get spawn pose from track start
    spawn_pose_array = track.interpolate_pose_matrix(track.start_time())
    if spawn_pose_array is None:
        logger.warning(f"Could not interpolate spawn pose for track {track.track_id}")
        return None

    # Apply blueprint offset (rear axle correction) for vehicles
    if is_vehicle:
        spawn_pose_matrix = blueprint_library.apply_offset_to_pose(
            spawn_pose_array, best_fit_blueprint.id, inverse=True
        )
    else:
        spawn_pose_matrix = spawn_pose_array

    # Transform to CARLA coordinates
    carla_spawn_pose_mat = spawn_pose_matrix
    spawn_pose = mat_to_carla_transform(carla_spawn_pose_mat)

    # Try spawn with retry logic (small elevation since ground is in the map)
    actor_inst = world.try_spawn_actor(actor_bp, spawn_pose)
    if actor_inst is None:
        # Retry 1 meter higher (user added ground to OpenDRIVE)
        spawn_pose.location.z += 1.0
        actor_inst = world.try_spawn_actor(actor_bp, spawn_pose)
        if actor_inst is None:
            # Retry 2 meters higher as final attempt
            spawn_pose.location.z += 1.0
            actor_inst = world.try_spawn_actor(actor_bp, spawn_pose)
            if actor_inst is None:
                logger.warning(
                    f"Failed to spawn actor {track.track_id} ({best_fit_blueprint.id})"
                )
                return None

    # ---> 核心修复：立刻强制关闭该车辆的物理引擎！
    # 否则 set_transform 会与 PhysX 引擎发生冲突导致严重抽搐
    actor_inst.set_simulate_physics(False)

    # Set initial transform
    actor_inst.set_transform(mat_to_carla_transform(carla_spawn_pose_mat))

    # Create DggtActor wrapper
    dggt_actor = DggtActor(
        actor_inst,
        track,
        physics=False,
        blueprint_id=best_fit_blueprint.id,
        object_id=track.object_id if hasattr(track, 'object_id') else None,
    )

    logger.debug(f"Spawned actor: {track.track_id} -> {best_fit_blueprint.id}")
    return dggt_actor


def update_actor_position(
    dggt_actor: DggtActor,
    current_frame: int,
    blueprint_library: BlueprintLibrary,
    world: carla.World
) -> None:
    """
    Update actor position from track interpolation.
    动态计算缺失的车头朝向 (Yaw)。
    对无物理的障碍物，每帧动态捕捉地面高度以适应起伏路面。
    """
    # 1. 获取当前帧的位置
    pose_matrix = dggt_actor.get_current_pose(current_frame)
    if pose_matrix is None:
        return

    # 应用 blueprint offset
    is_vehicle = dggt_actor.track.label in VEHICLE_LABELS
    if is_vehicle and dggt_actor.blueprint_id is not None:
        pose_matrix = blueprint_library.apply_offset_to_pose(
            pose_matrix, dggt_actor.blueprint_id, inverse=True
        )

    # 提取 CARLA 基础变换
    carla_transform = mat_to_carla_transform(pose_matrix)

    # ================= 核心修复：推断真实车头朝向 =================
    # 获取下一帧的位置，用来计算移动向量
    next_pose_matrix = dggt_actor.get_current_pose(current_frame + 1)
    if next_pose_matrix is not None:
        if is_vehicle and dggt_actor.blueprint_id is not None:
            next_pose_matrix = blueprint_library.apply_offset_to_pose(
                next_pose_matrix, dggt_actor.blueprint_id, inverse=True
            )
        next_transform = mat_to_carla_transform(next_pose_matrix)

        # 计算 X 和 Y 方向的位移
        dx = next_transform.location.x - carla_transform.location.x
        dy = next_transform.location.y - carla_transform.location.y

        # 只有在车辆发生了实质性移动时才改变车头方向 (避免原地抖动)
        if math.hypot(dx, dy) > 0.05:
            # 利用位移向量计算 Yaw 偏航角
            carla_transform.rotation.yaw = math.degrees(math.atan2(dy, dx))
        else:
            # 如果没动，保持上一帧的朝向
            carla_transform.rotation.yaw = dggt_actor.actor_inst.get_transform().rotation.yaw
    else:
        # 如果是最后一帧，保持上一帧的朝向
        carla_transform.rotation.yaw = dggt_actor.actor_inst.get_transform().rotation.yaw
    # ==========================================================

    # 强制锁定 Pitch 和 Roll 为 0，确保车辆四轮平稳着地，彻底杜绝翻车和 Gimbal lock
    carla_transform.rotation.pitch = 0.0
    carla_transform.rotation.roll = 0.0

    # ================= 核心优化：动态障碍物每帧捕捉地面高度 =================
    # 全局 Z 偏移无法适应起伏路面，通过 CARLA 地图 API 将 Z 轴捕捉到实际路面上。
    # 对于有物理的车辆，物理引擎会覆盖 Z 值（此操作无害）；
    # 对于无物理的障碍物，此捕捉是唯一高度修正手段，避免浮空或陷入地下。
    wp = world.get_map().get_waypoint(
        carla_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if wp is not None:
        carla_transform.location.z = wp.transform.location.z + 0.1
    # =====================================================================

    # 应用最新的包含正确朝向的变换
    dggt_actor.actor_inst.set_transform(carla_transform)


def destroy_actor(dggt_actor: DggtActor) -> None:
    """
    Destroy a DggtActor and clean up.

    Args:
        dggt_actor: DggtActor to destroy
    """
    if dggt_actor.is_alive():
        dggt_actor.destroy()
        logger.debug(f"Destroyed actor: {dggt_actor.get_track_id()}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DGGT Rendering Example - CARLA + DGGT Neural Rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python example_dggt_replay.py --config configs/dggt_config.yaml

  # Override scene ID and frame count
  python example_dggt_replay.py --config configs/dggt_config.yaml --scene-id 001 --num-frames 50

  # Custom output directory
  python example_dggt_replay.py --config configs/dggt_config.yaml --output-dir ./my_output

  # Use specific camera
  python example_dggt_replay.py --config configs/dggt_config.yaml --camera-id front_camera
"""
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dggt_config.yaml",
        help="Path to DGGT config file (default: configs/dggt_config.yaml)"
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default=None,
        help="Scene ID (overrides config default_scene_id)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/dggt_renders",
        help="Output directory for rendered images (default: output/dggt_renders)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=20,
        help="Number of frames to render (default: 20)"
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default=None,
        help="Camera logical ID to use (default: first available camera)"
    )
    parser.add_argument(
        "--carla-host",
        type=str,
        default="localhost",
        help="CARLA server host (default: localhost)"
    )
    parser.add_argument(
        "--carla-port",
        type=int,
        default=2000,
        help="CARLA server port (default: 2000)"
    )
    parser.add_argument(
        "--resolution-ratio",
        type=float,
        default=1.0,
        help="Resolution scaling factor (default: 1.0, use 0.5 for half resolution)"
    )
    parser.add_argument(
        "--framerate",
        type=float,
        default=10.0,
        help="Rendering framerate (default: 10.0 Hz)"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start frame index for scene replay (default: 0)"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="End frame index for scene replay (default: num-frames + start-frame - 1)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    # Phase 5: New arguments for pygame multi-camera display
    parser.add_argument(
        "--saveimages",
        action="store_true",
        help="Save images to disk (default: False)"
    )
    parser.add_argument(
        "--camera-config",
        type=str,
        default="configs/dggt_camera_config.yaml",
        help="Camera configuration YAML file"
    )
    return parser.parse_args()


def main():
    """Main entry point for DGGT rendering example."""
    args = parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Step 1: Load configuration
    logger.info(f"Loading config from: {args.config}")
    try:
        config = DggtConfigLoader.from_yaml_with_env(args.config)
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {args.config}")
        logger.error(f"Create a config file or use --config with valid path")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Config validation failed: {e}")
        sys.exit(1)

    # Override scene ID if specified
    scene_id = args.scene_id or config.default_scene_id
    if not scene_id:
        logger.error("No scene ID specified. Use --scene-id or set default_scene_id in config")
        sys.exit(1)

    logger.info(f"Scene ID: {scene_id}")
    logger.info(f"DGGT server: {config.server_host}:{config.server_port}")

    # Step 2: Initialize DGGT scenario
    logger.info("Initializing DGGT scenario...")
    try:
        scenario = DggtScenario(config, scene_id)
        scenario.load_scene()
    except ValueError as e:
        logger.error(f"Scene initialization failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load scene: {e}")
        sys.exit(1)

    renderer = scenario.get_renderer()

    # Step 3: Verify cameras are available
    available_cameras = renderer.get_available_cameras()
    if not available_cameras:
        logger.error("No cameras available from DGGT server")
        renderer.disconnect()
        sys.exit(1)

    logger.info(f"Available DGGT cameras: {list(available_cameras.keys())}")
    logger.info(f"Camera configuration will be loaded from: {args.camera_config}")

    # Step 4: Connect to CARLA
    logger.info(f"Connecting to CARLA at {args.carla_host}:{args.carla_port}...")
    try:
        client = carla.Client(args.carla_host, args.carla_port)
        client.set_timeout(60.0)
        world = client.get_world()
    except RuntimeError as e:
        logger.error(f"Failed to connect to CARLA: {e}")
        renderer.disconnect()
        sys.exit(1)

    logger.info("Connected to CARLA")

    # Step 4.5: Setup pygame display (Phase 1 - 新增)
    pygame_display = PygameDisplay(
        window_title="DGGT Camera View",
        cell_width=481,
        cell_height=271
    )
    logger.info("Pygame display initialized")

    # Step 5: Setup OpenDRIVE world using DggtOpendriveScenario (Phase 4)
    # Build scene_path from config.scene_base_path and scene_id
    scene_path = os.path.join(config.scene_base_path, scene_id)
    logger.info(f"Scene path: {scene_path}")

    # Create OpenDRIVE scenario and generate world from map.xodr
    opendrive_scenario = DggtOpendriveScenario(client, scene_path)
    world = opendrive_scenario.setup_opendrive_world()

    if opendrive_scenario.world is not None:
        logger.info(f"OpenDRIVE world generated successfully")
        geo_ref = opendrive_scenario.get_opendrive_geo_reference()
        if geo_ref:
            logger.info(f"geoReference: lat={geo_ref.get('lat')}, lon={geo_ref.get('lon')}, alt={geo_ref.get('alt')}")
    else:
        logger.warning("Using default CARLA world (OpenDRIVE generation failed)")

    # Step 6: Setup synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.framerate
    world.apply_settings(settings)

    logger.info(f"Synchronous mode enabled at {args.framerate} Hz")

    # Calculate frame range for scene replay (needed for ego track)
    start_frame = args.start_frame
    end_frame = args.end_frame if args.end_frame is not None else start_frame + args.num_frames - 1
    num_frames_to_render = end_frame - start_frame + 1
    logger.info(f"Frame range: {start_frame} - {end_frame} ({num_frames_to_render} frames)")

    # Step 7: Build ego track and spawn ego vehicle
    fps = 10.0  # DGGT default 10Hz for Waymo

    # ================== 核心坐标系修正 ==================
    # 1. DGGT World -> NuRec World: 基础坐标系转换
    T_dggt_world_to_nurec_world = np.array([[ 0,  0,  1,  0.0],[-1,  0,  0,  0.0],[ 0, -1,  0,  1.5],[ 0,  0,  0,  1.0]
    ], dtype=np.float64)

    # 2. NuRec Vehicle -> DGGT Camera: 车辆基准点到相机(车顶)的变换
    T_nurec_veh_to_dggt_cam = np.array([[ 0, -1,  0,  0.0],[ 0,  0, -1,  1.5],[ 1,  0,  0,  0.0],[ 0,  0,  0,  1.0]
    ], dtype=np.float64)

    # Build ego track from scene data
    ego_track = build_ego_track(scene_path, num_frames_to_render, fps)

    # 先应用基础转换以获取原始初始点
    ego_track.set_transform(T_dggt_world_to_nurec_world)
    ego_track.set_post_transform(T_nurec_veh_to_dggt_cam)

    us_per_frame = int(1_000_000 / fps)
    initial_timestamp = start_frame * us_per_frame
    raw_initial_pose = ego_track.interpolate_pose_matrix(initial_timestamp)

    if raw_initial_pose is None:
        logger.error("Failed to get initial ego pose from track")
        sys.exit(1)

    raw_transform = mat_to_carla_transform(raw_initial_pose)

    # ---> 核心修复 1：自动寻路偏移 (Map Projection)
    # OpenDRIVE 真实的道路可能在远处，我们将 (0,0) 投影到真实路面上，计算偏移量
    carla_map = world.get_map()
    nearest_wp = carla_map.get_waypoint(raw_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)

    offset_x, offset_y, offset_z = 0.0, 0.0, 0.0
    if nearest_wp:
        offset_x = nearest_wp.transform.location.x - raw_transform.location.x
        offset_y = nearest_wp.transform.location.y - raw_transform.location.y
        offset_z = nearest_wp.transform.location.z - raw_transform.location.z
        logger.info(f"Map offset calculated from road waypoint: dX={offset_x:.2f}, dY={offset_y:.2f}, dZ={offset_z:.2f}")
    else:
        spawn_points = carla_map.get_spawn_points()
        if spawn_points:
            offset_x = spawn_points[0].location.x - raw_transform.location.x
            offset_y = spawn_points[0].location.y - raw_transform.location.y
            offset_z = spawn_points[0].location.z - raw_transform.location.z
            logger.info(f"Map offset calculated from spawn_points: dX={offset_x:.2f}, dY={offset_y:.2f}, dZ={offset_z:.2f}")

    # 构建全局偏移矩阵 (注意 NuRec 的 Y 轴方向)
    T_global_offset = np.array([[1, 0, 0, offset_x],
        [0, 1, 0, -offset_y],
        [0, 0, 1, offset_z],[0, 0, 0, 1]
    ], dtype=np.float64)

    # 将全局偏移融合进世界转换矩阵中
    T_dggt_world_to_nurec_world = T_global_offset @ T_dggt_world_to_nurec_world
    ego_track.set_transform(T_dggt_world_to_nurec_world)

    # 重新获取带上了 offset 的正确位置
    initial_pose = ego_track.interpolate_pose_matrix(initial_timestamp)
    ego_transform = mat_to_carla_transform(initial_pose)
    ego_transform.location.z += 0.5  # 略微抬高防止生成时磕底盘

    logger.info(f"Adjusted Ego initial transform on road: location={ego_transform.location}")
    # ====================================================

    # Spawn ego vehicle in CARLA
    bp_library = world.get_blueprint_library()
    ego_bp = bp_library.find(EGO_LABEL)

    ego_actor_inst = world.try_spawn_actor(ego_bp, ego_transform)
    if ego_actor_inst is None:
        logger.error("Failed to spawn ego vehicle")
        sys.exit(1)

    logger.info(f"Spawned ego vehicle: id={ego_actor_inst.id}, blueprint={EGO_LABEL}")

    # 初始阶段将 physics 设为 False (配合后续的混合控制)
    ego_dggt_actor = DggtActor(
        actor_inst=ego_actor_inst,
        track=ego_track,
        physics=False,
        blueprint_id=EGO_LABEL,
    )

    opendrive_scenario.add_actor(EGO_TRACK_ID, ego_dggt_actor)

    # Step 8: Setup trajectory follower for ego vehicle
    trajectory_follower = DggtTrajectoryFollower(
        dggt_actor=ego_dggt_actor,
        world=world,
    )

    # Calculate simulation time step
    dt = 1.0 / args.framerate  # Time step in seconds

    # Set trajectory from ego track
    trajectory_start_time = initial_timestamp
    # Calculate trajectory end time based on SIMULATION duration, not DGGT frame count
    # This ensures the trajectory timeline matches the simulation tick rate
    simulation_duration_us = int(num_frames_to_render * dt * 1_000_000)
    trajectory_end_time = trajectory_start_time + simulation_duration_us
    # Use simulation's dt for time spacing, not DGGT's us_per_frame
    time_spacing_us = int(dt * 1_000_000)
    trajectory_follower.set_trajectory_from_track(
        trajectory_start_time,
        trajectory_end_time,
        time_spacing=time_spacing_us
    )
    logger.info(f"Set trajectory with {len(trajectory_follower.trajectory_points)} points "
                f"(spacing={time_spacing_us}us, duration={simulation_duration_us}us)")

    # Start trajectory following
    trajectory_follower.start_following(0.0)  # World time starts at 0
    logger.info("Started trajectory following")

    # Step 9: Scan scene for dynamic object tracks and initialize blueprint library
    logger.info("Scanning scene for dynamic object tracks...")
    tracks_collection, _ = scan_scene_for_tracks(scene_path, num_frames_to_render, fps)
    logger.info(f"Found {len(tracks_collection.track_data)} dynamic object tracks")

    # Initialize blueprint library for dynamic actor spawning
    dggt_project_dir = os.path.join(os.path.dirname(__file__), "dggt_project")
    blueprint_library = BlueprintLibrary(data_dir=dggt_project_dir)
    logger.info("Initialized blueprint library for dynamic actor spawning")

    # ---> 核心修复：恢复动态障碍物局部旋转，把侧翻的车辆"拧"正！
    T_obj_local_rot = np.array([[ 0, -1,  0,  0.0],[ 0,  0, -1,  0.0],
        [ 1,  0,  0,  0.0],[ 0,  0,  0,  1.0]
    ], dtype=np.float64)

    # 为所有的障碍物 Track 设置转换矩阵（带全局偏移和局部旋转修复）
    for track in tracks_collection.track_data:
        track.set_transform(T_dggt_world_to_nurec_world)
        track.set_post_transform(T_obj_local_rot)

    # Dynamic actor mapping: track_id -> DggtActor
    dynamic_actor_mapping: Dict[str, DggtActor] = {}

    # Set minimum lifetime filter (filter out very short tracks)
    tracks_collection.set_minimum_lifetime_frames(3)  # At least 3 frames
    logger.info(f"Filtered tracks with minimum lifetime: {len(tracks_collection.get_all_possible_tracks())} tracks")

    # Step 10 (Phase 5): Setup cameras with pygame display using add_cameras
    output_dir = os.path.join(args.output_dir, scene_id.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)

    # Get timestamp range for scene
    start_ts, end_ts = scenario.get_timestamp_range()
    logger.info(f"Scene timestamp range: {start_ts} - {end_ts} us")

    # Use add_cameras for multi-camera setup with pygame display
    dggt_sensors, carla_cameras = add_cameras(
        scenario=scenario,
        renderer=renderer,
        world=world,
        ego_actor=ego_actor_inst,
        pygame_display=pygame_display,
        output_dir=output_dir,
        saveimages=args.saveimages,
        resolution_ratio=args.resolution_ratio,
        camera_config_path=args.camera_config,
    )
    logger.info("Cameras initialized with pygame display integration")

    # Step 11: Run simulation with ego trajectory following and dynamic actor lifecycle
    logger.info(f"Starting simulation for {num_frames_to_render} frames (frames {start_frame}-{end_frame})...")

    try:
        for frame_idx in range(num_frames_to_render):
            world_time = frame_idx * dt

            # ================= 核心修复 2：混合驱动逻辑 =================
            if frame_idx < 8:
                # 前 8 帧（约 0.8 秒）：保持物理关闭，使用位姿强行拖拽车辆完全驶入安全路面
                ego_dggt_actor.set_physics(False, frame_idx)

                current_timestamp = start_frame * us_per_frame + frame_idx * us_per_frame
                ego_pose = ego_track.interpolate_pose_matrix(current_timestamp)
                if ego_pose is not None:
                    ego_pose_carla = mat_to_carla_transform(ego_pose)
                    ego_pose_carla.location.z += 0.1  # 保持轻微悬空防止磕底盘
                    ego_actor_inst.set_transform(ego_pose_carla)

                # 同步 Follower 内部时间戳，防止后续切换时发生跳变
                trajectory_follower.last_world_time = world_time
                trajectory_follower.current_target_index = frame_idx
            else:
                # 第 8 帧起：车辆已经完全进入合法路面，开启物理引擎，平滑过渡给 PID 控制
                # set_physics 内部会自动计算当前速度并赋予车身惯性，不会产生急刹
                ego_dggt_actor.set_physics(True, frame_idx)
                control = trajectory_follower.update(world_time)
                ego_dggt_actor.apply_control(control)
            # ==========================================================

            # Update dynamic actor lifecycle
            # Step 1: Update tracks collection to get new/removed tracks
            new_tracks, tracks_to_remove = tracks_collection.update(frame_step=1)

            # Step 2: Spawn new actors from new tracks
            for track in new_tracks:
                if track.track_id not in dynamic_actor_mapping:
                    dggt_actor = spawn_dynamic_actor(
                        world, track, blueprint_library
                    )
                    if dggt_actor is not None:
                        dynamic_actor_mapping[track.track_id] = dggt_actor

            # Step 3: Update positions of all active dynamic actors
            for track_id, dggt_actor in dynamic_actor_mapping.items():
                if dggt_actor.is_alive():
                    update_actor_position(dggt_actor, frame_idx, blueprint_library, world)

            # Step 4: Destroy actors for removed tracks
            for track in tracks_to_remove:
                if track.track_id in dynamic_actor_mapping:
                    destroy_actor(dynamic_actor_mapping[track.track_id])
                    del dynamic_actor_mapping[track.track_id]

            # Tick simulation
            world.tick()

            # ================= 核心 DEBUG 日志 =================
            if frame_idx == 0 or frame_idx == 10:
                progress = trajectory_follower.get_progress()
                ego_location = ego_actor_inst.get_transform().location
                num_dynamic = len([a for a in dynamic_actor_mapping.values() if a.is_alive()])

                logger.info(f"\n{'='*50}\nDEBUG INFO FRAME {frame_idx}\n{'='*50}")

                # 1. 打印 Ego 的矩阵信息
                raw_ego_pose = ego_track._load_pose_for_frame(frame_idx)
                if raw_ego_pose is not None:
                    logger.info(f"[Ego RAW DGGT Pose (4x4)]:\n{np.array2string(raw_ego_pose, separator=', ', suppress_small=True)}")
                logger.info(f"[Ego CARLA Transform]:\n{ego_actor_inst.get_transform()}")

                # 2. 打印第一个动态障碍物的矩阵信息
                if num_dynamic > 0:
                    first_dyn_id = next(iter(dynamic_actor_mapping.keys()))
                    first_dyn_actor = dynamic_actor_mapping[first_dyn_id]

                    raw_dyn_pose = first_dyn_actor.track._load_pose_for_frame(frame_idx)
                    if raw_dyn_pose is not None:
                        logger.info(f"\n[DynObj '{first_dyn_id}' RAW DGGT Pose (4x4)]:\n{np.array2string(raw_dyn_pose, separator=', ', suppress_small=True)}")

                    transformed_dyn = first_dyn_actor.track.interpolate_pose_matrix(frame_idx * us_per_frame)
                    if transformed_dyn is not None:
                        logger.info(f"[DynObj '{first_dyn_id}' Transformed Matrix (4x4)]:\n{np.array2string(transformed_dyn, separator=', ', suppress_small=True)}")

                    logger.info(f"[DynObj '{first_dyn_id}' CARLA Transform]:\n{first_dyn_actor.actor_inst.get_transform()}")

                logger.info(f"{'='*50}\n")
            # ===================================================

        logger.info(f"Simulation complete. Total frames: {num_frames_to_render}")
        logger.info(f"Trajectory following complete: {trajectory_follower.is_complete()}")

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation error: {e}")

    # Step 12: Cleanup
    # IMPORTANT: Disable synchronous mode FIRST before destroying actors
    # In sync mode, CARLA requires tick() to respond, but after cleanup there's no tick loop
    try:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        logger.info("CARLA synchronous mode disabled")
    except Exception as e:
        logger.warning(f"Failed to disable synchronous mode: {e}")

    # Destroy pygame display (now safe since sync mode is off)
    if pygame_display is not None:
        pygame_display.destroy()
        logger.info("Pygame display destroyed")

    # Unregister and destroy DGGT sensors
    for sensor in dggt_sensors:
        sensor.unregister_tick_callback(world)
    logger.info(f"Unregistered {len(dggt_sensors)} DGGT sensor callbacks")

    # Destroy CARLA camera sensors
    for camera in carla_cameras:
        camera.destroy()
    logger.info(f"Destroyed {len(carla_cameras)} CARLA cameras")

    # Destroy all dynamic actors
    for track_id, dggt_actor in list(dynamic_actor_mapping.items()):
        destroy_actor(dggt_actor)
    dynamic_actor_mapping.clear()
    logger.info("Destroyed all dynamic actors")

    # Destroy ego vehicle
    ego_dggt_actor.destroy()
    logger.info("Destroyed ego vehicle")

    # Disconnect DGGT
    renderer.disconnect()

    logger.info(f"Rendering complete. Images saved to: {output_dir}")

    # Force terminate process - background threads (gRPC, CARLA client) may block normal exit
    # os._exit(0) immediately terminates without running Python cleanup hooks
    os._exit(0)


if __name__ == "__main__":
    main()