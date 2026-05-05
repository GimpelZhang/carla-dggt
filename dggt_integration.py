# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Integration Module

This module provides the main integration between DGGT (Gaussian Splatting)
and CARLA simulation. It contains classes for:

- Rendering photorealistic images using DGGT's Gaussian Splatting
- Managing gRPC communication with DGGT server
- Handling coordinate transformations between DGGT and CARLA coordinate systems

Key Classes:
- DggtRenderer: Handles gRPC communication and rendering requests

Coordinate Transform Chain (CRITICAL - must follow NuRec pattern exactly):
- Import from utils.py: se3_to_grpc_pose, undo_carla_coordinate_transform, actor_to_grpc_pose
- NEVER reimplement these functions locally

Transform Chain Design:
    NuRec: dggt_pose = t_carla_nurec @ undo_carla_coordinate_transform(carla_pose)
    DGGT:  dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)

The undo_carla_coordinate_transform is applied by the caller (DggtSensor.on_tick),
so DggtRenderer.render only applies t_carla_dggt.
"""

import grpc
import numpy as np
import logging
import json
import os
from typing import Dict, Optional, List, Any, Tuple, Callable

import carla

# gRPC generated code
from nre.grpc.protos import sensorsim_pb2, sensorsim_pb2_grpc, common_pb2

# Phase 1 and Phase 4 modules
from dggt_server.scene_metadata import TrackIDMapping, DGGTSceneMetadata

# Phase 1-2: Actor management and track system
from dggt_project import (
    DggtTrack,
    DggtTracks,
    DggtTrackIndex,
    DggtActor,
    BlueprintLibrary,
    build_ego_track,
)
from constants import EGO_TRACK_ID, EGO_LABEL, VEHICLE_LABELS

# CRITICAL: Import from utils.py (same as NuRec)
# DO NOT reimplement these functions - they are the core of the transform chain
from utils import (
    se3_to_grpc_pose,           # SE3 matrix to gRPC Pose
    undo_carla_coordinate_transform,  # CARLA coordinate correction
    actor_to_grpc_pose,         # Actor pose transform chain
)

# Config
from dggt_config import DggtConfig

# Image decode
# Image decode - prefer nvimgcodec for performance, fallback to PIL/cv2
try:
    import nvidia.nvimgcodec as nvimgcodec
    _HAS_NVIMGCODEC = True
except ImportError:
    _HAS_NVIMGCODEC = False
    try:
        from PIL import Image
        import io
    except ImportError:
        import cv2
        import numpy as np

# Constants
from constants import MAX_MESSAGE_LENGTH

logger = logging.getLogger(__name__)


class DggtRenderer:
    """
    DGGT Renderer

    gRPC client for sending render requests to DGGT server.

    Coordinate Transform Handling (matches NuRec NurecRenderer pattern exactly):
    - Receives t_scenario_dggt (scene to DGGT transform)
    - Computes internal t_carla_dggt = np.linalg.inv(t_scenario_dggt)
    - Applies complete transform chain in render()

    Transform Chain (from Phase5_Detailed_Plan.md lines 604-631):

        Important: NuRec transform chain is two-step:

        Step 1 - Caller pre-applies undo (NurecSensor.on_world_tick line 503-505):
            camera_transform = undo_carla_coordinate_transform(actor_pose) @ sensor_offset

        Step 2 - generate_request applies t_carla_nurec (line 120):
            camera_pose = t_carla_nurec @ camera_pose
            # Note: camera_pose already includes undo, no duplicate application

        Actor transform (actor_to_grpc_pose internal, line 146):
            transform_matrix = t_carla_nurec @ undo_carla_coordinate_transform(transform_matrix)
            # actor_to_grpc_pose applies both undo and t_carla_nurec

        DGGT follows the exact same pattern:
        - DggtSensor.on_tick: camera_pose = undo_carla_coordinate_transform(actor_pose) @ sensor_offset
        - DggtRenderer.render: dggt_pose = t_carla_dggt @ camera_pose (no duplicate undo)
        - actor_to_grpc_pose(actor, t_carla_dggt, ...) - same function
    """

    def __init__(
        self,
        config: DggtConfig,
        scene_id: str,
        t_scenario_dggt: np.ndarray = None,
        blueprint_library: Optional[Any] = None,
        actor_blueprints: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize renderer (matches NuRec NurecRenderer.__init__ pattern)

        Args:
            config: DGGT configuration
            scene_id: Scene ID (directory name, e.g. "0328/001")
            t_scenario_dggt: Scene to DGGT transform matrix (4x4)
                           **Naming convention**: corresponds to NuRec's t_scenario_carla
                           NuRec: NurecRenderer(..., t_scenario_carla, ...)
                           DGGT: DggtRenderer(..., t_scenario_dggt, ...)
                           If None, uses identity matrix
            blueprint_library: Blueprint library for offset calculations (optional)
            actor_blueprints: Actor ID to Blueprint ID mapping (optional)

        Note:
            NuRec NurecRenderer receives t_scenario_carla parameter:
            self.t_carla_nurec = np.linalg.inv(t_scenario_carla)

            DGGT uses the same pattern:
            self._t_carla_dggt = np.linalg.inv(t_scenario_dggt)
        """
        self._config = config
        self._scene_id = scene_id

        # Scene -> DGGT transform matrix
        self._t_scenario_dggt = t_scenario_dggt if t_scenario_dggt is not None else np.eye(4)

        # CARLA -> DGGT transform matrix (inverse, matches NuRec pattern)
        # NuRec: self.t_carla_nurec = np.linalg.inv(t_scenario_carla)
        self._t_carla_dggt = np.linalg.inv(self._t_scenario_dggt)

        # gRPC connection
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[sensorsim_pb2_grpc.SensorsimServiceStub] = None

        # Available cameras
        self._available_cameras: Dict[str, sensorsim_pb2.CameraSpec] = {}

        # ID mapping
        self._track_id_mapping = TrackIDMapping()

        # Image decoder (nvimgcodec for GPU acceleration, PIL fallback)
        if _HAS_NVIMGCODEC:
            self._jpeg_decoder = nvimgcodec.Decoder()
        else:
            self._jpeg_decoder = None

        # Scene metadata (optional, loaded separately)
        self._scene_metadata: Optional[DGGTSceneMetadata] = None

        # Blueprint library and actor blueprints (for dynamic object pose offset)
        self._blueprint_library = blueprint_library
        self._actor_blueprints = actor_blueprints

        logger.info(f"DggtRenderer initialized for scene: {scene_id}")

    def connect(self) -> None:
        """Establish gRPC connection"""
        if self._stub is not None:
            logger.warning("Already connected")
            return

        address = f"{self._config.server_host}:{self._config.server_port}"
        logger.info(f"Connecting to DGGT server at {address}")

        # Create gRPC channel
        self._channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", self._config.max_message_length),
                ("grpc.max_receive_message_length", self._config.max_message_length),
            ],
        )

        # Create service stub
        self._stub = sensorsim_pb2_grpc.SensorsimServiceStub(self._channel)

        # Fetch available cameras
        self._fetch_available_cameras()

        logger.info("Connected to DGGT server")

    def disconnect(self) -> None:
        """Close gRPC connection"""
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None
            logger.info("Disconnected from DGGT server")

    def _fetch_available_cameras(self) -> None:
        """Fetch available camera list from server"""
        request = sensorsim_pb2.AvailableCamerasRequest(scene_id=self._scene_id)
        response = self._stub.get_available_cameras(request)

        self._available_cameras = {}
        for cam in response.available_cameras:
            self._available_cameras[cam.logical_id] = cam.intrinsics

        logger.info(f"Available cameras: {list(self._available_cameras.keys())}")

    def get_available_cameras(self) -> Dict[str, sensorsim_pb2.CameraSpec]:
        """Return available cameras dict"""
        return self._available_cameras.copy()

    def get_camera_spec(self, camera_logical_id: str) -> sensorsim_pb2.CameraSpec:
        """Get camera spec by logical ID"""
        return self._available_cameras[camera_logical_id]

    def render(
        self,
        world_snapshot,  # carla.WorldSnapshot
        camera_spec: sensorsim_pb2.CameraSpec,
        camera_pose: np.ndarray,
        timestamp: int,
        resolution_ratio: float = 1.0,
        active_actors: Optional[Dict[int, str]] = None,
        controllable_tracks: Optional[set] = None,
        use_raw_pose: bool = False,
        dynamic_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Render single frame image

        Args:
            world_snapshot: CARLA world snapshot
            camera_spec: Camera specification
            camera_pose: Camera pose (4x4 matrix)
            timestamp: Timestamp (microseconds)
            resolution_ratio: Resolution scaling factor
            active_actors: Active actor mapping {actor_id: track_id}
            controllable_tracks: Controllable track_id set
            use_raw_pose: If True, camera_pose is already in DGGT coords (scene replay mode)
                          If False, camera_pose is in CARLA coords (needs t_carla_dggt transform)
            dynamic_objects: Pre-loaded dynamic objects list (scene replay mode).
                          If provided, bypasses _collect_dynamic_objects and uses these directly.
                          Format: List of dicts with object_id, pose_world (4x4 np.array), dimensions.

        Returns:
            np.ndarray: RGB image [H, W, 3]

        Raises:
            RuntimeError: Render failed

        Transform Chain (CARLA mode - Phase5_Detailed_Plan.md lines 737-744):
            Step 1 - DggtSensor.on_tick pre-applies undo:
                camera_pose = undo_carla_coordinate_transform(actor_pose) @ sensor_offset
            Step 2 - render() applies t_carla_dggt only:
                dggt_pose = t_carla_dggt @ camera_pose  # undo already applied

        Scene Replay Mode (use_raw_pose=True):
            - camera_pose is raw camera_extrinsics_world from scene files
            - Already in DGGT coordinate system - NO transforms applied
            - dynamic_objects contains pre-loaded scene objects (poses already in DGGT coords)
        """
        if self._stub is None:
            raise RuntimeError("Not connected to DGGT server")

        # Step 1: Coordinate transform (CARLA -> DGGT)
        if use_raw_pose:
            # Scene replay mode: pose is already in DGGT coords, no transform needed
            dggt_pose = camera_pose
        elif self._config.coordinate_transform_enabled:
            # CARLA mode: camera_pose already has undo applied by caller
            # NuRec pattern: generate_request line 120 only applies t_carla_nurec @ camera_pose
            dggt_pose = self._t_carla_dggt @ camera_pose  # Only apply t_carla_dggt
        else:
            dggt_pose = camera_pose

        # Step 2: Build request
        request = self._build_render_request(
            camera_spec=camera_spec,
            camera_pose=dggt_pose,
            timestamp=timestamp,
            resolution_ratio=resolution_ratio,
            world_snapshot=world_snapshot,
            active_actors=active_actors or {},
            controllable_tracks=controllable_tracks or set(),
            dynamic_objects=dynamic_objects,
        )

        # Step 3: Send gRPC request
        try:
            response = self._stub.render_rgb(request, timeout=self._config.request_timeout)
        except grpc.RpcError as e:
            raise RuntimeError(f"gRPC render_rgb failed: {e.code()}: {e.details()}")

        # Step 4: Decode image
        image = self._decode_image(response.image_bytes)

        return image

    def _build_render_request(
        self,
        camera_spec: sensorsim_pb2.CameraSpec,
        camera_pose: np.ndarray,
        timestamp: int,
        resolution_ratio: float,
        world_snapshot = None,
        active_actors: Dict[int, str] = {},
        controllable_tracks: set = set(),
        dynamic_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> sensorsim_pb2.RGBRenderRequest:
        """Build render request

        Args:
            camera_spec: Camera specification
            camera_pose: Camera pose (4x4, DGGT coordinate system)
            timestamp: Timestamp (microseconds)
            resolution_ratio: Resolution scaling factor
            world_snapshot: CARLA world snapshot (optional, for dynamic objects)
            active_actors: Active actor mapping (optional)
            controllable_tracks: Controllable tracks (optional)
            dynamic_objects: Pre-loaded dynamic objects list (scene replay mode).
                          If provided, bypasses _collect_dynamic_objects and uses these directly.
                          Format: List of dicts with object_id, pose_world (4x4 np.array), dimensions.

        Returns:
            sensorsim_pb2.RGBRenderRequest

        Note:
            Uses se3_to_grpc_pose from utils.py for pose conversion (Phase5_Detailed_Plan.md line 776)
        """
        # Use utils.py's se3_to_grpc_pose (matches NuRec exactly)
        pose = se3_to_grpc_pose(camera_pose)

        # Build dynamic objects list
        # Priority: use pre-loaded dynamic_objects if provided (scene replay mode)
        # Otherwise collect from world_snapshot (CARLA mode)
        if dynamic_objects is not None:
            # Scene replay mode: convert pre-loaded objects to gRPC format
            dynamic_objects_pb = self._convert_preloaded_dynamic_objects(dynamic_objects)
        elif world_snapshot is not None and active_actors and controllable_tracks:
            # CARLA mode: collect dynamic objects from world snapshot
            dynamic_objects_pb = self._collect_dynamic_objects(
                world_snapshot,
                active_actors,
                controllable_tracks,
            )
        else:
            dynamic_objects_pb = []

        # Build request
        request = sensorsim_pb2.RGBRenderRequest(
            scene_id=self._scene_id,
            resolution_h=int(camera_spec.resolution_h * resolution_ratio),
            resolution_w=int(camera_spec.resolution_w * resolution_ratio),
            camera_intrinsics=camera_spec,
            frame_start_us=timestamp,
            frame_end_us=timestamp + 1,  # Must be different (NuRec protocol requirement)
            sensor_pose=sensorsim_pb2.PosePair(
                start_pose=pose,
                end_pose=pose,
            ),
            dynamic_objects=dynamic_objects_pb,
            image_format=sensorsim_pb2.ImageFormat.JPEG
                if self._config.image_format == "JPEG"
                else sensorsim_pb2.ImageFormat.PNG,
            image_quality=int(self._config.image_quality),
        )

        return request

    def _convert_preloaded_dynamic_objects(
        self,
        dynamic_objects: List[Dict[str, Any]],
    ) -> List[sensorsim_pb2.DynamicObject]:
        """
        Convert pre-loaded dynamic objects to gRPC format (scene replay mode).

        Args:
            dynamic_objects: Pre-loaded dynamic objects list from DggtScenario.get_dynamic_objects()
                          Format: List of dicts with object_id, pose_world (4x4 np.array), dimensions

        Returns:
            List[sensorsim_pb2.DynamicObject]: gRPC DynamicObject list

        Note:
            In scene replay mode, pose_world is ALREADY in DGGT coordinate system.
            No coordinate transforms applied - just convert to gRPC Pose format.

        Debug Logging (for pose flow tracing):
            - pose_world matrix values from JSON
            - resulting gRPC Pose values (position and quaternion)
            - track_id being used
        """
        dynamic_objects_pb = []

        for obj in dynamic_objects:
            # pose_world is already in DGGT coords (world frame)
            pose_world_matrix = obj["pose_world"]

            # DEBUG: Log pose_world matrix values from JSON
            logger.debug(f"[POSE_TRACE] object_id={obj.get('object_id', 0)}")
            logger.debug(f"[POSE_TRACE]   pose_world matrix:\n{pose_world_matrix}")
            logger.debug(f"[POSE_TRACE]   position from matrix: [{pose_world_matrix[0,3]:.6f}, {pose_world_matrix[1,3]:.6f}, {pose_world_matrix[2,3]:.6f}]")

            # Extract rotation matrix for debug logging
            from scipy.spatial.transform import Rotation as R
            rot_matrix = pose_world_matrix[:3, :3]
            q_scipy = R.from_matrix(rot_matrix).as_quat(canonical=False)  # [x, y, z, w]
            logger.debug(f"[POSE_TRACE]   rotation matrix:\n{rot_matrix}")
            logger.debug(f"[POSE_TRACE]   scipy quaternion (x,y,z,w): [{q_scipy[0]:.6f}, {q_scipy[1]:.6f}, {q_scipy[2]:.6f}, {q_scipy[3]:.6f}]")

            # Convert to gRPC Pose format using se3_to_grpc_pose
            pose = se3_to_grpc_pose(pose_world_matrix)

            # DEBUG: Log resulting gRPC Pose values
            logger.debug(f"[POSE_TRACE]   gRPC Pose vec: (x={pose.vec.x:.6f}, y={pose.vec.y:.6f}, z={pose.vec.z:.6f})")
            logger.debug(f"[POSE_TRACE]   gRPC Pose quat: (w={pose.quat.w:.6f}, x={pose.quat.x:.6f}, y={pose.quat.y:.6f}, z={pose.quat.z:.6f})")

            # Use object_id as track_id for scene replay mode
            # Format must match dggt_client.py: "dggt_obj_{object_id:04d}"
            track_id = f"dggt_obj_{obj.get('object_id', 0):04d}"
            logger.debug(f"[POSE_TRACE]   track_id: {track_id}")

            dynamic_objects_pb.append(sensorsim_pb2.DynamicObject(
                track_id=track_id,
                pose_pair=sensorsim_pb2.PosePair(start_pose=pose, end_pose=pose),
            ))

        logger.debug(f"Converted {len(dynamic_objects_pb)} pre-loaded dynamic objects")
        return dynamic_objects_pb

    def _collect_dynamic_objects(
        self,
        world_snapshot,  # carla.WorldSnapshot
        active_actors: Dict[int, str],  # actor_id -> track_id
        controllable_tracks: set,
    ) -> List[sensorsim_pb2.DynamicObject]:
        """
        Collect dynamic objects information

        Args:
            world_snapshot: CARLA world snapshot
            active_actors: Active actor mapping {actor_id: track_id}
            controllable_tracks: Controllable track_id set

        Returns:
            DynamicObject list

        Transform Chain (Phase5_Detailed_Plan.md lines 832-859):
            NuRec code (nurec_integration.py:120-139):
                camera_pose = t_carla_nurec @ camera_pose
                for actor in actors:
                    if actor.id in active_actors:
                        track_id = active_actors[actor.id]
                        if track_id in controllable_tracks:
                            pose = actor_to_grpc_pose(actor, t_carla_nurec, blueprint_library, actor_blueprints)
                            dynamic_objects.append(DynamicObject(...))

            DGGT code (exact same pattern):
                pose = actor_to_grpc_pose(actor, self._t_carla_dggt, blueprint_library, actor_blueprints)
                dynamic_objects.append(DynamicObject(...))

        Note:
            actor_to_grpc_pose internal transform chain (utils.py:119-159):
                transform_matrix = np.array(actor.get_transform().get_matrix()).reshape(4, 4)
                # Apply blueprint offset if available
                transform_matrix = blueprint_library.apply_offset_to_pose(...)
                # Apply coordinate transform
                transform_matrix = t_carla_nurec @ undo_carla_coordinate_transform(transform_matrix)
                return Pose(vec=..., quat=...)

            DGGT uses the same actor_to_grpc_pose function, passing t_carla_dggt instead of t_carla_nurec
        """
        dynamic_objects = []

        for actor in world_snapshot:
            if actor.id not in active_actors:
                continue

            track_id = active_actors[actor.id]
            if track_id not in controllable_tracks:
                continue

            # Use actor_to_grpc_pose (most recommended, matches NuRec exactly)
            # actor_to_grpc_pose applies: t_carla_dggt @ undo_carla_coordinate_transform(transform_matrix)
            pose = actor_to_grpc_pose(
                actor,
                self._t_carla_dggt,  # Corresponds to NuRec's t_carla_nurec
                self._blueprint_library,
                self._actor_blueprints,
            )

            dynamic_objects.append(sensorsim_pb2.DynamicObject(
                track_id=track_id,
                pose_pair=sensorsim_pb2.PosePair(start_pose=pose, end_pose=pose),
            ))

        return dynamic_objects

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode image bytes to numpy array

        Args:
            image_bytes: JPEG/PNG encoded image bytes

        Returns:
            np.ndarray: RGB image [H, W, 3] uint8
        """
        if _HAS_NVIMGCODEC and self._jpeg_decoder is not None:
            # Use nvimgcodec for GPU-accelerated decoding
            image = self._jpeg_decoder.decode(image_bytes)
            image = np.array(image.cpu()).astype(np.uint8)
        else:
            # Fallback to PIL for CPU decoding
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_bytes))
            image = np.array(image).astype(np.uint8)
            # PIL returns RGB, no conversion needed
        return image

    def set_blueprint_library(self, blueprint_library: Any, actor_blueprints: Dict[int, str]) -> None:
        """Set blueprint library for offset calculations

        Args:
            blueprint_library: BlueprintLibrary instance
            actor_blueprints: Actor ID to Blueprint ID mapping
        """
        self._blueprint_library = blueprint_library
        self._actor_blueprints = actor_blueprints

    def set_scene_metadata(self, metadata: DGGTSceneMetadata) -> None:
        """Set scene metadata

        Args:
            metadata: DGGTSceneMetadata instance
        """
        self._scene_metadata = metadata


class DggtScenario:
    """
    DGGT Scene Management

    Similar to NurecScenario, manages scene loading and rendering.

    Transform Handling (matches NuRec pattern exactly):
    - NuRec: t_scenario_carla = get_t_rig_enu_from_ecef(t_world_base, data)
             t_carla_nurec = np.linalg.inv(t_scenario_carla)
    - DGGT: t_scenario_dggt = from scene metadata or georeference
            t_carla_dggt = np.linalg.inv(t_scenario_dggt)
    """

    def __init__(
        self,
        config: DggtConfig,
        scene_id: Optional[str] = None,
        carla_world: Optional[Any] = None,
    ):
        """
        Initialize DGGT scenario

        Args:
            config: DGGT configuration
            scene_id: Scene ID (optional, defaults to config.default_scene_id)
            carla_world: CARLA world instance (optional)

        Raises:
            ValueError: If scene path not found
        """
        from pathlib import Path

        self._config = config
        self._scene_id = scene_id or config.default_scene_id
        self._carla_world = carla_world

        # Scene path
        self._scene_path = Path(config.scene_base_path) / self._scene_id
        if not self._scene_path.exists():
            raise ValueError(f"Scene path not found: {self._scene_path}")

        # Scene manager (Phase 1)
        from dggt_server.scene_manager import DGGTSceneManager
        self._scene_manager = DGGTSceneManager(str(self._scene_path))
        # Use "." as scene_id since _scene_path already points to the scene directory
        self._scene_metadata = self._scene_manager.get_scene(".")

        # Transform derivation (matches NuRec pattern)
        self._t_scenario_dggt = self._compute_transform_from_scene()
        self._t_carla_dggt = np.linalg.inv(self._t_scenario_dggt)

        # Renderer
        self._renderer: Optional[DggtRenderer] = None

        # Time range
        self._start_timestamp = self._scene_metadata.start_timestamp_us
        self._end_timestamp = self._scene_metadata.end_timestamp_us

        # Ego pose cache (for efficiency)
        self._ego_pose_cache: Dict[int, Tuple[np.ndarray, np.ndarray, int, int]] = {}

        # Actor management (Phase 1-2)
        self.actor_mapping: Dict[str, DggtActor] = {}  # track_id -> DggtActor
        self.actor_blueprints: Dict[int, str] = {}  # actor.id -> blueprint_id
        self.actors_to_disable_physics: List[DggtActor] = []
        self._blueprint_library: Optional[BlueprintLibrary] = None

        logger.info(f"DggtScenario initialized: {self._scene_id}")

    def _compute_transform_from_scene(self) -> np.ndarray:
        """
        Compute transform matrix from scene metadata

        Sources (priority order):
        1. Config t_scenario_dggt (manual override)
        2. Scene metadata georeference (using CoordinateTransform)
        3. Scene metadata t_scenario_dggt field
        4. Default identity matrix

        Returns:
            4x4 transform matrix (scenario to DGGT)
        """
        # 1. Check config override
        if self._config.t_scenario_dggt is not None:
            return self._config.t_scenario_dggt

        # 2. Check scene metadata t_scenario_dggt
        if hasattr(self._scene_metadata, 't_scenario_dggt') and \
           self._scene_metadata.t_scenario_dggt is not None:
            return self._scene_metadata.t_scenario_dggt

        # 3. Check georeference (if available)
        if hasattr(self._scene_metadata, 'georeference') and \
           self._scene_metadata.georeference is not None:
            try:
                from dggt_server.coordinate_transform import CoordinateTransform
                ct = CoordinateTransform.from_georeference(self._scene_metadata.georeference)
                return ct.get_t_scenario_dggt()
            except Exception as e:
                logger.warning(f"Failed to compute transform from georeference: {e}")

        # 4. Default identity
        logger.info("Using identity transform (no georeference or config override)")
        return np.eye(4)

    def load_scene(self) -> None:
        """
        Load scene and connect renderer

        Initializes DggtRenderer with computed transform.
        """
        self._renderer = DggtRenderer(
            self._config,
            self._scene_id,
            t_scenario_dggt=self._t_scenario_dggt,
        )
        self._renderer.connect()
        logger.info(f"Scene loaded: {self._scene_id}")

    def get_renderer(self) -> DggtRenderer:
        """
        Get renderer instance

        Returns:
            DggtRenderer

        Raises:
            RuntimeError: Scene not loaded
        """
        if self._renderer is None:
            raise RuntimeError("Scene not loaded, call load_scene() first")
        return self._renderer

    def get_timestamp_range(self) -> tuple:
        """
        Get timestamp range

        Returns:
            Tuple[int, int]: (start_timestamp_us, end_timestamp_us)
        """
        return (self._start_timestamp, self._end_timestamp)

    def get_scene_metadata(self) -> DGGTSceneMetadata:
        """
        Get scene metadata

        Returns:
            DGGTSceneMetadata
        """
        return self._scene_metadata

    def get_scene_id(self) -> str:
        """Get scene ID"""
        return self._scene_id

    def get_t_carla_dggt(self) -> np.ndarray:
        """Get CARLA to DGGT transform matrix"""
        return self._t_carla_dggt

    def _load_ego_pose(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Load ego pose data from scene's ego_pose/frame_XXXX_ego.json files.

        Args:
            frame_idx: Frame index (0-based)

        Returns:
            Tuple of (camera_extrinsics_world, camera_intrinsics, width, height):
            - camera_extrinsics_world: 4x4 numpy array (camera to world transform)
            - camera_intrinsics: 3x3 numpy array (intrinsic matrix)
            - width: Image width in pixels
            - height: Image height in pixels

        Raises:
            FileNotFoundError: If ego pose file not found
            KeyError: If required fields missing in ego pose data
        """
        # Check cache first
        if frame_idx in self._ego_pose_cache:
            return self._ego_pose_cache[frame_idx]

        # Load ego pose file
        ego_path = os.path.join(
            str(self._scene_path),
            "ego_pose",
            f"frame_{frame_idx:04d}_ego.json"
        )

        if not os.path.exists(ego_path):
            raise FileNotFoundError(f"Ego pose file not found: {ego_path}")

        with open(ego_path, "r") as f:
            ego_data = json.load(f)

        # Extract required fields (matches dggt_client.py pattern)
        camera_extrinsics_world = np.array(ego_data["camera_extrinsics_world"])
        camera_intrinsics = np.array(ego_data["camera_intrinsics"])
        width = ego_data["camera"]["width"]
        height = ego_data["camera"]["height"]

        # Cache for efficiency
        self._ego_pose_cache[frame_idx] = (camera_extrinsics_world, camera_intrinsics, width, height)

        logger.debug(f"Loaded ego pose for frame {frame_idx}: {ego_path}")
        return camera_extrinsics_world, camera_intrinsics, width, height

    def get_ego_pose(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Get ego pose data for a frame (public method).

        Args:
            frame_idx: Frame index (0-based)

        Returns:
            Tuple of (camera_extrinsics_world, camera_intrinsics, width, height):
            - camera_extrinsics_world: 4x4 numpy array (camera to world transform)
            - camera_intrinsics: 3x3 numpy array (intrinsic matrix)
            - width: Image width in pixels
            - height: Image height in pixels
        """
        return self._load_ego_pose(frame_idx)

    def get_dynamic_objects(self, frame_idx: int) -> List[Dict[str, Any]]:
        """
        Get dynamic objects data for a frame.

        Args:
            frame_idx: Frame index (0-based)

        Returns:
            List of dynamic object dictionaries, each containing:
            - object_id: Object identifier
            - pose_world: 4x4 numpy array (object to world transform)
            - dimensions: [width, height, depth] list

        Note:
            Returns empty list if dynamic_objects file not found.
        """
        objects_path = os.path.join(
            str(self._scene_path),
            "dynamic_objects",
            f"frame_{frame_idx:04d}_objects.json"
        )

        if not os.path.exists(objects_path):
            logger.debug(f"Dynamic objects file not found: {objects_path}")
            return []

        with open(objects_path, "r") as f:
            objects_list = json.load(f)

        # Convert pose_world to numpy arrays
        for obj in objects_list:
            if "pose_world" in obj:
                obj["pose_world"] = np.array(obj["pose_world"])

        logger.debug(f"Loaded {len(objects_list)} dynamic objects for frame {frame_idx}")
        return objects_list

    def add_ego(
        self,
        frame_idx: int = 0,
        ego_bp: str = EGO_LABEL,
        enable_physics: bool = False,
        move_spectator: bool = True,
    ) -> carla.Actor:
        """
        Spawn ego vehicle in CARLA world from DGGT scene data.

        Adapted from NuRec NurecScenario.add_ego() (nurec_integration.py:560-607).

        Args:
            frame_idx: Frame index for spawn pose (default 0)
            ego_bp: CARLA blueprint name (default EGO_LABEL from constants)
            enable_physics: Whether to enable physics simulation
            move_spectator: Whether to move spectator camera to follow ego

        Returns:
            carla.Actor: The spawned ego vehicle

        Raises:
            RuntimeError: If scenario not loaded or CARLA world not available
        """
        if self._carla_world is None:
            raise RuntimeError("CARLA world not set. Initialize with carla_world parameter.")

        world = self._carla_world
        bp_library = world.get_blueprint_library()

        # Get ego blueprint
        ego_bp_obj = bp_library.find(ego_bp)
        if ego_bp_obj is None:
            logger.warning(f"Blueprint {ego_bp} not found, using default")
            ego_bp_obj = bp_library.find(EGO_LABEL)

        # Get ego pose from DGGT scene
        ego_extrinsics, _, _, _ = self.get_ego_pose(frame_idx)

        # Transform to CARLA coordinates
        # ego_extrinsics is in DGGT world frame, apply _t_carla_dggt to get CARLA world frame
        carla_ego_pose_mat = self._t_carla_dggt @ ego_extrinsics
        carla_ego_pose = mat_to_carla_transform(carla_ego_pose_mat)

        # Try to spawn at calculated position
        ego_instance = world.try_spawn_actor(ego_bp_obj, carla_ego_pose)

        # Fallback: spawn at arbitrary location if spawn fails
        if ego_instance is None:
            logger.debug(
                f"Failed to spawn ego at {carla_ego_pose}. Spawning at arbitrary location."
            )
            carla_ego_pose = carla.Transform(
                carla.Location(x=0, y=0, z=1000),
                carla.Rotation(pitch=0, yaw=0, roll=0)
            )
            ego_instance = world.spawn_actor(ego_bp_obj, carla_ego_pose)

        # Build ego track
        ego_track = build_ego_track(
            str(self._scene_path),
            self._scene_metadata.num_frames,
            fps=10.0,  # DGGT default
        )

        # Apply transform to track
        ego_track.set_transform(self._t_carla_dggt)
        ego_track.set_ignore_out_of_bounds(True)

        # Create DggtActor wrapper
        self.actor_mapping[EGO_TRACK_ID] = DggtActor(
            ego_instance,
            ego_track,
            physics=enable_physics,
            blueprint_id=ego_bp,
        )

        # Store blueprint mapping
        self.actor_blueprints[ego_instance.id] = ego_bp

        # Track actors needing physics disabled
        if not enable_physics:
            self.actors_to_disable_physics.append(self.actor_mapping[EGO_TRACK_ID])

        # Move spectator to follow ego
        if move_spectator:
            spectator = world.get_spectator()
            spectator_transform = np.eye(4)
            spectator_transform[:3, 3] = [-5, 0, 3]  # 5m behind, 3m up
            ego_transform = carla_ego_pose_mat @ spectator_transform
            spectator.set_transform(mat_to_carla_transform(ego_transform))

        logger.info(f"Spawned ego vehicle: {ego_bp} (id={ego_instance.id})")
        return ego_instance

    def _add_actors(self, tracks: List[DggtTrack]) -> List[DggtActor]:
        """
        Spawn dynamic actors from active tracks.

        Adapted from NuRec NurecScenario._add_actors() (nurec_integration.py:609-650).

        Args:
            tracks: List of DggtTrack objects to spawn as actors

        Returns:
            List of newly spawned DggtActor objects
        """
        if self._carla_world is None:
            logger.warning("CARLA world not set, cannot spawn actors")
            return []

        # Initialize blueprint library if needed
        if self._blueprint_library is None:
            dggt_project_dir = os.path.dirname(__file__)
            dggt_project_dir = os.path.join(dggt_project_dir, "dggt_project")
            self._blueprint_library = BlueprintLibrary(data_dir=dggt_project_dir)

        world = self._carla_world
        bp_library = world.get_blueprint_library()
        new_actors = []

        for track in tracks:
            # Filter by label - only spawn vehicles and pedestrians
            if not (track.label in VEHICLE_LABELS or track.label == "person"):
                continue

            # Skip non-dynamic objects
            if not track.is_dynamic():
                continue

            # Get best-fit blueprint
            is_vehicle = track.label != "person"
            best_fit_blueprint = self._blueprint_library.get_best_fit_blueprint(
                track.dims, vehicle=is_vehicle
            )
            actor_bp = bp_library.find(best_fit_blueprint.id)
            if actor_bp is None:
                logger.warning(f"Blueprint {best_fit_blueprint.id} not found")
                continue
            blueprint_id = best_fit_blueprint.id

            # Get spawn pose
            spawn_pose_array = track.interpolate_pose_matrix(track.start_time())
            if spawn_pose_array is None:
                logger.warning(f"Could not interpolate spawn pose for track {track.track_id}")
                continue

            # Apply blueprint offset (rear axle correction)
            spawn_pose_matrix = self._blueprint_library.apply_offset_to_pose(
                spawn_pose_array, blueprint_id, inverse=True
            )

            # Transform to CARLA coordinates
            carla_spawn_pose_mat = self._t_carla_dggt @ spawn_pose_matrix
            spawn_pose = mat_to_carla_transform(carla_spawn_pose_mat)

            # Try spawn with retry logic
            actor_inst = world.try_spawn_actor(actor_bp, spawn_pose)
            if actor_inst is None:
                # Retry 10 meters higher
                spawn_pose.location.z += 10
                actor_inst = world.try_spawn_actor(actor_bp, spawn_pose)
                if actor_inst is None:
                    logger.warning(
                        f"Failed to spawn actor {track.track_id} ({blueprint_id})"
                    )
                    continue

            # Set initial transform
            actor_inst.set_transform(mat_to_carla_transform(carla_spawn_pose_mat))

            # Apply transform to track
            track.set_transform(self._t_carla_dggt)

            # Create DggtActor wrapper
            dggt_actor = DggtActor(
                actor_inst,
                track,
                physics=False,
                blueprint_id=blueprint_id,
                object_id=track.object_id if hasattr(track, 'object_id') else None,
            )
            new_actors.append(dggt_actor)

            # Store mappings
            self.actor_mapping[track.track_id] = dggt_actor
            self.actor_blueprints[actor_inst.id] = actor_bp.id
            self.actors_to_disable_physics.append(dggt_actor)

            logger.debug(f"Spawned actor: {track.track_id} -> {blueprint_id}")

        logger.info(f"Spawned {len(new_actors)} actors from {len(tracks)} tracks")
        return new_actors

    def get_blueprint_library(self) -> BlueprintLibrary:
        """Get or initialize the blueprint library."""
        if self._blueprint_library is None:
            dggt_project_dir = os.path.dirname(__file__)
            dggt_project_dir = os.path.join(dggt_project_dir, "dggt_project")
            self._blueprint_library = BlueprintLibrary(data_dir=dggt_project_dir)
        return self._blueprint_library


class DggtSensor:
    """
    DGGT Camera Sensor

    Registers CARLA tick callback and triggers rendering.

    Supports two modes:
    1. Scene Replay Mode: Loads ego_pose per frame from DggtScenario
       - Uses raw camera_extrinsics_world from scene files
       - No coordinate transforms applied (pose is already in DGGT coords)
    2. CARLA Mode: Uses preset_pose or parent_actor
       - Applies undo_carla_coordinate_transform + t_carla_dggt transform chain

    Transform Chain (CARLA mode - matches NurecSensor.on_world_tick exactly):
    - camera_pose = undo_carla_coordinate_transform(actor_transform) @ sensor_offset
    - Renderer.render applies t_carla_dggt @ camera_pose
    """

    def __init__(
        self,
        renderer: DggtRenderer,
        camera_id: str,
        resolution_ratio: float = 1.0,
        output_dir: Optional[str] = None,
        parent_actor: Optional[Any] = None,
        sensor_offset: Optional[np.ndarray] = None,
        preset_pose: Optional[np.ndarray] = None,
        callback: Optional[Any] = None,
        framerate: float = 10.0,
        scenario: Optional[DggtScenario] = None,
    ):
        """
        Initialize DGGT sensor

        Args:
            renderer: DGGT renderer instance
            camera_id: Camera logical ID
            resolution_ratio: Resolution scaling factor
            output_dir: Image output directory (optional)
            parent_actor: Parent CARLA actor (for mounted camera)
            sensor_offset: Sensor offset relative to parent actor (4x4 matrix)
            preset_pose: Preset pose for free camera mode (4x4 matrix)
            callback: Custom callback function for rendered images
            framerate: Target rendering framerate
            scenario: DggtScenario for scene replay mode (loads ego_pose per frame)

        Raises:
            ValueError: Camera not found
        """
        self._renderer = renderer
        self._camera_id = camera_id
        self._resolution_ratio = resolution_ratio
        self._output_dir = output_dir
        self._callback = callback
        self._framerate = framerate

        # Camera spec
        available_cameras = renderer.get_available_cameras()
        if camera_id not in available_cameras:
            raise ValueError(f"Camera '{camera_id}' not found. Available: {list(available_cameras.keys())}")
        self._camera_spec = available_cameras[camera_id]

        # Scene replay mode (uses ego_pose from DggtScenario)
        self._scenario = scenario

        # Parent actor mode (mounted on vehicle)
        self._parent_actor = parent_actor
        self._sensor_offset = sensor_offset if sensor_offset is not None else np.eye(4)

        # Free camera mode (preset pose)
        self._preset_pose = preset_pose

        # Callback ID
        self._callback_id: Optional[int] = None

        # Image callback for pygame display (Phase 4.2)
        self._image_callback: Optional[Callable] = None

        # Frame tracking
        self._frame_count = 0
        self._frame_index = 0  # Frame index for scene replay mode
        self._last_timestamp: float = 0.0
        self._max_frames: Optional[int] = None  # Max frames for scene replay mode

        # Get max frames from scenario if available
        if scenario is not None:
            try:
                metadata = scenario.get_scene_metadata()
                self._max_frames = metadata.num_frames
            except Exception:
                pass

        logger.info(f"DggtSensor initialized: {camera_id}, scenario_mode={scenario is not None}")

    def set_image_callback(self, callback: Callable) -> None:
        """
        Set callback function for rendered images (Phase 4.2).

        Args:
            callback: Callback function that receives image array (np.ndarray)
        """
        self._image_callback = callback

    def register_tick_callback(self, world: Any) -> None:
        """
        Register CARLA tick callback

        Args:
            world: CARLA world instance
        """
        self._callback_id = world.on_tick(self.on_tick)
        logger.info(f"Registered tick callback: {self._callback_id}")

    def unregister_tick_callback(self, world: Any) -> None:
        """
        Unregister CARLA tick callback

        Args:
            world: CARLA world instance
        """
        if self._callback_id is not None:
            world.remove_on_tick(self._callback_id)
            self._callback_id = None
            logger.info("Unregistered tick callback")

    def _should_render(self, timestamp: float) -> bool:
        """Check if should render at this timestamp (framerate control)"""
        if timestamp - self._last_timestamp < 1.0 / self._framerate:
            return False
        self._last_timestamp = timestamp
        return True

    def on_tick(self, world_snapshot: Any) -> None:
        """
        CARLA tick callback

        Renders image using DGGT renderer.

        Supports three modes:
        1. Scene Replay Mode (scenario provided): Loads ego_pose per frame from DggtScenario
           - Uses raw camera_extrinsics_world (already in DGGT coords)
           - NO coordinate transforms applied
        2. Mounted Camera Mode (parent_actor provided): Follows NuRec pattern
           - camera_pose = undo_carla_coordinate_transform(actor_transform) @ sensor_offset
        3. Free Camera Mode (preset_pose provided): Static pose
           - camera_pose = undo_carla_coordinate_transform(preset_pose)

        Args:
            world_snapshot: CARLA world snapshot
        """
        # Get timestamp
        timestamp_s = world_snapshot.timestamp.elapsed_seconds
        timestamp_us = int(timestamp_s * 1_000_000)

        # Framerate control
        if not self._should_render(timestamp_s):
            return

        # Step 1: Compute camera pose
        if self._scenario is not None:
            # Scene Replay Mode - load ego_pose per frame
            # IMPORTANT: camera_extrinsics_world is ALREADY in DGGT coordinate system
            # Do NOT apply undo_carla_coordinate_transform or t_carla_dggt

            # Check frame bounds to prevent out-of-range access during cleanup
            if self._max_frames is not None and self._frame_index >= self._max_frames:
                logger.debug(f"Frame index {self._frame_index} >= max frames {self._max_frames}, skipping render")
                return

            try:
                ego_extrinsics, ego_intrinsics, ego_width, ego_height = self._scenario.get_ego_pose(self._frame_index)
                # Pass raw pose directly - no transforms needed for scene replay
                camera_pose = ego_extrinsics
                use_raw_pose = True  # Flag to bypass coordinate transforms in renderer
                # Load dynamic objects for this frame
                dynamic_objects = self._scenario.get_dynamic_objects(self._frame_index)

                # CRITICAL FIX: Use frame_index-based timestamp for scene replay mode
                # The server computes frame_idx from timestamp using DGGT's 10Hz framerate.
                # We must send a timestamp that maps to the correct frame_idx.
                # DGGT framerate: 10Hz -> FRAME_DURATION_US = 100000us
                DGGT_FRAME_DURATION_US = 100000  # 1/10 second = 100000 microseconds
                timestamp_us = self._frame_index * DGGT_FRAME_DURATION_US
                logger.debug(f"[DGGT_SENSOR] Scene replay frame {self._frame_index}: using timestamp {timestamp_us}us")

            except FileNotFoundError as e:
                logger.warning(f"Ego pose not found for frame {self._frame_index}: {e}")
                return
            except Exception as e:
                logger.error(f"Failed to load ego pose for frame {self._frame_index}: {e}")
                return
        elif self._parent_actor is not None:
            # Mounted camera mode
            actor = world_snapshot.find(self._parent_actor.id)
            if actor is None:
                logger.warning(f"Parent actor {self._parent_actor.id} not found in snapshot")
                return

            # Get actor transform matrix (CARLA 4x4)
            actor_transform = np.array(actor.get_transform().get_matrix()).reshape(4, 4)

            # Apply undo + sensor_offset (NuRec pattern)
            camera_pose = undo_carla_coordinate_transform(actor_transform) @ self._sensor_offset
            use_raw_pose = False
            dynamic_objects = None
        else:
            # Free camera mode
            if self._preset_pose is None:
                logger.warning("No preset pose for free camera mode")
                return
            camera_pose = undo_carla_coordinate_transform(self._preset_pose)
            use_raw_pose = False
            dynamic_objects = None

        # Step 2: Render
        try:
            image = self._renderer.render(
                world_snapshot=world_snapshot,
                camera_spec=self._camera_spec,
                camera_pose=camera_pose,
                timestamp=timestamp_us,
                resolution_ratio=self._resolution_ratio,
                use_raw_pose=use_raw_pose,
                dynamic_objects=dynamic_objects,
            )

            # Save or callback
            if self._output_dir:
                self._save_image(image)
            if self._callback:
                self._callback(image)
            if self._image_callback is not None:
                self._image_callback(image)

            self._frame_count += 1

            # Increment frame index for scene replay mode
            if self._scenario is not None:
                self._frame_index += 1

        except Exception as e:
            logger.error(f"Render failed: {e}")

    def _save_image(self, image: np.ndarray) -> None:
        """
        Save image to output directory

        Args:
            image: RGB image array [H, W, 3]
        """
        import os
        try:
            import imageio
        except ImportError:
            from PIL import Image
            imageio = None

        os.makedirs(self._output_dir, exist_ok=True)
        output_path = os.path.join(self._output_dir, f"frame_{self._frame_count:06d}.jpg")

        if imageio:
            imageio.imwrite(output_path, image)
        else:
            # PIL fallback
            Image.fromarray(image).save(output_path, quality=95)

        logger.debug(f"Saved image: {output_path}")

    def get_frame_count(self) -> int:
        """Get total frames rendered"""
        return self._frame_count

    def get_camera_id(self) -> str:
        """Get camera ID"""
        return self._camera_id

    def get_frame_index(self) -> int:
        """Get current frame index (scene replay mode)"""
        return self._frame_index

    def set_frame_index(self, frame_idx: int) -> None:
        """Set frame index for scene replay mode"""
        self._frame_index = frame_idx