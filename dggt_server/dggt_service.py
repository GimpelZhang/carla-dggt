# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT gRPC Service Implementation

Implements SensorsimServiceServicer for DGGT scene rendering.

Uses Phase 1 components:
- DGGTSceneManager: Scene metadata caching and loading
- DGGTSceneMetadata: Scene metadata structure
- TrackIDMapping: track_id to object_id conversion

Uses Phase 2 components:
- RequestConverter: gRPC request conversion utilities
"""

import logging
import threading
import numpy as np
from typing import Optional, Dict, List

import grpc
from nre.grpc.protos import common_pb2 as common_pb
from nre.grpc.protos import sensorsim_pb2 as sensorsim_pb
from nre.grpc.protos.sensorsim_pb2_grpc import SensorsimServiceServicer

from .scene_manager import DGGTSceneManager
from .scene_metadata import DGGTSceneMetadata, TrackIDMapping
from .request_converter import RequestConverter
from .config import DGGTServerConfig
from .exceptions import SceneNotFoundError, RenderingError

# Optional import: DGGTRenderer requires torch/gsplat (only in dggt conda env)
try:
    from .dggt_renderer import DGGTRenderer
    _RENDERER_AVAILABLE = True
except ImportError:
    DGGTRenderer = None
    _RENDERER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Version info
DGGT_VERSION = "0.1.0"
DGGT_GIT_HASH = "phase3-dev"


class DGGTService(SensorsimServiceServicer):
    """
    DGGT gRPC Service Implementation

    Implements SensorsimService for DGGT Gaussian Splatting scene rendering.

    Key responsibilities:
    - Scene management via DGGTSceneManager
    - Request conversion via RequestConverter
    - Coordinate transformation via t_carla_dggt matrix
    - RGB rendering (LiDAR not implemented)
    """

    def __init__(
        self,
        scene_manager: DGGTSceneManager,
        config: DGGTServerConfig,
        t_carla_dggt: np.ndarray = None
    ):
        """
        Initialize DGGT Service

        Args:
            scene_manager: Phase 1 DGGTSceneManager for scene loading
            config: Server configuration
            t_carla_dggt: CARLA to DGGT coordinate transform (4x4 matrix)
                         If None, identity matrix is used
        """
        self.scene_manager = scene_manager
        self.config = config
        self.t_carla_dggt = t_carla_dggt if t_carla_dggt is not None else np.eye(4, dtype=np.float32)

        # Track ID mapping per scene
        self._track_mappings: Dict[str, TrackIDMapping] = {}

        # DGGT renderer instances (lazy init per scene)
        self._renderers: Dict[str, DGGTRenderer] = {}
        self._renderer_lock = threading.Lock()

        # Shutdown flag
        self._shutdown_requested = False
        self._shutdown_lock = threading.Lock()

        # Initialize track mappings for loaded scenes
        self._init_track_mappings()

        logger.info(f"DGGTService initialized with {len(scene_manager.list_scenes())} scenes")

    def _init_track_mappings(self) -> None:
        """Initialize TrackIDMapping for each loaded scene"""
        for scene_id in self.scene_manager.list_scenes():
            try:
                meta = self.scene_manager.get_scene(scene_id)
                mapping = TrackIDMapping(auto_prefix="dggt_obj")
                mapping.from_scene_objects(meta.dynamic_object_ids)
                self._track_mappings[scene_id] = mapping
                logger.debug(f"Initialized track mapping for scene {scene_id}")
            except Exception as e:
                logger.warning(f"Failed to init track mapping for {scene_id}: {e}")

    def _get_renderer(self, scene_id: str) -> DGGTRenderer:
        """
        Get or create DGGTRenderer for a scene (lazy initialization, thread-safe)

        Args:
            scene_id: Scene identifier

        Returns:
            DGGTRenderer instance for the scene

        Raises:
            RuntimeError: If DGGTRenderer is not available (torch/gsplat not installed)
        """
        if not _RENDERER_AVAILABLE:
            raise RuntimeError("DGGTRenderer not available: torch/gsplat not installed")
        with self._renderer_lock:
            if scene_id not in self._renderers:
                meta = self.scene_manager.get_scene(scene_id)
                scene_path = meta.scene_path
                device = getattr(self.config, 'device', 'cuda')
                self._renderers[scene_id] = DGGTRenderer(scene_path, device=device)
                logger.info(f"Created DGGTRenderer for scene {scene_id}")
            return self._renderers[scene_id]

    def get_version(self, request: common_pb.Empty, context) -> common_pb.VersionId:
        """
        Return service version information

        Args:
            request: Empty request
            context: gRPC context

        Returns:
            VersionId with version string and git hash
        """
        logger.debug("get_version called")

        return common_pb.VersionId(
            version_id=f"DGGT-{DGGT_VERSION}",
            git_hash=DGGT_GIT_HASH,
            grpc_api_version=common_pb.VersionId.APIVersion(
                major=1,
                minor=0,
                patch=0
            )
        )

    def get_available_scenes(self, request: common_pb.Empty, context) -> common_pb.AvailableScenesReturn:
        """
        Return list of available scene IDs

        Args:
            request: Empty request
            context: gRPC context

        Returns:
            AvailableScenesReturn with scene_ids list
        """
        logger.debug("get_available_scenes called")

        scene_ids = self.scene_manager.list_scenes()

        return common_pb.AvailableScenesReturn(
            scene_ids=scene_ids
        )

    def get_available_cameras(
        self,
        request: sensorsim_pb.AvailableCamerasRequest,
        context
    ) -> sensorsim_pb.AvailableCamerasReturn:
        """
        Return available camera configurations for a scene

        Args:
            request: AvailableCamerasRequest with scene_id
            context: gRPC context

        Returns:
            AvailableCamerasReturn with camera list

        Note:
            DGGT scenes have a single ego camera (logical_id: "ego_camera")
        """
        scene_id = request.scene_id
        logger.debug(f"get_available_cameras called for scene: {scene_id}")

        try:
            meta = self.scene_manager.get_scene(scene_id)
        except SceneNotFoundError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Scene not found: {scene_id}")
            return sensorsim_pb.AvailableCamerasReturn()

        # Build camera info from scene metadata
        # DGGT has single ego camera with fixed intrinsics (or varying per frame)
        # Get intrinsic matrix from scene metadata
        K = meta.intrinsic_matrix

        camera = sensorsim_pb.AvailableCamerasReturn.AvailableCamera(
            logical_id="ego_camera",
            trajectory_idx=0,
            intrinsics=sensorsim_pb.CameraSpec(
                logical_id="ego_camera",
                trajectory_idx=0,
                resolution_w=meta.camera_width,
                resolution_h=meta.camera_height,
                shutter_type=sensorsim_pb.ShutterType.GLOBAL,
                # Use opencv_fisheye_param to store intrinsics (OpenCVPinholeCameraParam is empty in proto)
                opencv_fisheye_param=sensorsim_pb.OpenCVFisheyeCameraParam(
                    focal_length_x=float(K[0, 0]),
                    focal_length_y=float(K[1, 1]),
                    principal_point_x=float(K[0, 2]),
                    principal_point_y=float(K[1, 2])
                )
            ),
            # Identity rig-to-camera transform (camera is at ego pose)
            rig_to_camera=common_pb.Pose(
                vec=common_pb.Vec3(x=0.0, y=0.0, z=0.0),
                quat=common_pb.Quat(x=0.0, y=0.0, z=0.0, w=1.0)
            )
        )

        return sensorsim_pb.AvailableCamerasReturn(
            available_cameras=[camera]
        )

    def render_rgb(
        self,
        request: sensorsim_pb.RGBRenderRequest,
        context
    ) -> sensorsim_pb.RGBRenderReturn:
        """
        Render RGB image from DGGT scene

        Full rendering flow:
        1. Get scene metadata
        2. Compute frame index from timestamps (use midpoint)
        3. Get frame metadata (c2w matrix, intrinsics)
        4. Convert sensor pose using coordinate transform
        5. Convert dynamic objects
        6. Render Gaussian splatting scene (DGGTRenderer)
        7. Encode image to requested format

        Args:
            request: RGBRenderRequest with scene_id, camera spec, poses, etc.
            context: gRPC context

        Returns:
            RGBRenderReturn with encoded image bytes
        """
        scene_id = request.scene_id
        logger.info(f"render_rgb called for scene: {scene_id}")

        # Check shutdown status
        with self._shutdown_lock:
            if self._shutdown_requested:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Service is shutting down")
                return sensorsim_pb.RGBRenderReturn()

        # 1. Get scene metadata
        try:
            scene_meta = self.scene_manager.get_scene(scene_id)
        except SceneNotFoundError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Scene not found: {scene_id}")
            logger.error(f"Scene not found: {scene_id}")
            return sensorsim_pb.RGBRenderReturn()

        # 2. Compute frame index from timestamp range (use midpoint)
        midpoint_us = (request.frame_start_us + request.frame_end_us) // 2
        frame_idx = scene_meta.get_frame_index_from_timestamp(midpoint_us)
        logger.debug(f"Frame index {frame_idx} from timestamps {request.frame_start_us}-{request.frame_end_us}")

        # 3. Get frame metadata
        try:
            frame_meta = self.scene_manager.get_frame_metadata(scene_id, frame_idx)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to load frame metadata: {e}")
            logger.error(f"Failed to load frame {frame_idx}: {e}")
            return sensorsim_pb.RGBRenderReturn()

        # 4. Convert camera intrinsics from request
        try:
            K, width, height = RequestConverter.convert_camera_intrinsics(request.camera_intrinsics)
        except Exception as e:
            # Use scene default intrinsics if request conversion fails
            logger.warning(f"Using scene default intrinsics: {e}")
            K = frame_meta.intrinsic_matrix
            width = request.resolution_w if request.resolution_w > 0 else scene_meta.camera_width
            height = request.resolution_h if request.resolution_h > 0 else scene_meta.camera_height

        # 5. Interpolate sensor pose
        # Use alpha=0.5 for global shutter (midpoint between start and end)
        sensor_pose = RequestConverter.interpolate_pose_pair(request.sensor_pose, alpha=0.5)

        # Get source_is_dggt from config (default True for DGGT scene data)
        source_is_dggt = getattr(self.config, 'source_is_dggt', True)

        # Apply coordinate transform chain based on source
        # dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)
        if source_is_dggt:
            # Data already in DGGT coordinates - no transform needed
            sensor_pose_dggt = sensor_pose
        else:
            # Data from CARLA - apply full transform chain
            try:
                from ..utils import undo_carla_coordinate_transform
            except ImportError:
                # Fallback for direct imports
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from utils import undo_carla_coordinate_transform
            sensor_pose_dggt = self.t_carla_dggt @ undo_carla_coordinate_transform(sensor_pose)

        # 6. Convert dynamic objects
        track_mapping = self._track_mappings.get(scene_id)
        if track_mapping is None:
            track_mapping = TrackIDMapping(auto_prefix="dggt_obj")
            track_mapping.from_scene_objects(scene_meta.dynamic_object_ids)

        dynamic_objects_dggt = RequestConverter.convert_dynamic_objects(
            request.dynamic_objects,
            track_mapping,
            self.t_carla_dggt,
            alpha=0.5,
            source_is_dggt=source_is_dggt
        )
        logger.info(f"[DGGT_SERVICE] Converted {len(dynamic_objects_dggt)} dynamic objects (source_is_dggt={source_is_dggt})")

        # DEBUG: Log the converted dynamic object poses
        for obj_id, pose in dynamic_objects_dggt:
            logger.info(f"[DGGT_SERVICE] object_id={obj_id}, pose_dggt position: [{pose[0,3]:.6f}, {pose[1,3]:.6f}, {pose[2,3]:.6f}]")

        # 7. Render with DGGT Gaussian Splatting renderer
        # Build object pose overrides from converted dynamic objects
        # convert_dynamic_objects returns List[Tuple[int, np.ndarray]]
        object_pose_overrides = {}
        for obj_id, pose in dynamic_objects_dggt:
            if obj_id >= 0 and pose is not None:
                object_pose_overrides[obj_id] = pose

        try:
            renderer = self._get_renderer(scene_id)
            rendered_image = renderer.render_frame(
                frame_idx=frame_idx,
                camera_pose_override=sensor_pose_dggt,
                object_pose_overrides=object_pose_overrides if object_pose_overrides else None,
                use_scene_defaults=True,
                intrinsics_override=K,
                width_override=width,
                height_override=height
            )
        except Exception as e:
            logger.error(f"DGGT rendering failed for scene {scene_id}, frame {frame_idx}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Rendering failed: {e}")
            return sensorsim_pb.RGBRenderReturn()

        # 8. Encode image
        image_format = request.image_format if request.image_format != sensorsim_pb.ImageFormat.UNDEFINED else sensorsim_pb.ImageFormat.JPEG
        quality = request.image_quality if request.image_quality > 0 else self.config.default_image_quality

        try:
            encoded_bytes = RequestConverter.encode_image(rendered_image, image_format, quality)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to encode image: {e}")
            logger.error(f"Image encoding failed: {e}")
            return sensorsim_pb.RGBRenderReturn()

        logger.info(f"render_rgb completed: {len(encoded_bytes)} bytes, format={image_format}")

        return sensorsim_pb.RGBRenderReturn(
            image_bytes=encoded_bytes
        )

    def _generate_placeholder_image(self, width: int, height: int, frame_idx: int) -> np.ndarray:
        """
        Generate placeholder image for testing

        Note: render_rgb now uses DGGTRenderer for actual rendering.
        This method is retained for fallback/testing purposes only.

        Args:
            width: Image width
            height: Image height
            frame_idx: Frame index (used for pattern)

        Returns:
            RGB image array (H, W, 3) uint8
        """
        # Create a gradient pattern that varies with frame_idx
        # This helps verify that different frames produce different images
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Horizontal gradient with frame-dependent offset
        for x in range(width):
            intensity = int(255 * (x + frame_idx * 10) / width) % 256
            image[:, x, 0] = intensity  # Red channel

        # Vertical gradient
        for y in range(height):
            intensity = int(255 * y / height)
            image[y, :, 1] = intensity  # Green channel

        # Blue channel: frame index indicator
        image[:, :, 2] = (frame_idx * 10) % 256

        # Add text overlay (frame number) - simple block pattern
        # Top-left corner shows frame index as binary pattern
        for bit in range(min(8, frame_idx.bit_length() if frame_idx > 0 else 1)):
            if frame_idx & (1 << bit):
                image[10:30, 10 + bit * 20: 10 + bit * 20 + 15, :] = 255

        return image

    def render_lidar(
        self,
        request: sensorsim_pb.LidarRenderRequest,
        context
    ) -> sensorsim_pb.LidarRenderReturn:
        """
        Render LiDAR point cloud (NOT IMPLEMENTED)

        Args:
            request: LidarRenderRequest
            context: gRPC context

        Returns:
            LidarRenderReturn (raises UNIMPLEMENTED)
        """
        logger.warning("render_lidar called - NOT IMPLEMENTED")

        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("LiDAR rendering not implemented for DGGT")
        return sensorsim_pb.LidarRenderReturn()

    def shut_down(self, request: common_pb.Empty, context) -> common_pb.Empty:
        """
        Graceful shutdown request

        Args:
            request: Empty request
            context: gRPC context

        Returns:
            Empty response
        """
        logger.info("shut_down requested")

        with self._shutdown_lock:
            self._shutdown_requested = True

        # Clear scene cache
        self.scene_manager.clear_cache()

        logger.info("DGGTService shutdown initiated")

        return common_pb.Empty()

    def get_available_trajectories(
        self,
        request: sensorsim_pb.AvailableTrajectoriesRequest,
        context
    ) -> sensorsim_pb.AvailableTrajectoriesReturn:
        """
        Return available trajectories for a scene (DGGT has single ego trajectory)

        Args:
            request: AvailableTrajectoriesRequest with scene_id
            context: gRPC context

        Returns:
            AvailableTrajectoriesReturn with trajectory list
        """
        scene_id = request.scene_id
        logger.debug(f"get_available_trajectories called for scene: {scene_id}")

        try:
            meta = self.scene_manager.get_scene(scene_id)
        except SceneNotFoundError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Scene not found: {scene_id}")
            return sensorsim_pb.AvailableTrajectoriesReturn()

        # DGGT has single ego trajectory
        # Build trajectory from frame timestamps and poses
        poses = []
        for idx in range(min(meta.num_frames, 10)):  # Sample first 10 frames
            try:
                frame = self.scene_manager.get_frame_metadata(scene_id, idx)
                # Convert c2w to Pose
                pos = frame.c2w_matrix[:3, 3]
                from scipy.spatial.transform import Rotation as R
                quat = R.from_matrix(frame.c2w_matrix[:3, :3]).as_quat()

                poses.append(common_pb.PoseAtTime(
                    timestamp_us=frame.timestamp_us,
                    pose=common_pb.Pose(
                        vec=common_pb.Vec3(x=pos[0], y=pos[1], z=pos[2]),
                        quat=common_pb.Quat(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
                    )
                ))
            except Exception as e:
                logger.warning(f"Failed to get pose for frame {idx}: {e}")

        trajectory = sensorsim_pb.AvailableTrajectoriesReturn.AvailableTrajectory(
            trajectory_idx=0,
            trajectory=common_pb.Trajectory(poses=poses)
        )

        return sensorsim_pb.AvailableTrajectoriesReturn(
            available_trajectories=[trajectory]
        )

    def get_available_ego_masks(
        self,
        request: common_pb.Empty,
        context
    ) -> sensorsim_pb.AvailableEgoMasksReturn:
        """
        Return available ego masks (DGGT does not support ego masks)

        Args:
            request: Empty request
            context: gRPC context

        Returns:
            AvailableEgoMasksReturn (empty list)
        """
        logger.debug("get_available_ego_masks called - DGGT has no ego masks")

        return sensorsim_pb.AvailableEgoMasksReturn(
            ego_mask_metadata=[]
        )
