# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Request Converter

Converts NuRec gRPC request objects to DGGT rendering parameters.

CRITICAL: This module imports undo_carla_coordinate_transform from utils.py.
Never reimplement coordinate transform functions locally.

Transform chain: dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)
Reference: utils.py line 146
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from nre.grpc.protos import sensorsim_pb2
from nre.grpc.protos import common_pb2

# CRITICAL: Import coordinate transform from utils.py, NEVER reimplement!
# Path: /mnt/E/carla/carla/PythonAPI/examples/nvidia/nurec/utils.py (lines 98-116)
try:
    from ..utils import undo_carla_coordinate_transform
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import undo_carla_coordinate_transform

from .scene_metadata import TrackIDMapping
from .exceptions import InvalidCameraSpecError, TrackIDMappingError

logger = logging.getLogger(__name__)


class RequestConverter:
    """
    NuRec protobuf to DGGT rendering parameter converter

    Coordinate transform:
    - Uses utils.py undo_carla_coordinate_transform() for CARLA coordinate correction
    - Transform chain: dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)
    - Matches utils.py actor_to_grpc_pose() pattern (line 146)
    """

    @staticmethod
    def interpolate_pose_pair(pose_pair: sensorsim_pb2.PosePair, alpha: float = 0.5) -> np.ndarray:
        """
        Interpolate PosePair using SLERP for rotation + linear for position

        Args:
            pose_pair: NuRec PosePair (start_pose, end_pose)
            alpha: Interpolation factor (0=start, 1=end), global shutter uses 0.5

        Returns:
            4x4 Camera-to-World transformation matrix (np.ndarray, float32)

        Note:
            Reference: track.py InterpolatedPoses.interpolate_pose_matrix()
            Uses scipy rotvec method for stable SLERP
        """
        # Extract positions
        start_pos = np.array([
            pose_pair.start_pose.vec.x,
            pose_pair.start_pose.vec.y,
            pose_pair.start_pose.vec.z
        ], dtype=np.float32)

        end_pos = np.array([
            pose_pair.end_pose.vec.x,
            pose_pair.end_pose.vec.y,
            pose_pair.end_pose.vec.z
        ], dtype=np.float32)

        # Extract quaternions
        # Protobuf Quat: w, x, y, z
        # scipy: x, y, z, w (need to reorder)
        start_q = np.array([
            pose_pair.start_pose.quat.x,
            pose_pair.start_pose.quat.y,
            pose_pair.start_pose.quat.z,
            pose_pair.start_pose.quat.w
        ], dtype=np.float32)

        end_q = np.array([
            pose_pair.end_pose.quat.x,
            pose_pair.end_pose.quat.y,
            pose_pair.end_pose.quat.z,
            pose_pair.end_pose.quat.w
        ], dtype=np.float32)

        # Linear interpolation for position
        interp_pos = start_pos + alpha * (end_pos - start_pos)

        # SLERP for rotation
        interp_q = RequestConverter._slerp_quaternion(start_q, end_q, alpha)

        # Build transformation matrix
        return RequestConverter._pose_to_matrix(interp_pos, interp_q)

    @staticmethod
    def _slerp_quaternion(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
        """
        Spherical linear interpolation for quaternions using scipy rotvec method

        Args:
            q1: Start quaternion [x, y, z, w] (scipy format)
            q2: End quaternion [x, y, z, w] (scipy format)
            alpha: Interpolation factor

        Returns:
            Interpolated quaternion [x, y, z, w]

        Note:
            Reference: track.py uses scipy rotvec for stable SLERP
        """
        start_rot = R.from_quat(q1)
        end_rot = R.from_quat(q2)

        # Use rotvec for smooth interpolation
        rotvec = (start_rot.inv() * end_rot).as_rotvec()
        interp_rot = start_rot * R.from_rotvec(rotvec * alpha)

        return interp_rot.as_quat().astype(np.float32)

    @staticmethod
    def _pose_to_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """
        Convert position + quaternion to 4x4 transformation matrix

        Args:
            position: [x, y, z]
            quaternion: [x, y, z, w] (scipy format)

        Returns:
            4x4 transformation matrix (np.ndarray, float32)
        """
        rotation = R.from_quat(quaternion)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rotation.as_matrix()
        T[:3, 3] = position
        return T

    @staticmethod
    def convert_dynamic_objects(
        dynamic_objects: List[sensorsim_pb2.DynamicObject],
        track_mapping: TrackIDMapping,
        t_carla_dggt: np.ndarray,
        alpha: float = 0.5,
        source_is_dggt: bool = False
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Convert NuRec DynamicObject list to DGGT format

        Coordinate transform chain depends on source_is_dggt flag:

        - If source_is_dggt=False (default): Data from CARLA
          dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)

        - If source_is_dggt=True: Data already in DGGT coordinates
          dggt_pose = pose_matrix (no transform needed)

        Reference: utils.py actor_to_grpc_pose() line 146

        Args:
            dynamic_objects: NuRec protobuf DynamicObject list
            track_mapping: Phase 1 TrackIDMapping for track_id to object_id conversion
            t_carla_dggt: CARLA to DGGT transformation matrix (4x4)
            alpha: Pose interpolation factor (global shutter uses 0.5)
            source_is_dggt: If True, poses are already in DGGT coordinates (skip CARLA transform)

        Returns:
            List of (object_id, 4x4_pose_matrix) tuples in DGGT coordinates
        """
        import logging
        logger = logging.getLogger(__name__)
        result = []

        for obj in dynamic_objects:
            try:
                # track_id to object_id using Phase 1 TrackIDMapping
                object_id = track_mapping.to_object_id(obj.track_id)

                # DEBUG: Log the incoming gRPC pose values
                logger.info(f"[REQUEST_CONVERTER] track_id={obj.track_id} -> object_id={object_id}")
                logger.info(f"[REQUEST_CONVERTER]   gRPC start_pose: pos=({obj.pose_pair.start_pose.vec.x:.6f}, {obj.pose_pair.start_pose.vec.y:.6f}, {obj.pose_pair.start_pose.vec.z:.6f})")
                logger.info(f"[REQUEST_CONVERTER]   gRPC start_pose: quat=({obj.pose_pair.start_pose.quat.w:.6f}, {obj.pose_pair.start_pose.quat.x:.6f}, {obj.pose_pair.start_pose.quat.y:.6f}, {obj.pose_pair.start_pose.quat.z:.6f})")

                # Interpolate pose
                pose_matrix = RequestConverter.interpolate_pose_pair(obj.pose_pair, alpha)
                logger.info(f"[REQUEST_CONVERTER]   interpolated pose_matrix position: [{pose_matrix[0,3]:.6f}, {pose_matrix[1,3]:.6f}, {pose_matrix[2,3]:.6f}]")

                # Apply coordinate transform chain based on source
                if source_is_dggt:
                    # Data already in DGGT coordinates - no transform needed
                    pose_dggt = pose_matrix
                    logger.info(f"[REQUEST_CONVERTER]   source_is_dggt=True, using pose as-is")
                else:
                    # Data from CARLA - apply full transform chain
                    # Reference: utils.py line 146
                    # dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)
                    pose_dggt = t_carla_dggt @ undo_carla_coordinate_transform(pose_matrix)
                    logger.info(f"[REQUEST_CONVERTER]   source_is_dggt=False, applied t_carla_dggt @ undo_carla_coordinate_transform")

                logger.info(f"[REQUEST_CONVERTER]   final pose_dggt position: [{pose_dggt[0,3]:.6f}, {pose_dggt[1,3]:.6f}, {pose_dggt[2,3]:.6f}]")
                result.append((object_id, pose_dggt))

            except ValueError as e:
                logger.warning(f"Skipping object with invalid track_id '{obj.track_id}': {e}")

        return result

    @staticmethod
    def convert_camera_intrinsics(
        camera_spec: sensorsim_pb2.CameraSpec
    ) -> Tuple[np.ndarray, int, int]:
        """
        Convert CameraSpec to intrinsic matrix K + width + height

        Supports OpenCV pinhole, fisheye, and f-theta camera models.

        Args:
            camera_spec: NuRec CameraSpec protobuf

        Returns:
            (K_matrix, width, height) tuple
            K_matrix: 3x3 intrinsic matrix (np.ndarray, float32)

        Raises:
            InvalidCameraSpecError: No recognized camera model
        """
        width = camera_spec.resolution_w
        height = camera_spec.resolution_h

        # Check which camera model is present
        # Note: CameraSpec uses oneof for camera_param

        if camera_spec.HasField("opencv_pinhole_param"):
            param = camera_spec.opencv_pinhole_param
            K = np.array([
                [param.focal_length_x, 0, param.principal_point_x],
                [0, param.focal_length_y, param.principal_point_y],
                [0, 0, 1]
            ], dtype=np.float32)

        elif camera_spec.HasField("opencv_fisheye_param"):
            # Fisheye simplified to pinhole approximation
            param = camera_spec.opencv_fisheye_param
            K = np.array([
                [param.focal_length_x, 0, param.principal_point_x],
                [0, param.focal_length_y, param.principal_point_y],
                [0, 0, 1]
            ], dtype=np.float32)

        elif camera_spec.HasField("ftheta_param"):
            # F-theta: estimate effective focal length
            param = camera_spec.ftheta_param
            focal = RequestConverter._estimate_ftheta_focal(param, width, height)
            K = np.array([
                [focal, 0, param.principal_point_x],
                [0, focal, param.principal_point_y],
                [0, 0, 1]
            ], dtype=np.float32)

        else:
            raise InvalidCameraSpecError(
                "camera_param", "none",
                "CameraSpec has no recognized camera model "
                "(opencv_pinhole_param, opencv_fisheye_param, or ftheta_param)"
            )

        return K, width, height

    @staticmethod
    def _estimate_ftheta_focal(
        param: sensorsim_pb2.FthetaCameraParam,
        width: int,
        height: int
    ) -> float:
        """
        Estimate effective focal length from f-theta parameters

        Args:
            param: FthetaCameraParam
            width: Image width
            height: Image height

        Returns:
            Estimated focal length (float)
        """
        # Use polynomial if available
        if param.pixeldist_to_angle_poly:
            # Approximate focal from first polynomial coefficient
            # For small angles: pixel_dist ~= focal * angle
            return width / (2.0 * param.max_angle) if param.max_angle > 0 else width / 2.0

        # Default: estimate from principal point (assume fov ~ 90 deg)
        return min(param.principal_point_x, param.principal_point_y)

    @staticmethod
    def encode_image(
        image: np.ndarray,
        format: sensorsim_pb2.ImageFormat,
        quality: float = 95.0
    ) -> bytes:
        """
        Encode image to specified format

        Args:
            image: RGB image array (H, W, 3) uint8
            format: ImageFormat enum (PNG=1, JPEG=2, JPEG2000=3, RGB_UINT8_PLANAR=4)
            quality: JPEG quality (0-100)

        Returns:
            Encoded image bytes

        Note:
            ImageFormat enum values:
            - UNDEFINED = 0
            - PNG = 1
            - JPEG = 2
            - JPEG2000 = 3
            - RGB_UINT8_PLANAR = 4
            - AVC = 5
            - AV1 = 6
        """
        import cv2

        # Convert RGB to BGR for OpenCV encoding (gsplat outputs RGB, cv2 expects BGR)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ImageFormat enum values
        if format == sensorsim_pb2.ImageFormat.PNG:  # value 1
            success, encoded = cv2.imencode(".png", image_bgr)
            if not success:
                raise ValueError("Failed to encode PNG image")
            return encoded.tobytes()

        elif format == sensorsim_pb2.ImageFormat.JPEG:  # value 2
            success, encoded = cv2.imencode(
                ".jpg", image_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
            )
            if not success:
                raise ValueError("Failed to encode JPEG image")
            return encoded.tobytes()

        elif format == sensorsim_pb2.ImageFormat.JPEG2000:  # value 3
            success, encoded = cv2.imencode(".jp2", image_bgr)
            if not success:
                raise ValueError("Failed to encode JPEG2000 image")
            return encoded.tobytes()

        elif format == sensorsim_pb2.ImageFormat.RGB_UINT8_PLANAR:  # value 4
            # Convert HWC to CHW (planar format)
            planar = np.transpose(image, (2, 0, 1))
            return planar.tobytes()

        else:
            raise ValueError(f"Unsupported image format: {format}")