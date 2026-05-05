# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Coordinate Transform Module

This module provides coordinate transformation utilities for converting between
CARLA and DGGT (OpenCV-style) coordinate systems.

Key Components:
- CoordinateTransform class: Handles CARLA ↔ DGGT coordinate conversions
- Rotation matrices: CARLA_TO_DGGT_ROTATION and DGGT_TO_CARLA_ROTATION

The module follows the same transformation chain pattern as NuRec:
    dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)

Coordinate System Definitions:
- CARLA: Left-handed, X=Forward, Y=Right, Z=Up
- DGGT: Right-handed (OpenCV), X=Right, Y=Down, Z=Forward
"""

from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import Optional, Tuple

# P0: Directly reuse NuRec's undo_carla_coordinate_transform for consistency
# Import from parent nurec directory
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import undo_carla_coordinate_transform


class CoordinateTransform:
    """
    CARLA ↔ DGGT Coordinate System Transformation Tool

    Handles conversions between CARLA left-handed coordinate system and
    DGGT (OpenCV) right-handed coordinate system.

    **Transformation Chain Design** (consistent with NuRec):
    - NuRec: `pose_nurec = t_carla_nurec @ undo_carla_coordinate_transform(carla_pose)`
    - DGGT: `pose_dggt = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)`

    **t_carla_dggt Calculation**:
    Uses `get_t_rig_enu_from_ecef()` from OpenDRIVE georeference, matching NuRec's
    `t_scenario_carla` calculation method exactly.
    """

    # CARLA → DGGT Rotation Matrix (right-handed coordinate transform)
    # Based on: CARLA (X=Forward, Y=Right, Z=Up) → DGGT (X=Right, Y=Down, Z=Forward)
    # X_dggt = Y_carla (Right), Y_dggt = -Z_carla (Down), Z_dggt = X_carla (Forward)
    #
    # Verification:
    # - CARLA Forward [1,0,0] → DGGT [0,0,1] (Forward) ✓
    # - CARLA Right [0,1,0] → DGGT [1,0,0] (Right) ✓
    # - CARLA Up [0,0,1] → DGGT [0,-1,0] (Down=-Up) ✓
    CARLA_TO_DGGT_ROTATION = np.array([
        [ 0,  1,  0],   # X_dggt = Y_carla (Right)
        [ 0,  0, -1],   # Y_dggt = -Z_carla (Down)
        [ 1,  0,  0]    # Z_dggt = X_carla (Forward)
    ], dtype=np.float64)

    # DGGT → CARLA Rotation Matrix (inverse transform = transpose, since orthogonal)
    # X_carla = Z_dggt (Forward), Y_carla = X_dggt (Right), Z_carla = -Y_dggt (Up)
    DGGT_TO_CARLA_ROTATION = np.array([
        [ 0,  0,  1],   # X_carla = Z_dggt (Forward)
        [ 1,  0,  0],   # Y_carla = X_dggt (Right)
        [ 0, -1,  0]    # Z_carla = -Y_dggt (Up)
    ], dtype=np.float64)

    @classmethod
    def transform_pose_carla_to_dggt(
        cls,
        pose_matrix: np.ndarray,
        t_carla_dggt: Optional[np.ndarray] = None,
        include_undo_carla: bool = True
    ) -> np.ndarray:
        """
        Transform CARLA pose matrix to DGGT coordinate system

        **Transformation Chain** (consistent with NuRec):
        ```
        dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)
        ```

        This matches NuRec's `actor_to_grpc_pose()` transformation chain exactly:
        ```
        nurec_pose = t_carla_nurec @ undo_carla_coordinate_transform(carla_pose)
        ```

        Args:
            pose_matrix: 4x4 CARLA pose matrix (Camera-to-World or Object-to-World)
            t_carla_dggt: Global CARLA→DGGT transformation matrix, computed by
                         `get_t_rig_enu_from_ecef()`. If None, uses basic rotation.
            include_undo_carla: Whether to first apply `undo_carla_coordinate_transform()`

        Returns:
            4x4 pose matrix in DGGT coordinate system
        """
        result = pose_matrix.copy()

        # Step 1: Undo CARLA special coordinate convention (using NuRec implementation)
        # P0: Reuse utils.py's undo_carla_coordinate_transform for exact NuRec consistency
        if include_undo_carla:
            result = undo_carla_coordinate_transform(result)

        # Step 2: Apply coordinate transformation
        # Standard multiplication chain consistent with NuRec: pose_dggt = T @ pose
        if t_carla_dggt is not None:
            # Use complete transformation matrix (from georeference calculation)
            result = t_carla_dggt @ result
        else:
            # Use basic rotation matrix
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = cls.CARLA_TO_DGGT_ROTATION
            result = transform @ result

        return result

    @classmethod
    def transform_pose_dggt_to_carla(
        cls,
        pose_matrix: np.ndarray,
        include_redo_carla: bool = True
    ) -> np.ndarray:
        """
        Transform DGGT pose matrix back to CARLA coordinate system

        Consistent with NuRec transformation chain: pose_carla = T_dggt_to_carla @ pose

        Args:
            pose_matrix: 4x4 DGGT pose matrix
            include_redo_carla: Whether to re-apply CARLA special coordinate transform

        Returns:
            4x4 pose matrix in CARLA coordinate system
        """
        # Step 1: Construct 4x4 transformation matrix
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = cls.DGGT_TO_CARLA_ROTATION

        # Step 2: Apply coordinate transformation (standard multiplication chain)
        result = transform @ pose_matrix

        # Step 3: Re-apply CARLA special coordinate transform (if needed)
        if include_redo_carla:
            result = cls.redo_carla_coordinate_transform(result)

        return result

    @staticmethod
    def redo_carla_coordinate_transform(transform: np.ndarray) -> np.ndarray:
        """
        Re-apply CARLA special coordinate transform (inverse of undo operation)

        This is the inverse of `undo_carla_coordinate_transform()` from utils.py.

        Args:
            transform: 4x4 transformation matrix after undo operation

        Returns:
            4x4 transformation matrix restored to CARLA coordinate convention
        """
        result = np.eye(4, dtype=np.float64)

        # Extract rotation matrix
        rotation = R.from_matrix(transform[:3, :3])

        # Convert to euler angles
        yaw, pitch, roll = rotation.as_euler("zyx", degrees=False)

        # Re-apply inverse yaw and inverse pitch
        yaw = -yaw
        pitch = -pitch

        rotation = R.from_euler("zyx", [yaw, pitch, roll], degrees=False)
        result[:3, :3] = rotation.as_matrix()

        # Y-axis mirror (position)
        result[:3, 3] = [transform[0, 3], -transform[1, 3], transform[2, 3]]

        return result

    @classmethod
    def transform_point_carla_to_dggt(
        cls,
        point: np.ndarray,
        include_undo_carla: bool = False
    ) -> np.ndarray:
        """
        Transform point coordinates (pure position vector, no rotation)

        Args:
            point: [x, y, z] CARLA coordinate point
            include_undo_carla: Whether to apply Y-axis mirror

        Returns:
            [x, y, z] DGGT coordinate point
        """
        if include_undo_carla:
            # Y-axis mirror
            point = np.array([point[0], -point[1], point[2]])

        return cls.CARLA_TO_DGGT_ROTATION @ point

    @classmethod
    def transform_rotation_carla_to_dggt(
        cls,
        rotation_matrix: np.ndarray,
        include_undo_carla: bool = True
    ) -> np.ndarray:
        """
        Transform rotation matrix (pure rotation, no position)

        Uses standard multiplication chain, consistent with NuRec.

        Args:
            rotation_matrix: 3x3 rotation matrix
            include_undo_carla: Whether to first undo CARLA special transform

        Returns:
            3x3 rotation matrix (DGGT coordinate system)
        """
        result = rotation_matrix.copy()

        if include_undo_carla:
            # Construct 4x4 matrix for undo
            # P0: Use imported undo_carla_coordinate_transform (from utils.py)
            pose = np.eye(4)
            pose[:3, :3] = rotation_matrix
            pose = undo_carla_coordinate_transform(pose)
            result = pose[:3, :3]

        # Standard multiplication chain: R_dggt = R_carla_to_dggt @ R_carla
        return cls.CARLA_TO_DGGT_ROTATION @ result

    @staticmethod
    def quaternion_carla_to_dggt(
        quat: np.ndarray,
        convention: str = "scipy"
    ) -> np.ndarray:
        """
        Transform quaternion from CARLA (left-handed) to DGGT (right-handed)

        Since CARLA_TO_DGGT_ROTATION has determinant -1 (reflection), we can't
        directly use matrix rotation transformation. Instead, we transform the
        quaternion components according to the axis mapping:
        - CARLA (X,Y,Z) -> DGGT (Z,X,-Y)
        - Quaternion [x,y,z,w] -> [z,x,-y,w]

        Args:
            quat: Quaternion array
            convention: "scipy" (x,y,z,w) or "protobuf" (w,x,y,z)

        Returns:
            Transformed quaternion in target coordinate system
        """
        if convention == "protobuf":
            # w,x,y,z -> x,y,z,w
            quat = np.array([quat[1], quat[2], quat[3], quat[0]])

        # Transform quaternion components based on axis mapping
        # CARLA: X=Forward, Y=Right, Z=Up (left-handed)
        # DGGT: X=Right, Y=Down, Z=Forward (right-handed)
        # Mapping: X_carla->Z_dggt, Y_carla->X_dggt, Z_carla->-Y_dggt
        # For quaternion: [qx,qy,qz,qw] -> [qz,qx,-qy,qw]
        #
        # Additionally, for LH->RH conversion, the rotation direction flips,
        # which is equivalent to negating the scalar part qw
        result_quat = np.array([quat[2], quat[0], -quat[1], -quat[3]])

        # Normalize to ensure unit quaternion
        result_quat = result_quat / np.linalg.norm(result_quat)

        if convention == "protobuf":
            # x,y,z,w -> w,x,y,z
            result_quat = np.array([result_quat[3], result_quat[0], result_quat[1], result_quat[2]])

        return result_quat