# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unit tests for Coordinate Transform module

Tests verify coordinate transformations between CARLA and DGGT coordinate systems.
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from dggt_server.coordinate_transform import CoordinateTransform
from utils import undo_carla_coordinate_transform


class TestCoordinateTransform:

    def test_carla_to_dggt_point_basic(self):
        """Test basic point transformation"""
        # CARLA forward point
        carla_forward = np.array([1, 0, 0])
        dggt_point = CoordinateTransform.transform_point_carla_to_dggt(carla_forward)

        # CARLA X(Forward) -> DGGT Z(Forward)
        # New matrix: Z_dggt = X_carla
        assert np.allclose(dggt_point, np.array([0, 0, 1]))

    def test_carla_to_dggt_point_full(self):
        """Test full point transformation with Y mirror"""
        carla_point = np.array([1, 2, 3])
        dggt_point = CoordinateTransform.transform_point_carla_to_dggt(
            carla_point, include_undo_carla=True
        )

        # After Y mirror: [1, -2, 3]
        # Matrix transform:
        # X_dggt = Y_carla = -2
        # Y_dggt = -Z_carla = -3
        # Z_dggt = X_carla = 1
        expected = np.array([-2, -3, 1])
        assert np.allclose(dggt_point, expected)

    def test_rotation_matrix_roundtrip(self):
        """Test rotation matrix roundtrip transformation"""
        # Random rotation
        original_rot = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()

        # CARLA -> DGGT -> CARLA
        dggt_rot = CoordinateTransform.transform_rotation_carla_to_dggt(original_rot)

        # Build pose and use inverse transform
        pose = np.eye(4)
        pose[:3, :3] = dggt_rot
        back_pose = CoordinateTransform.transform_pose_dggt_to_carla(pose, include_redo_carla=True)
        back_rot = back_pose[:3, :3]

        assert np.allclose(original_rot, back_rot, atol=1e-10)

    def test_pose_matrix_roundtrip(self):
        """Test pose matrix roundtrip transformation"""
        # Construct random pose
        rotation = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        translation = np.array([10, 20, 30])

        original_pose = np.eye(4)
        original_pose[:3, :3] = rotation
        original_pose[:3, 3] = translation

        # CARLA -> DGGT -> CARLA
        dggt_pose = CoordinateTransform.transform_pose_carla_to_dggt(original_pose)
        back_pose = CoordinateTransform.transform_pose_dggt_to_carla(dggt_pose)

        assert np.allclose(original_pose, back_pose, atol=1e-10)

    def test_undo_redo_carla_transform(self):
        """Test CARLA special transform undo and redo"""
        # P0: Use utils.py's undo_carla_coordinate_transform (consistent with NuRec)

        # Original pose
        rotation = R.from_euler('zyx', [45, 30, 0], degrees=True).as_matrix()
        translation = np.array([5, 10, 15])
        original = np.eye(4)
        original[:3, :3] = rotation
        original[:3, 3] = translation

        # undo -> redo (using CoordinateTransform's redo, since undo comes from utils)
        undone = undo_carla_coordinate_transform(original)
        redone = CoordinateTransform.redo_carla_coordinate_transform(undone)

        assert np.allclose(original, redone, atol=1e-10)

    def test_quaternion_transform(self):
        """Test quaternion transformation"""
        # Original quaternion (scipy format)
        original_quat = np.array([0.1, 0.2, 0.3, 0.9])
        original_quat = original_quat / np.linalg.norm(original_quat)

        # Transform to DGGT
        dggt_quat = CoordinateTransform.quaternion_carla_to_dggt(original_quat)

        # Verify magnitude preserved
        assert np.allclose(np.linalg.norm(dggt_quat), 1.0)

    def test_quaternion_transform_protobuf_convention(self):
        """Test quaternion transformation with protobuf convention"""
        # Original quaternion in protobuf format (w,x,y,z)
        original_quat = np.array([0.9, 0.1, 0.2, 0.3])
        original_quat = original_quat / np.linalg.norm(original_quat)

        # Transform to DGGT with protobuf convention
        dggt_quat = CoordinateTransform.quaternion_carla_to_dggt(original_quat, convention="protobuf")

        # Verify magnitude preserved
        assert np.allclose(np.linalg.norm(dggt_quat), 1.0)

    def test_dynamic_object_pose_transform(self):
        """Test dynamic object pose transformation"""
        # Simulate CARLA object pose
        carla_pose = np.eye(4)
        carla_pose[:3, :3] = R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
        carla_pose[:3, 3] = np.array([10, 0, 0])

        # Transform to DGGT
        dggt_pose = CoordinateTransform.transform_pose_carla_to_dggt(carla_pose)

        # Verify shape
        assert dggt_pose.shape == (4, 4)

        # Verify homogeneous coordinates
        assert dggt_pose[3, 3] == 1.0

    def test_identity_pose(self):
        """Test identity pose transformation"""
        identity = np.eye(4)
        dggt_pose = CoordinateTransform.transform_pose_carla_to_dggt(
            identity, include_undo_carla=False
        )

        # Identity pose after transform should have unit rotation + transformed position
        # Position [0,0,0] after transform is still [0,0,0]
        assert np.allclose(dggt_pose[:3, 3], np.zeros(3))

    def test_known_point_correspondence(self):
        """Test known point correspondence"""
        # Feature points in CARLA
        carla_points = {
            'forward': np.array([1, 0, 0]),   # X+
            'right':   np.array([0, 1, 0]),   # Y+
            'up':      np.array([0, 0, 1]),   # Z+
        }

        # Expected positions in DGGT (based on corrected matrix)
        # X_dggt = Y_carla, Y_dggt = -Z_carla, Z_dggt = X_carla
        expected_dggt = {
            'forward': np.array([0, 0, 1]),   # Z+ (Forward in DGGT)
            'right':   np.array([1, 0, 0]),   # X+ (Right in DGGT)
            'up':      np.array([0, -1, 0]),  # Y- (Down in DGGT, so up is -Y)
        }

        for name, carla_pt in carla_points.items():
            dggt_pt = CoordinateTransform.transform_point_carla_to_dggt(carla_pt)
            assert np.allclose(dggt_pt, expected_dggt[name])

    def test_rotation_matrices_are_orthogonal(self):
        """Test that rotation matrices are orthogonal"""
        # CARLA_TO_DGGT_ROTATION should be orthogonal
        carla_to_dggt = CoordinateTransform.CARLA_TO_DGGT_ROTATION
        assert np.allclose(carla_to_dggt @ carla_to_dggt.T, np.eye(3), atol=1e-10)

        # DGGT_TO_CARLA_ROTATION should be orthogonal
        dggt_to_carla = CoordinateTransform.DGGT_TO_CARLA_ROTATION
        assert np.allclose(dggt_to_carla @ dggt_to_carla.T, np.eye(3), atol=1e-10)

    def test_rotation_matrices_are_inverse(self):
        """Test that rotation matrices are inverses of each other"""
        carla_to_dggt = CoordinateTransform.CARLA_TO_DGGT_ROTATION
        dggt_to_carla = CoordinateTransform.DGGT_TO_CARLA_ROTATION

        # Should be inverses
        assert np.allclose(carla_to_dggt @ dggt_to_carla, np.eye(3), atol=1e-10)
        assert np.allclose(dggt_to_carla @ carla_to_dggt, np.eye(3), atol=1e-10)


class TestCoordinateTransformWithNuRec:
    """NuRec transform comparison tests"""

    def test_compare_with_nurec_undo(self):
        """Compare with NuRec undo_carla_coordinate_transform"""
        # P0: Use imported undo_carla_coordinate_transform (from utils.py)

        # Construct test pose
        carla_pose = np.eye(4)
        carla_pose[:3, :3] = R.from_euler('zyx', [30, 45, 60], degrees=True).as_matrix()
        carla_pose[:3, 3] = np.array([100, 50, 20])

        our_result = undo_carla_coordinate_transform(carla_pose)

        # Verify key properties:
        # 1. Y position should be negated
        assert our_result[1, 3] == -carla_pose[1, 3]

        # 2. Rotation part should conform to expectations
        original_rot = R.from_matrix(carla_pose[:3, :3])
        result_rot = R.from_matrix(our_result[:3, :3])

        yaw_o, pitch_o, roll_o = original_rot.as_euler("zyx", degrees=False)
        yaw_r, pitch_r, roll_r = result_rot.as_euler("zyx", degrees=False)

        assert np.isclose(yaw_r, -yaw_o)
        assert np.isclose(pitch_r, -pitch_o)
        assert np.isclose(roll_r, roll_o)

    def test_full_transform_chain_matches_nurec(self):
        """Test full transform chain matches NuRec

        Transform chain (consistent with nurec_integration.py:146):
        dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)
        """
        # Construct test pose (CARLA actor pose)
        carla_pose = np.eye(4)
        carla_pose[:3, :3] = R.from_euler('zyx', [45, 0, 0], degrees=True).as_matrix()
        carla_pose[:3, 3] = np.array([10, 5, 2])

        # Simulate t_carla_dggt (identity matrix for basic test)
        t_carla_dggt = np.eye(4)

        # Full transform chain (consistent with NuRec actor_to_grpc_pose)
        dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)

        # Verify: Y position should be negated
        assert dggt_pose[1, 3] == -carla_pose[1, 3]

        # Verify: Position correctly transformed
        # After undo: [10, -5, 2]
        # Apply t_carla_dggt (identity): [10, -5, 2]
        assert np.allclose(dggt_pose[:3, 3], [10, -5, 2])

    def test_coordinate_transform_class_uses_same_undo(self):
        """Verify CoordinateTransform uses same undo as NuRec"""
        carla_pose = np.eye(4)
        carla_pose[:3, 3] = np.array([50, 25, 10])
        carla_pose[:3, :3] = R.from_euler('zyx', [90, 0, 0], degrees=True).as_matrix()

        # Using CoordinateTransform class
        dggt_pose_class = CoordinateTransform.transform_pose_carla_to_dggt(
            carla_pose, t_carla_dggt=None, include_undo_carla=True
        )

        # Manual chain (NuRec style)
        dggt_pose_manual = undo_carla_coordinate_transform(carla_pose)
        # Then apply basic rotation transform
        transform = np.eye(4)
        transform[:3, :3] = CoordinateTransform.CARLA_TO_DGGT_ROTATION
        dggt_pose_manual = transform @ dggt_pose_manual

        # Should produce same result
        assert np.allclose(dggt_pose_class, dggt_pose_manual)


class TestRotationMatrixProperties:
    """Tests for rotation matrix mathematical properties"""

    def test_carla_to_dggt_determinant(self):
        """CARLA-to-DGGT matrix is a reflection (left-handed to right-handed conversion)

        The determinant is -1 because it converts from left-handed (CARLA) to
        right-handed (DGGT) coordinate system. This is expected for coordinate
        system conversions involving handedness changes.
        """
        det = np.linalg.det(CoordinateTransform.CARLA_TO_DGGT_ROTATION)
        # Reflection has determinant -1, not +1
        assert np.isclose(det, -1.0)

    def test_dggt_to_carla_determinant(self):
        """DGGT-to-CARLA matrix is also a reflection

        Since it's the inverse of a reflection, it also has determinant -1.
        """
        det = np.linalg.det(CoordinateTransform.DGGT_TO_CARLA_ROTATION)
        # Reflection has determinant -1, not +1
        assert np.isclose(det, -1.0)

    def test_rotation_preserves_vector_length(self):
        """Reflection should preserve vector length"""
        test_vectors = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([1, 1, 1]),
            np.array([2, 3, 4]),
        ]

        for vec in test_vectors:
            transformed = CoordinateTransform.CARLA_TO_DGGT_ROTATION @ vec
            # Reflection preserves length (only direction flips)
            assert np.isclose(np.linalg.norm(vec), np.linalg.norm(transformed))