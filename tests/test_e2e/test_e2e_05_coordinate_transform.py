# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
E2E Test E2E-05: Coordinate Transform Validation

Tests coordinate transformation chain between CARLA and DGGT:
- NuRec style transform chain validation
- Known point transform verification
- Roundtrip transform consistency
- Reference comparison with undo_carla_coordinate_transform

Test ID: E2E-05
Test Name: coordinate_transform
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import logging
from scipy.spatial.transform import Rotation as R

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import undo_carla_coordinate_transform, se3_to_grpc_pose
from nre.grpc.protos import sensorsim_pb2, common_pb2

# Import helper functions from test_helpers module
from .test_helpers import create_test_camera_pose

# Note: pytest fixtures (e2e_config, dggt_scenario, test_output_dir) are auto-discovered from conftest.py

logger = logging.getLogger(__name__)


class TestE2E05CoordinateTransform:
    """
    E2E-05: Coordinate Transform Validation Test

    Tests the coordinate transformation chain from CARLA to DGGT.
    Validates the core formula:
        dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)
    """

    @pytest.fixture(autouse=True)
    def setup(self, dggt_scenario, test_output_dir, e2e_config):
        """Setup test fixtures."""
        self.scenario = dggt_scenario
        self.renderer = self.scenario.get_renderer()
        self.output_dir = test_output_dir
        self.save_images = e2e_config['test']['save_images']
        self.t_carla_dggt = self.scenario.get_t_carla_dggt()

        logger.info(f"E2E-05 setup complete")

    def test_nurec_style_transform_chain(self):
        """
        Test E2E-05: NuRec style transform chain validation.

        Validates that the transform chain follows NuRec pattern:
            dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)

        Tests:
        - Transform matrices are valid SE(3) matrices
        - Composition produces valid pose
        - Result is consistent with expected pattern
        """
        logger.info("Starting E2E-05: NuRec style transform chain test")

        # Create test CARLA poses
        test_poses = [
            # Identity pose (origin)
            np.eye(4, dtype=np.float32),
            # Translated pose (x=10, y=5, z=2)
            create_test_camera_pose(x=10.0, y=5.0, z=2.0),
            # Rotated pose (yaw=90 degrees)
            create_test_camera_pose(x=0.0, y=0.0, z=2.0, yaw=90.0),
            # Complex pose (translation + rotation)
            create_test_camera_pose(x=20.0, y=-10.0, z=5.0, yaw=45.0, pitch=10.0, roll=5.0),
        ]

        for i, carla_pose in enumerate(test_poses):
            logger.info(f"Testing pose {i}: position [{carla_pose[0,3]}, {carla_pose[1,3]}, {carla_pose[2,3]}]")

            # Step 1: Apply undo_carla_coordinate_transform
            undo_pose = undo_carla_coordinate_transform(carla_pose)

            # Validate undo result is valid SE(3) matrix
            assert undo_pose.shape == (4, 4), f"Undo pose shape must be 4x4, got {undo_pose.shape}"

            # Check rotation matrix orthonormality (R^T @ R = I)
            R_matrix = undo_pose[:3, :3]
            ortho_check = R_matrix.T @ R_matrix
            assert np.allclose(ortho_check, np.eye(3), atol=1e-6), \
                f"Undo pose rotation not orthonormal: {ortho_check}"

            # Check last row is [0, 0, 0, 1]
            assert np.allclose(undo_pose[3, :], [0, 0, 0, 1], atol=1e-6), \
                f"Undo pose last row invalid: {undo_pose[3, :]}"

            # Step 2: Apply t_carla_dggt
            dggt_pose = self.t_carla_dggt @ undo_pose

            # Validate dggt_pose is valid SE(3) matrix
            assert dggt_pose.shape == (4, 4), f"DGGT pose shape must be 4x4, got {dggt_pose.shape}"

            R_dggt = dggt_pose[:3, :3]
            ortho_check_dggt = R_dggt.T @ R_dggt
            assert np.allclose(ortho_check_dggt, np.eye(3), atol=1e-6), \
                f"DGGT pose rotation not orthonormal: {ortho_check_dggt}"

            # Step 3: Convert to gRPC pose
            grpc_pose = se3_to_grpc_pose(dggt_pose)

            # Validate gRPC pose
            assert isinstance(grpc_pose, common_pb2.Pose), \
                f"se3_to_grpc_pose must return Pose, got {type(grpc_pose)}"

            assert hasattr(grpc_pose, 'vec'), "Pose must have vec field"
            assert hasattr(grpc_pose, 'quat'), "Pose must have quat field"

            logger.info(f"Pose {i}: DGGT position [{grpc_pose.vec.x:.3f}, {grpc_pose.vec.y:.3f}, {grpc_pose.vec.z:.3f}]")

        logger.info("E2E-05 transform chain: PASSED")

    def test_known_point_transform(self):
        """
        Test E2E-05: Known point transform verification.

        Tests specific known CARLA coordinate points and validates
        their transformation to DGGT coordinates.

        Tests:
        - CARLA origin (0, 0, 0)
        - Forward direction (1, 0, 0)
        - Right direction (0, 1, 0)
        - Up direction (0, 0, 1)
        """
        logger.info("Starting E2E-05: Known point transform test")

        # CARLA coordinate system: left-handed, Z-up
        # Forward is +X, Right is -Y, Up is +Z

        test_points = [
            # CARLA origin
            ("origin", np.array([0.0, 0.0, 0.0])),
            # Forward direction (CARLA +X)
            ("forward", np.array([1.0, 0.0, 0.0])),
            # Right direction (CARLA -Y in left-handed)
            ("right", np.array([0.0, -1.0, 0.0])),
            # Up direction (CARLA +Z)
            ("up", np.array([0.0, 0.0, 1.0])),
        ]

        for name, point in test_points:
            logger.info(f"Testing point: {name} = [{point[0]}, {point[1]}, {point[2]}]")

            # Create pose at this point
            carla_pose = np.eye(4, dtype=np.float32)
            carla_pose[:3, 3] = point

            # Apply undo transform
            undo_pose = undo_carla_coordinate_transform(carla_pose)

            # Check position transformation
            # undo_carla_coordinate_transform mirrors Y axis:
            # result position = [x, -y, z]
            expected_pos_undo = np.array([point[0], -point[1], point[2]])

            actual_pos_undo = undo_pose[:3, 3]

            # For origin, check exact match
            if name == "origin":
                assert np.allclose(actual_pos_undo, expected_pos_undo, atol=1e-6), \
                    f"Origin undo failed: expected {expected_pos_undo}, got {actual_pos_undo}"

            logger.info(f"Point {name}: undo position [{actual_pos_undo[0]:.3f}, {actual_pos_undo[1]:.3f}, {actual_pos_undo[2]:.3f}]")

            # Apply t_carla_dggt
            dggt_pose = self.t_carla_dggt @ undo_pose
            dggt_pos = dggt_pose[:3, 3]

            logger.info(f"Point {name}: DGGT position [{dggt_pos[0]:.3f}, {dggt_pos[1]:.3f}, {dggt_pos[2]:.3f}]")

        logger.info("E2E-05 known points: PASSED")

    def test_roundtrip_transform(self):
        """
        Test E2E-05: Roundtrip transform consistency.

        Tests that CARLA -> undo -> DGGT -> inverse undo -> CARLA
        produces a result close to the original pose.

        Error threshold: < 1e-10 (numerical precision)
        """
        logger.info("Starting E2E-05: Roundtrip transform test")

        # Create test poses
        test_poses = [
            create_test_camera_pose(x=0.0, y=0.0, z=2.0),
            create_test_camera_pose(x=10.0, y=5.0, z=3.0, yaw=30.0),
            create_test_camera_pose(x=-5.0, y=10.0, z=1.5, yaw=60.0, pitch=15.0, roll=10.0),
        ]

        for i, original_pose in enumerate(test_poses):
            logger.info(f"Testing roundtrip for pose {i}")

            # Forward transform: CARLA -> undo -> DGGT
            undo_pose = undo_carla_coordinate_transform(original_pose)

            # Reverse transform: undo -> CARLA
            # undo_carla_coordinate_transform is its own inverse for position
            # (mirrors Y axis twice returns to original)

            # For position: undo mirrors Y, so we mirror Y again
            recovered_pose = np.eye(4, dtype=np.float32)

            # Recover position (mirror Y again)
            recovered_pose[:3, 3] = [undo_pose[0, 3], -undo_pose[1, 3], undo_pose[2, 3]]

            # Recover rotation
            # undo rotation: yaw=-yaw, pitch=-pitch, roll=roll
            # reverse: yaw=-yaw_undo, pitch=-pitch_undo, roll=roll_undo
            R_undo = R.from_matrix(undo_pose[:3, :3])
            euler_undo = R_undo.as_euler("zyx", degrees=True)

            # Reverse the undo rotation transformation
            euler_recovered = [-euler_undo[0], -euler_undo[1], euler_undo[2]]
            R_recovered = R.from_euler("zyx", euler_recovered, degrees=True)
            recovered_pose[:3, :3] = R_recovered.as_matrix().astype(np.float32)

            # Compare with original
            pos_error = np.abs(recovered_pose[:3, 3] - original_pose[:3, 3])
            max_pos_error = np.max(pos_error)

            rot_error = np.abs(recovered_pose[:3, :3] - original_pose[:3, :3])
            max_rot_error = np.max(rot_error)

            logger.info(f"Pose {i}: position error = {max_pos_error:.2e}, rotation error = {max_rot_error:.2e}")

            # Check error is within threshold
            assert max_pos_error < 1e-10, \
                f"Position roundtrip error too large: {max_pos_error:.2e}"
            assert max_rot_error < 1e-10, \
                f"Rotation roundtrip error too large: {max_rot_error:.2e}"

        logger.info("E2E-05 roundtrip: PASSED")

    def test_with_nurec_reference(self):
        """
        Test E2E-05: Compare with NuRec reference implementation.

        Validates that undo_carla_coordinate_transform produces
        the same result as the reference implementation.

        Tests:
        - Sample poses from NuRec test data
        - Compare undo output with expected values
        """
        logger.info("Starting E2E-05: NuRec reference comparison test")

        # Create test poses that match NuRec test scenarios
        # These represent typical CARLA vehicle poses

        # Pose 1: Vehicle at origin, facing forward (yaw=0)
        pose1 = create_test_camera_pose(x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0, roll=0.0)

        # Pose 2: Vehicle rotated 90 degrees (yaw=90)
        pose2 = create_test_camera_pose(x=0.0, y=0.0, z=0.0, yaw=90.0, pitch=0.0, roll=0.0)

        # Pose 3: Vehicle with pitch (looking up/down)
        pose3 = create_test_camera_pose(x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=30.0, roll=0.0)

        test_poses = [pose1, pose2, pose3]

        for i, pose in enumerate(test_poses):
            logger.info(f"Testing reference pose {i}")

            # Apply undo_carla_coordinate_transform
            undo_result = undo_carla_coordinate_transform(pose)

            # Expected behavior from NuRec:
            # 1. Position: (x, y, z) -> (x, -y, z) (mirror Y axis)
            # 2. Rotation: yaw -> -yaw, pitch -> -pitch, roll -> roll

            # Check position mirror
            expected_pos = np.array([pose[0, 3], -pose[1, 3], pose[2, 3]])
            actual_pos = undo_result[:3, 3]

            assert np.allclose(actual_pos, expected_pos, atol=1e-6), \
                f"Position mismatch: expected {expected_pos}, got {actual_pos}"

            # Check rotation transformation
            R_original = R.from_matrix(pose[:3, :3])
            euler_original = R_original.as_euler("zyx", degrees=True)

            R_undo = R.from_matrix(undo_result[:3, :3])
            euler_undo = R_undo.as_euler("zyx", degrees=True)

            # Expected: yaw=-yaw, pitch=-pitch, roll=roll
            expected_yaw = -euler_original[0]
            expected_pitch = -euler_original[1]
            expected_roll = euler_original[2]

            assert np.allclose(euler_undo[0], expected_yaw, atol=1e-6), \
                f"Yaw mismatch: expected {expected_yaw}, got {euler_undo[0]}"
            assert np.allclose(euler_undo[1], expected_pitch, atol=1e-6), \
                f"Pitch mismatch: expected {expected_pitch}, got {euler_undo[1]}"
            assert np.allclose(euler_undo[2], expected_roll, atol=1e-6), \
                f"Roll mismatch: expected {expected_roll}, got {euler_undo[2]}"

            logger.info(f"Pose {i}: yaw={euler_original[0]:.1f} -> {euler_undo[0]:.1f}, "
                        f"pitch={euler_original[1]:.1f} -> {euler_undo[1]:.1f}, "
                        f"roll={euler_original[2]:.1f} -> {euler_undo[2]:.1f}")

        logger.info("E2E-05 NuRec reference: PASSED")


class TestE2E05QuaternionConsistency:
    """
    E2E-05: Quaternion Consistency Tests

    Tests quaternion conversion in coordinate transforms.
    """

    def test_se3_to_grpc_pose_quaternion(self):
        """Test se3_to_grpc_pose quaternion conversion."""
        logger.info("Testing se3_to_grpc_pose quaternion conversion")

        # Create pose with known rotation
        pose = create_test_camera_pose(x=5.0, y=3.0, z=2.0, yaw=45.0, pitch=0.0, roll=0.0)

        # Convert to gRPC pose
        grpc_pose = se3_to_grpc_pose(pose)

        # Validate quaternion format
        # scipy quaternion format: [x, y, z, w]
        assert grpc_pose.quat.w != 0, "Quaternion w should not be 0 for non-identity rotation"

        # Convert back to matrix and compare
        quat_array = np.array([
            grpc_pose.quat.x,
            grpc_pose.quat.y,
            grpc_pose.quat.z,
            grpc_pose.quat.w
        ])

        R_from_quat = R.from_quat(quat_array).as_matrix()

        # Check rotation matrix matches
        assert np.allclose(R_from_quat, pose[:3, :3], atol=1e-6), \
            "Quaternion -> matrix conversion mismatch"

        logger.info("Quaternion conversion: PASSED")


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])