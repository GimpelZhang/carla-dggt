# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unit tests for Coordinate Transform

Tests the coordinate transformation utilities:
- CARLA to DGGT translation transform
- CARLA to DGGT rotation transform
- Roundtrip consistency
- undo_carla_coordinate_transform exact implementation test
- NuRec-style transform chain: dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)

CRITICAL: MUST import undo_carla_coordinate_transform from utils.py, NEVER reimplement!
Reference: utils.py lines 98-116
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R

# CRITICAL: Import coordinate transform from utils.py, NEVER reimplement!
try:
    from utils import undo_carla_coordinate_transform
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from utils import undo_carla_coordinate_transform

# Import DGGT server modules
try:
    from dggt_server.request_converter import RequestConverter
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from dggt_server.request_converter import RequestConverter

# Import fixtures helper
from tests.conftest import assert_valid_transformation_matrix


@pytest.mark.unit
class TestCoordinateTransform:
    """Tests for undo_carla_coordinate_transform from utils.py"""

    def test_identity_transform_returns_modified_identity(self):
        """Test that identity input produces correct output"""
        # Input: identity transform
        carla_pose = np.eye(4, dtype=np.float32)

        result = undo_carla_coordinate_transform(carla_pose)

        # Output should have:
        # - Same rotation (identity rotation stays identity after yaw/pitch/roll negation)
        # - Position: [x, -y, z] from [x, y, z]
        assert result.shape == (4, 4)
        # For identity rotation: yaw=0, pitch=0, roll=0
        # After transformation: yaw=-0=0, pitch=-0=0, roll=0
        # Rotation stays identity
        assert np.allclose(result[:3, :3], np.eye(3), atol=1e-6)
        # Position: [0, -0, 0] = [0, 0, 0]
        assert np.allclose(result[:3, 3], [0.0, 0.0, 0.0], atol=1e-6)
        # Last row unchanged
        assert np.allclose(result[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6)

    def test_translation_transform_y_axis_negated(self):
        """Test that translation y-axis is negated"""
        carla_pose = np.eye(4, dtype=np.float32)
        carla_pose[:3, 3] = [10.0, 5.0, 2.0]

        result = undo_carla_coordinate_transform(carla_pose)

        # Position: [x, -y, z] -> [10.0, -5.0, 2.0]
        expected_pos = np.array([10.0, -5.0, 2.0])
        assert np.allclose(result[:3, 3], expected_pos, atol=1e-6)

    def test_rotation_yaw_negated(self):
        """Test that yaw rotation is negated"""
        # Create transform with 45 degree yaw around Z
        yaw_angle = np.pi / 4  # 45 degrees
        rotation = R.from_euler('z', yaw_angle, degrees=False).as_matrix()
        carla_pose = np.eye(4, dtype=np.float32)
        carla_pose[:3, :3] = rotation

        result = undo_carla_coordinate_transform(carla_pose)

        # Extract yaw from result
        result_rot = R.from_matrix(result[:3, :3])
        yaw_result, pitch_result, roll_result = result_rot.as_euler('zyx', degrees=False)

        # Original yaw was +45 deg, result should be -45 deg
        assert np.isclose(yaw_result, -yaw_angle, atol=1e-6)

    def test_rotation_pitch_negated(self):
        """Test that pitch rotation is negated"""
        # Create transform with 30 degree pitch around Y
        pitch_angle = np.pi / 6  # 30 degrees
        rotation = R.from_euler('y', pitch_angle, degrees=False).as_matrix()
        carla_pose = np.eye(4, dtype=np.float32)
        carla_pose[:3, :3] = rotation

        result = undo_carla_coordinate_transform(carla_pose)

        # Extract pitch from result
        result_rot = R.from_matrix(result[:3, :3])
        yaw_result, pitch_result, roll_result = result_rot.as_euler('zyx', degrees=False)

        # Original pitch was +30 deg, result should be -30 deg
        assert np.isclose(pitch_result, -pitch_angle, atol=1e-6)

    def test_roll_unchanged(self):
        """Test that roll rotation is unchanged"""
        # Create transform with 20 degree roll around X
        roll_angle = np.pi / 9  # 20 degrees
        rotation = R.from_euler('x', roll_angle, degrees=False).as_matrix()
        carla_pose = np.eye(4, dtype=np.float32)
        carla_pose[:3, :3] = rotation

        result = undo_carla_coordinate_transform(carla_pose)

        # Extract roll from result
        result_rot = R.from_matrix(result[:3, :3])
        yaw_result, pitch_result, roll_result = result_rot.as_euler('zyx', degrees=False)

        # Roll should be unchanged
        assert np.isclose(roll_result, roll_angle, atol=1e-6)

    def test_combined_rotation_and_translation(self):
        """Test combined rotation and translation transformation"""
        # Create pose with both rotation and translation
        yaw_angle = np.pi / 6  # 30 degrees
        rotation = R.from_euler('z', yaw_angle, degrees=False).as_matrix()
        carla_pose = np.eye(4, dtype=np.float32)
        carla_pose[:3, :3] = rotation.astype(np.float32)
        carla_pose[:3, 3] = [10.0, 5.0, 2.0]

        result = undo_carla_coordinate_transform(carla_pose)

        # Verify position transform
        expected_pos = np.array([10.0, -5.0, 2.0])
        assert np.allclose(result[:3, 3], expected_pos, atol=1e-6)

        # Verify yaw is negated
        result_rot = R.from_matrix(result[:3, :3])
        yaw_result, _, _ = result_rot.as_euler('zyx', degrees=False)
        assert np.isclose(yaw_result, -yaw_angle, atol=1e-6)

    def test_output_dtype_preserved(self):
        """Test that output dtype matches input (float64 for default np.eye)"""
        # Note: utils.py uses np.eye(4) which defaults to float64
        carla_pose = np.eye(4, dtype=np.float64)
        result = undo_carla_coordinate_transform(carla_pose)
        # Implementation uses np.eye(4) which is float64, so output is float64
        assert result.dtype == np.float64

        # If input is float32, result is float64 (due to np.eye(4) in implementation)
        carla_pose_f32 = np.eye(4, dtype=np.float32)
        result_f32 = undo_carla_coordinate_transform(carla_pose_f32)
        # Implementation internally uses np.eye(4) which is float64
        assert result_f32.dtype == np.float64

    def test_output_shape_is_4x4(self):
        """Test that output is always 4x4"""
        for _ in range(5):
            random_rot = R.from_euler('zyx', np.random.uniform(-np.pi, np.pi, 3)).as_matrix()
            carla_pose = np.eye(4)
            carla_pose[:3, :3] = random_rot
            carla_pose[:3, 3] = np.random.uniform(-100, 100, 3)

            result = undo_carla_coordinate_transform(carla_pose)
            assert result.shape == (4, 4)


@pytest.mark.unit
class TestCoordinateTransformRoundtrip:
    """Tests for roundtrip consistency of coordinate transform"""

    def test_roundtrip_identity_pose(self):
        """Test roundtrip on identity pose"""
        carla_pose = np.eye(4, dtype=np.float32)

        # Apply transform twice
        result1 = undo_carla_coordinate_transform(carla_pose)
        result2 = undo_carla_coordinate_transform(result1)

        # After two applications: yaw negated twice = original, pitch negated twice = original
        # y-axis negated twice = original y
        # Position: [x, -y, z] -> [x, -(-y), z] = [x, y, z]
        assert np.allclose(result2[:3, 3], carla_pose[:3, 3], atol=1e-6)
        # Rotation: yaw and pitch negated twice, so they return to original
        # This is NOT a perfect roundtrip for arbitrary poses due to the nature of the transform
        # But for identity rotation it works

    def test_double_transform_position_signs(self):
        """Test that double transform flips y-axis twice"""
        carla_pose = np.eye(4, dtype=np.float32)
        carla_pose[:3, 3] = [10.0, 5.0, 2.0]

        result1 = undo_carla_coordinate_transform(carla_pose)
        result2 = undo_carla_coordinate_transform(result1)

        # y flips twice: 5.0 -> -5.0 -> 5.0
        assert np.isclose(result2[1, 3], 5.0, atol=1e-6)
        # x and z unchanged
        assert np.isclose(result2[0, 3], 10.0, atol=1e-6)
        assert np.isclose(result2[2, 3], 2.0, atol=1e-6)


@pytest.mark.unit
class TestNuRecTransformChain:
    """Tests for NuRec-style transform chain used in DGGT rendering"""

    def test_transform_chain_identity_t_carla_dggt(self):
        """Test transform chain with identity t_carla_dggt"""
        # NuRec transform chain: dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)
        t_carla_dggt = np.eye(4, dtype=np.float32)

        carla_pose = np.eye(4, dtype=np.float32)
        carla_pose[:3, 3] = [10.0, 5.0, 2.0]

        # Apply the chain (same as utils.py line 146)
        dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)

        # With identity t_carla_dggt, should match undo result
        expected = undo_carla_coordinate_transform(carla_pose)
        assert np.allclose(dggt_pose, expected, atol=1e-6)

    def test_transform_chain_with_translation_offset(self):
        """Test transform chain with t_carla_dggt translation"""
        t_carla_dggt = np.eye(4, dtype=np.float32)
        t_carla_dggt[:3, 3] = [100.0, 50.0, 10.0]  # DGGT world offset

        carla_pose = np.eye(4, dtype=np.float32)
        carla_pose[:3, 3] = [10.0, 5.0, 2.0]

        # Apply the chain
        dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)

        # undo_carla_coordinate_transform(carla_pose) position: [10.0, -5.0, 2.0]
        # Then add t_carla_dggt translation: [110.0, 45.0, 12.0]
        expected_pos = np.array([110.0, 45.0, 12.0])
        assert np.allclose(dggt_pose[:3, 3], expected_pos, atol=1e-6)

    def test_transform_chain_with_rotation_transform(self):
        """Test transform chain with t_carla_dggt rotation"""
        # t_carla_dggt with 90 degree rotation around Z
        t_rotation = R.from_euler('z', np.pi / 2, degrees=False).as_matrix()
        t_carla_dggt = np.eye(4, dtype=np.float32)
        t_carla_dggt[:3, :3] = t_rotation.astype(np.float32)

        carla_pose = np.eye(4, dtype=np.float32)
        carla_pose[:3, 3] = [10.0, 0.0, 0.0]

        # Apply the chain
        dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)

        # undo result: position [10.0, 0.0, 0.0], rotation identity
        # t_rotation rotates the position: 90 deg Z rotates (10, 0) -> (0, 10)
        expected_pos = np.array([0.0, 10.0, 0.0])
        assert np.allclose(dggt_pose[:3, 3], expected_pos, atol=1e-6)

    def test_request_converter_uses_correct_chain(self):
        """Verify RequestConverter uses the exact transform chain from utils.py"""
        # This test ensures RequestConverter.convert_dynamic_objects
        # uses the same chain as utils.py actor_to_grpc_pose (line 146)

        from dggt_server.scene_metadata import TrackIDMapping
        from unittest.mock import Mock
        from nre.grpc.protos import sensorsim_pb2

        track_mapping = TrackIDMapping(auto_prefix="dggt_obj")
        track_mapping.register("test_obj", 0)

        t_carla_dggt = np.eye(4, dtype=np.float32)
        t_carla_dggt[:3, 3] = [100.0, 50.0, 10.0]

        # Create mock dynamic object
        dyn_obj = sensorsim_pb2.DynamicObject()
        dyn_obj.track_id = "test_obj"
        # Identity poses
        for pose_attr in ['start_pose', 'end_pose']:
            pose = getattr(dyn_obj.pose_pair, pose_attr)
            pose.vec.x = 10.0
            pose.vec.y = 5.0
            pose.vec.z = 2.0
            pose.quat.w = 1.0
            pose.quat.x = 0.0
            pose.quat.y = 0.0
            pose.quat.z = 0.0

        # Convert using RequestConverter
        result = RequestConverter.convert_dynamic_objects(
            [dyn_obj], track_mapping, t_carla_dggt, alpha=0.5
        )

        # Manually compute expected result using the chain
        carla_pose = np.eye(4, dtype=np.float32)
        carla_pose[:3, 3] = [10.0, 5.0, 2.0]
        expected_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)

        # Compare
        assert len(result) == 1
        _, actual_pose = result[0]
        assert np.allclose(actual_pose, expected_pose, atol=1e-6)


@pytest.mark.unit
class TestCoordinateTransformExactImplementation:
    """Tests verifying exact implementation from utils.py lines 98-116"""

    def test_exact_line_115_position_formula(self):
        """Test exact position formula from utils.py line 115"""
        # Line 115: result[:3, 3] = [transform[0, 3], -transform[1, 3], transform[2, 3]]
        carla_pose = np.eye(4, dtype=np.float32)
        carla_pose[0, 3] = 100.0  # x
        carla_pose[1, 3] = -25.0  # y (negative value to test sign flip)
        carla_pose[2, 3] = 50.0   # z

        result = undo_carla_coordinate_transform(carla_pose)

        # Exact formula: [x, -y, z]
        assert result[0, 3] == carla_pose[0, 3]  # x unchanged
        assert result[1, 3] == -carla_pose[1, 3]  # y negated: -(-25) = 25
        assert result[2, 3] == carla_pose[2, 3]   # z unchanged

    def test_exact_rotation_formula_lines_105_112(self):
        """Test exact rotation formula from utils.py lines 105-112"""
        # Create known rotation
        yaw = np.pi / 4   # 45 degrees
        pitch = np.pi / 6  # 30 degrees
        roll = np.pi / 9   # 20 degrees

        rotation = R.from_euler('zyx', [yaw, pitch, roll], degrees=False).as_matrix()
        carla_pose = np.eye(4, dtype=np.float32)
        carla_pose[:3, :3] = rotation.astype(np.float32)

        result = undo_carla_coordinate_transform(carla_pose)

        # Extract Euler angles from result
        result_rot = R.from_matrix(result[:3, :3])
        yaw_r, pitch_r, roll_r = result_rot.as_euler('zyx', degrees=False)

        # Exact formula from lines 109-111:
        # yaw = -yaw, pitch = -pitch, roll = roll
        assert np.isclose(yaw_r, -yaw, atol=1e-6)
        assert np.isclose(pitch_r, -pitch, atol=1e-6)
        assert np.isclose(roll_r, roll, atol=1e-6)

    def test_result_matrix_structure_line_104_116(self):
        """Test that result matrix structure matches utils.py implementation"""
        carla_pose = np.eye(4, dtype=np.float32)

        result = undo_carla_coordinate_transform(carla_pose)

        # Line 104: result = np.eye(4)
        # Lines 114-115: rotation set, position set
        # Line 116: return result

        # Verify result is 4x4
        assert result.shape == (4, 4)

        # Verify last row is [0, 0, 0, 1] (from np.eye(4) initialization)
        assert np.allclose(result[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])