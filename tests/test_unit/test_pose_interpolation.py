# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unit tests for Pose Interpolation

Tests the interpolate_pose_pair method from RequestConverter:
- Linear interpolation for position
- SLERP interpolation for rotation (quaternions)
- Edge cases: identity poses, alpha boundaries
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R

# Import the module under test
try:
    from dggt_server.request_converter import RequestConverter
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from dggt_server.request_converter import RequestConverter

# Import fixtures helper
from tests.conftest import create_mock_pose, create_mock_pose_pair, assert_valid_transformation_matrix


@pytest.mark.unit
class TestPoseInterpolation:
    """Tests for RequestConverter.interpolate_pose_pair()"""

    def test_identity_pose_pair_returns_identity(self):
        """Test that interpolating two identity poses returns identity matrix"""
        pose_pair = create_mock_pose_pair()

        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.5)

        assert result.shape == (4, 4)
        assert np.allclose(result, np.eye(4), atol=1e-6)
        assert result.dtype == np.float32

    def test_identity_pose_pair_different_alpha_values(self):
        """Test that identity poses return identity for any alpha"""
        pose_pair = create_mock_pose_pair()

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=alpha)
            assert np.allclose(result, np.eye(4), atol=1e-6)

    def test_linear_position_interpolation_alpha_0_5(self):
        """Test linear interpolation of position at alpha=0.5"""
        start_pose = create_mock_pose(x=0.0, y=0.0, z=0.0)
        end_pose = create_mock_pose(x=10.0, y=5.0, z=2.0)
        pose_pair = create_mock_pose_pair(start_pose=start_pose, end_pose=end_pose)

        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.5)

        expected_pos = np.array([5.0, 2.5, 1.0])
        assert np.allclose(result[:3, 3], expected_pos, atol=1e-6)

    def test_linear_position_interpolation_alpha_0_25(self):
        """Test linear interpolation of position at alpha=0.25"""
        start_pose = create_mock_pose(x=0.0, y=0.0, z=0.0)
        end_pose = create_mock_pose(x=10.0, y=5.0, z=2.0)
        pose_pair = create_mock_pose_pair(start_pose=start_pose, end_pose=end_pose)

        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.25)

        expected_pos = np.array([2.5, 1.25, 0.5])
        assert np.allclose(result[:3, 3], expected_pos, atol=1e-6)

    def test_linear_position_interpolation_alpha_0_75(self):
        """Test linear interpolation of position at alpha=0.75"""
        start_pose = create_mock_pose(x=0.0, y=0.0, z=0.0)
        end_pose = create_mock_pose(x=10.0, y=5.0, z=2.0)
        pose_pair = create_mock_pose_pair(start_pose=start_pose, end_pose=end_pose)

        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.75)

        expected_pos = np.array([7.5, 3.75, 1.5])
        assert np.allclose(result[:3, 3], expected_pos, atol=1e-6)

    def test_alpha_zero_returns_start_pose(self):
        """Test that alpha=0 returns exactly the start pose"""
        start_pose = create_mock_pose(x=0.0, y=0.0, z=0.0, qw=1.0)
        end_pose = create_mock_pose(x=10.0, y=10.0, z=10.0, qw=0.707, qz=0.707)
        pose_pair = create_mock_pose_pair(start_pose=start_pose, end_pose=end_pose)

        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.0)

        assert np.allclose(result[:3, 3], [0.0, 0.0, 0.0])
        assert np.allclose(result[:3, :3], np.eye(3), atol=1e-6)

    def test_alpha_one_returns_end_pose(self):
        """Test that alpha=1 returns exactly the end pose"""
        start_pose = create_mock_pose(x=0.0, y=0.0, z=0.0, qw=1.0)
        end_pose = create_mock_pose(x=10.0, y=5.0, z=2.0, qw=1.0)
        pose_pair = create_mock_pose_pair(start_pose=start_pose, end_pose=end_pose)

        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=1.0)

        assert np.allclose(result[:3, 3], [10.0, 5.0, 2.0])
        assert np.allclose(result[:3, :3], np.eye(3), atol=1e-6)

    def test_slerp_rotation_interpolation_90_deg_z(self):
        """Test SLERP interpolation for 90 degree rotation around Z"""
        # Start: identity rotation (w=1, x=y=z=0)
        start_pose = create_mock_pose(qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        # End: 90 degree rotation around Z axis
        # Quaternion for 90 deg Z: w=cos(45)=0.707, z=sin(45)=0.707
        angle_45 = np.pi / 4
        end_pose = create_mock_pose(
            qw=np.cos(angle_45),
            qx=0.0,
            qy=0.0,
            qz=np.sin(angle_45)
        )

        pose_pair = create_mock_pose_pair(start_pose=start_pose, end_pose=end_pose)

        # Interpolate at alpha=0.5 should give ~45 degree rotation
        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.5)

        # Verify it's a valid transformation matrix
        assert_valid_transformation_matrix(result)

        # Should be approximately 45 degrees around Z
        expected_rot = R.from_euler('z', 45, degrees=True).as_matrix()
        assert np.allclose(result[:3, :3], expected_rot, atol=0.1)

    def test_slerp_rotation_preserves_quaternion_norm(self):
        """Test that SLERP produces valid unit quaternions"""
        # Create two different rotations
        start_pose = create_mock_pose(qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        end_pose = create_mock_pose(qw=0.5, qx=0.5, qy=0.5, qz=0.5)

        pose_pair = create_mock_pose_pair(start_pose=start_pose, end_pose=end_pose)

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=alpha)

            # Check rotation matrix is valid (orthogonal, det=1)
            rot = result[:3, :3]
            assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-5)
            assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-6)

    def test_output_dtype_is_float32(self):
        """Test that output matrix has float32 dtype"""
        pose_pair = create_mock_pose_pair()
        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.5)

        assert result.dtype == np.float32

    def test_output_shape_is_4x4(self):
        """Test that output is always 4x4 matrix"""
        pose_pair = create_mock_pose_pair()

        for alpha in [0.0, 0.5, 1.0]:
            result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=alpha)
            assert result.shape == (4, 4)

    def test_combined_translation_and_rotation(self):
        """Test interpolation with both translation and rotation"""
        # Start pose: identity at origin
        start_pose = create_mock_pose(x=0.0, y=0.0, z=0.0, qw=1.0)

        # End pose: translated and rotated 90 deg around Z
        angle_45 = np.pi / 4
        end_pose = create_mock_pose(
            x=10.0, y=5.0, z=2.0,
            qw=np.cos(angle_45),
            qz=np.sin(angle_45)
        )

        pose_pair = create_mock_pose_pair(start_pose=start_pose, end_pose=end_pose)

        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.5)

        # Check position is mid-point
        expected_pos = np.array([5.0, 2.5, 1.0])
        assert np.allclose(result[:3, 3], expected_pos, atol=1e-6)

        # Check rotation is approximately 45 deg around Z
        expected_rot = R.from_euler('z', 45, degrees=True).as_matrix()
        assert np.allclose(result[:3, :3], expected_rot, atol=0.1)

    def test_negative_translation_values(self):
        """Test interpolation with negative coordinates"""
        start_pose = create_mock_pose(x=-5.0, y=-10.0, z=-2.0)
        end_pose = create_mock_pose(x=5.0, y=10.0, z=2.0)
        pose_pair = create_mock_pose_pair(start_pose=start_pose, end_pose=end_pose)

        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.5)

        expected_pos = np.array([0.0, 0.0, 0.0])
        assert np.allclose(result[:3, 3], expected_pos, atol=1e-6)

    def test_large_translation_values(self):
        """Test interpolation with large coordinate values"""
        start_pose = create_mock_pose(x=0.0, y=0.0, z=0.0)
        end_pose = create_mock_pose(x=1000.0, y=500.0, z=200.0)
        pose_pair = create_mock_pose_pair(start_pose=start_pose, end_pose=end_pose)

        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.5)

        expected_pos = np.array([500.0, 250.0, 100.0])
        assert np.allclose(result[:3, 3], expected_pos, atol=1e-6)


@pytest.mark.unit
class TestPoseInterpolationEdgeCases:
    """Edge case tests for pose interpolation"""

    def test_very_small_alpha(self):
        """Test interpolation with very small alpha value"""
        start_pose = create_mock_pose(x=0.0, y=0.0, z=0.0)
        end_pose = create_mock_pose(x=10.0, y=5.0, z=2.0)
        pose_pair = create_mock_pose_pair(start_pose=start_pose, end_pose=end_pose)

        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=1e-6)

        # Should be very close to start pose
        assert np.allclose(result[:3, 3], [0.0, 0.0, 0.0], atol=1e-4)

    def test_very_close_to_one_alpha(self):
        """Test interpolation with alpha very close to 1"""
        start_pose = create_mock_pose(x=0.0, y=0.0, z=0.0)
        end_pose = create_mock_pose(x=10.0, y=5.0, z=2.0)
        pose_pair = create_mock_pose_pair(start_pose=start_pose, end_pose=end_pose)

        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.999999)

        # Should be very close to end pose
        assert np.allclose(result[:3, 3], [10.0, 5.0, 2.0], atol=1e-4)

    def test_same_start_and_end_pose(self):
        """Test interpolation when start and end are identical"""
        pose = create_mock_pose(x=5.0, y=3.0, z=1.0, qw=0.707, qz=0.707)
        pose_pair = create_mock_pose_pair(start_pose=pose, end_pose=pose)

        for alpha in [0.0, 0.5, 1.0]:
            result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=alpha)
            # Result should be the same pose for any alpha
            assert np.allclose(result[:3, 3], [5.0, 3.0, 1.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])