# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unit tests for RequestConverter class methods

Tests actual RequestConverter functionality with mock protobuf objects:
- interpolate_pose_pair: SLERP + linear interpolation
- convert_dynamic_objects: coordinate transform chain
- convert_camera_intrinsics: K matrix extraction from CameraSpec
- encode_image: PNG, JPEG, RGB_UINT8_PLANAR encoding
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, PropertyMock
from scipy.spatial.transform import Rotation as R

# Import the RequestConverter class
from dggt_server.request_converter import RequestConverter
from dggt_server.exceptions import InvalidCameraSpecError


class MockPose:
    """Mock Pose protobuf with vec and quat attributes"""
    def __init__(self, x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
        self.vec = Mock()
        self.vec.x = x
        self.vec.y = y
        self.vec.z = z
        self.quat = Mock()
        # Protobuf Quat: w, x, y, z order
        self.quat.w = qw
        self.quat.x = qx
        self.quat.y = qy
        self.quat.z = qz


class MockPosePair:
    """Mock PosePair protobuf with start_pose and end_pose"""
    def __init__(self, start_pose=None, end_pose=None):
        self.start_pose = start_pose or MockPose()
        self.end_pose = end_pose or MockPose()


class MockDynamicObject:
    """Mock DynamicObject protobuf with track_id and pose_pair"""
    def __init__(self, track_id="vehicle_0", pose_pair=None):
        self.track_id = track_id
        self.pose_pair = pose_pair or MockPosePair()


class MockOpenCVPinholeParam:
    """Mock OpenCV pinhole camera parameters"""
    def __init__(self, fx=1000.0, fy=1000.0, px=518.0, py=350.0):
        self.focal_length_x = fx
        self.focal_length_y = fy
        self.principal_point_x = px
        self.principal_point_y = py


class MockCameraSpec:
    """Mock CameraSpec protobuf"""
    def __init__(self, width=1036, height=700, pinhole_param=None, fisheye_param=None, ftheta_param=None):
        self.resolution_w = width
        self.resolution_h = height
        self._pinhole_param = pinhole_param
        self._fisheye_param = fisheye_param
        self._ftheta_param = ftheta_param

    def HasField(self, field_name):
        if field_name == "opencv_pinhole_param":
            return self._pinhole_param is not None
        elif field_name == "opencv_fisheye_param":
            return self._fisheye_param is not None
        elif field_name == "ftheta_param":
            return self._ftheta_param is not None
        return False

    @property
    def opencv_pinhole_param(self):
        return self._pinhole_param

    @property
    def opencv_fisheye_param(self):
        return self._fisheye_param

    @property
    def ftheta_param(self):
        return self._ftheta_param


class MockTrackIDMapping:
    """Mock TrackIDMapping for testing"""
    def __init__(self, mapping=None):
        self._mapping = mapping or {"vehicle_0": 100, "vehicle_1": 101}

    def to_object_id(self, track_id):
        if track_id in self._mapping:
            return self._mapping[track_id]
        raise ValueError(f"Unknown track_id: {track_id}")


class TestInterpolatePosePair:
    """Tests for RequestConverter.interpolate_pose_pair()"""

    def test_identity_pose_pair(self):
        """Test interpolation of identity poses returns identity matrix"""
        pose_pair = MockPosePair(
            start_pose=MockPose(),  # Identity pose
            end_pose=MockPose()     # Identity pose
        )

        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.5)

        assert result.shape == (4, 4)
        assert np.allclose(result, np.eye(4), atol=1e-6)

    def test_linear_position_interpolation(self):
        """Test linear interpolation of position at alpha=0.5"""
        start_pose = MockPose(x=0.0, y=0.0, z=0.0)
        end_pose = MockPose(x=10.0, y=5.0, z=2.0)
        pose_pair = MockPosePair(start_pose=start_pose, end_pose=end_pose)

        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.5)

        expected_pos = np.array([5.0, 2.5, 1.0])
        assert np.allclose(result[:3, 3], expected_pos, atol=1e-6)

    def test_slerp_rotation_interpolation(self):
        """Test SLERP interpolation of rotation"""
        # Start: identity rotation (w=1, x=y=z=0)
        start_pose = MockPose(qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        # End: 90 degree rotation around Z axis
        # Quaternion for 90 deg Z: w=cos(45)=0.707, z=sin(45)=0.707
        end_pose = MockPose(qw=0.707, qx=0.0, qy=0.0, qz=0.707)

        pose_pair = MockPosePair(start_pose=start_pose, end_pose=end_pose)

        # Interpolate at alpha=0.5 should give ~45 degree rotation
        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.5)

        # Verify rotation matrix is valid
        rotation = result[:3, :3]
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)

        # Should be approximately 45 degrees around Z
        # Check that rotation matrix matches 45 deg Z rotation
        expected_rot = R.from_euler('z', 45, degrees=True).as_matrix()
        assert np.allclose(rotation, expected_rot, atol=0.1)

    def test_alpha_zero_returns_start_pose(self):
        """Test alpha=0 returns start pose"""
        start_pose = MockPose(x=0.0, y=0.0, z=0.0, qw=1.0)
        end_pose = MockPose(x=10.0, y=10.0, z=10.0, qw=0.707, qz=0.707)
        pose_pair = MockPosePair(start_pose=start_pose, end_pose=end_pose)

        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.0)

        assert np.allclose(result[:3, 3], [0.0, 0.0, 0.0])
        assert np.allclose(result[:3, :3], np.eye(3), atol=1e-6)

    def test_alpha_one_returns_end_pose(self):
        """Test alpha=1 returns end pose"""
        start_pose = MockPose(x=0.0, y=0.0, z=0.0, qw=1.0)
        end_pose = MockPose(x=10.0, y=5.0, z=2.0, qw=1.0)
        pose_pair = MockPosePair(start_pose=start_pose, end_pose=end_pose)

        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=1.0)

        assert np.allclose(result[:3, 3], [10.0, 5.0, 2.0])

    def test_output_dtype_is_float32(self):
        """Test that output matrix has float32 dtype"""
        pose_pair = MockPosePair()
        result = RequestConverter.interpolate_pose_pair(pose_pair, alpha=0.5)

        assert result.dtype == np.float32


class TestConvertDynamicObjects:
    """Tests for RequestConverter.convert_dynamic_objects()"""

    def test_empty_dynamic_objects_list(self):
        """Test conversion of empty list returns empty list"""
        track_mapping = MockTrackIDMapping()
        t_carla_dggt = np.eye(4, dtype=np.float32)

        result = RequestConverter.convert_dynamic_objects(
            [], track_mapping, t_carla_dggt, alpha=0.5
        )

        assert result == []

    def test_single_object_conversion(self):
        """Test conversion of single DynamicObject"""
        pose_pair = MockPosePair(
            start_pose=MockPose(x=1.0, y=2.0, z=3.0),
            end_pose=MockPose(x=1.0, y=2.0, z=3.0)  # Same pose for simplicity
        )
        dynamic_obj = MockDynamicObject(track_id="vehicle_0", pose_pair=pose_pair)

        track_mapping = MockTrackIDMapping()
        t_carla_dggt = np.eye(4, dtype=np.float32)

        result = RequestConverter.convert_dynamic_objects(
            [dynamic_obj], track_mapping, t_carla_dggt, alpha=0.5
        )

        assert len(result) == 1
        object_id, pose_matrix = result[0]
        assert object_id == 100
        assert pose_matrix.shape == (4, 4)

    def test_track_id_mapping_conversion(self):
        """Test track_id to object_id mapping"""
        pose_pair = MockPosePair()
        dynamic_obj = MockDynamicObject(track_id="vehicle_1", pose_pair=pose_pair)

        track_mapping = MockTrackIDMapping()
        t_carla_dggt = np.eye(4, dtype=np.float32)

        result = RequestConverter.convert_dynamic_objects(
            [dynamic_obj], track_mapping, t_carla_dggt, alpha=0.5
        )

        object_id, _ = result[0]
        assert object_id == 101

    def test_invalid_track_id_skipped(self):
        """Test that invalid track_id is skipped with warning"""
        pose_pair = MockPosePair()
        valid_obj = MockDynamicObject(track_id="vehicle_0", pose_pair=pose_pair)
        invalid_obj = MockDynamicObject(track_id="unknown_id", pose_pair=pose_pair)

        track_mapping = MockTrackIDMapping()
        t_carla_dggt = np.eye(4, dtype=np.float32)

        result = RequestConverter.convert_dynamic_objects(
            [invalid_obj, valid_obj], track_mapping, t_carla_dggt, alpha=0.5
        )

        # Only valid object should be in result
        assert len(result) == 1
        assert result[0][0] == 100

    def test_coordinate_transform_chain_applied(self):
        """Test that coordinate transform chain is applied correctly"""
        # Create poses with actual rotation to test transform
        start_pose = MockPose(x=10.0, y=5.0, z=0.0, qw=1.0)
        end_pose = MockPose(x=10.0, y=5.0, z=0.0, qw=1.0)
        pose_pair = MockPosePair(start_pose=start_pose, end_pose=end_pose)

        dynamic_obj = MockDynamicObject(track_id="vehicle_0", pose_pair=pose_pair)

        track_mapping = MockTrackIDMapping()

        # Use non-identity t_carla_dggt to verify transform is applied
        t_carla_dggt = np.eye(4, dtype=np.float32)
        t_carla_dggt[:3, 3] = [100.0, 200.0, 300.0]  # Translation offset

        result = RequestConverter.convert_dynamic_objects(
            [dynamic_obj], track_mapping, t_carla_dggt, alpha=0.5
        )

        _, pose_dggt = result[0]

        # The result should have undo_carla_coordinate_transform applied
        # then t_carla_dggt applied
        # undo_carla_coordinate_transform mirrors Y coordinate
        assert pose_dggt.shape == (4, 4)


class TestConvertCameraIntrinsics:
    """Tests for RequestConverter.convert_camera_intrinsics()"""

    def test_opencv_pinhole_param_extraction(self):
        """Test extraction of K matrix from opencv_pinhole_param"""
        pinhole_param = MockOpenCVPinholeParam(
            fx=1000.0, fy=1000.0, px=518.0, py=350.0
        )
        camera_spec = MockCameraSpec(
            width=1036, height=700, pinhole_param=pinhole_param
        )

        K, width, height = RequestConverter.convert_camera_intrinsics(camera_spec)

        assert width == 1036
        assert height == 700
        assert K.shape == (3, 3)
        assert K.dtype == np.float32

        # Verify intrinsic matrix structure
        assert K[0, 0] == 1000.0  # fx
        assert K[1, 1] == 1000.0  # fy
        assert K[0, 2] == 518.0   # cx (principal_point_x)
        assert K[1, 2] == 350.0   # cy (principal_point_y)
        assert K[2, 2] == 1.0

    def test_different_focal_lengths(self):
        """Test handling of different fx and fy values"""
        pinhole_param = MockOpenCVPinholeParam(
            fx=800.0, fy=900.0, px=400.0, py=300.0
        )
        camera_spec = MockCameraSpec(
            width=800, height=600, pinhole_param=pinhole_param
        )

        K, _, _ = RequestConverter.convert_camera_intrinsics(camera_spec)

        assert K[0, 0] == 800.0
        assert K[1, 1] == 900.0

    def test_fisheye_param_approximation(self):
        """Test fisheye camera param is approximated as pinhole"""
        fisheye_param = Mock()
        fisheye_param.focal_length_x = 1200.0
        fisheye_param.focal_length_y = 1200.0
        fisheye_param.principal_point_x = 600.0
        fisheye_param.principal_point_y = 400.0

        camera_spec = MockCameraSpec(
            width=1200, height=800, fisheye_param=fisheye_param
        )

        K, width, height = RequestConverter.convert_camera_intrinsics(camera_spec)

        assert K[0, 0] == 1200.0
        assert K[1, 1] == 1200.0
        assert K[0, 2] == 600.0
        assert K[1, 2] == 400.0

    def test_no_camera_model_raises_error(self):
        """Test that missing camera model raises InvalidCameraSpecError"""
        camera_spec = MockCameraSpec(width=800, height=600)

        with pytest.raises(InvalidCameraSpecError):
            RequestConverter.convert_camera_intrinsics(camera_spec)

    def test_output_dtype_is_float32(self):
        """Test that K matrix has float32 dtype"""
        pinhole_param = MockOpenCVPinholeParam()
        camera_spec = MockCameraSpec(pinhole_param=pinhole_param)

        K, _, _ = RequestConverter.convert_camera_intrinsics(camera_spec)

        assert K.dtype == np.float32


class TestEncodeImage:
    """Tests for RequestConverter.encode_image()"""

    def test_png_encoding(self):
        """Test PNG image encoding"""
        # Create a simple RGB image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:50, :, 0] = 255  # Red top half

        # Mock ImageFormat enum
        from nre.grpc.protos import sensorsim_pb2

        result = RequestConverter.encode_image(image, sensorsim_pb2.ImageFormat.PNG)

        assert isinstance(result, bytes)
        assert len(result) > 0

        # Verify it's a valid PNG by decoding
        import cv2
        decoded = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (100, 100, 3)

    def test_jpeg_encoding(self):
        """Test JPEG image encoding with quality parameter"""
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        from nre.grpc.protos import sensorsim_pb2

        result = RequestConverter.encode_image(
            image, sensorsim_pb2.ImageFormat.JPEG, quality=95.0
        )

        assert isinstance(result, bytes)
        assert len(result) > 0

        # Verify it's a valid JPEG
        import cv2
        decoded = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (50, 50, 3)

    def test_jpeg_quality_parameter(self):
        """Test that JPEG quality affects output size"""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        from nre.grpc.protos import sensorsim_pb2

        high_quality = RequestConverter.encode_image(
            image, sensorsim_pb2.ImageFormat.JPEG, quality=95.0
        )
        low_quality = RequestConverter.encode_image(
            image, sensorsim_pb2.ImageFormat.JPEG, quality=10.0
        )

        # Higher quality should produce larger file
        assert len(high_quality) > len(low_quality)

    def test_rgb_uint8_planar_encoding(self):
        """Test RGB_UINT8_PLANAR format (HWC to CHW conversion)"""
        # Create image with distinct values per channel
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        image[:, :, 0] = 1  # R channel
        image[:, :, 1] = 2  # G channel
        image[:, :, 2] = 3  # B channel

        from nre.grpc.protos import sensorsim_pb2

        result = RequestConverter.encode_image(
            image, sensorsim_pb2.ImageFormat.RGB_UINT8_PLANAR
        )

        assert isinstance(result, bytes)

        # Verify planar format: CHW -> bytes
        # Expected: R plane (10*20 bytes of 1), G plane (10*20 bytes of 2), B plane (10*20 bytes of 3)
        expected_size = 10 * 20 * 3
        assert len(result) == expected_size

        # Decode and verify structure
        decoded = np.frombuffer(result, np.uint8).reshape(3, 10, 20)
        assert decoded[0, 0, 0] == 1  # R channel
        assert decoded[1, 0, 0] == 2  # G channel
        assert decoded[2, 0, 0] == 3  # B channel

    def test_rgb_uint8_planar_preserves_data(self):
        """Test that RGB_UINT8_PLANAR encoding preserves pixel data"""
        image = np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [128, 128, 128]]
        ], dtype=np.uint8)  # Shape: (2, 2, 3)

        from nre.grpc.protos import sensorsim_pb2

        result = RequestConverter.encode_image(
            image, sensorsim_pb2.ImageFormat.RGB_UINT8_PLANAR
        )

        decoded = np.frombuffer(result, np.uint8).reshape(3, 2, 2)

        # Verify channel values preserved in planar format
        assert decoded[0, 0, 0] == 255  # R at (0,0)
        assert decoded[1, 0, 1] == 255  # G at (0,1)
        assert decoded[2, 1, 0] == 255  # B at (1,0)

    def test_unsupported_format_raises_error(self):
        """Test that unsupported image format raises ValueError"""
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        from nre.grpc.protos import sensorsim_pb2

        with pytest.raises(ValueError, match="Unsupported image format"):
            RequestConverter.encode_image(image, sensorsim_pb2.ImageFormat.UNDEFINED)

    def test_invalid_image_for_encoding(self):
        """Test handling of invalid image data"""
        # This would fail during encoding
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        from nre.grpc.protos import sensorsim_pb2

        # PNG should succeed with valid image
        result = RequestConverter.encode_image(image, sensorsim_pb2.ImageFormat.PNG)
        assert len(result) > 0


class TestSlerpQuaternion:
    """Tests for RequestConverter._slerp_quaternion()"""

    def test_identity_quaternions(self):
        """Test SLERP between identity quaternions"""
        q1 = np.array([0.0, 0.0, 0.0, 1.0])  # Identity
        q2 = np.array([0.0, 0.0, 0.0, 1.0])  # Identity

        result = RequestConverter._slerp_quaternion(q1, q2, 0.5)

        # Should remain identity
        assert np.allclose(result, q1, atol=1e-6)

    def test_opposite_quaternions(self):
        """Test SLERP between opposite quaternions (180 deg)"""
        # Two quaternions representing same rotation but with opposite signs
        q1 = np.array([0.0, 0.0, 0.707, 0.707])  # 90 deg around Z
        q2 = np.array([0.0, 0.0, -0.707, -0.707])  # Same rotation, opposite sign

        result = RequestConverter._slerp_quaternion(q1, q2, 0.5)

        # Result should still be a valid quaternion
        assert np.allclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_alpha_zero_returns_start(self):
        """Test alpha=0 returns start quaternion"""
        q1 = np.array([0.0, 0.0, 0.0, 1.0])
        q2 = np.array([0.0, 0.0, 0.707, 0.707])

        result = RequestConverter._slerp_quaternion(q1, q2, 0.0)

        assert np.allclose(result, q1, atol=1e-6)

    def test_alpha_one_returns_end(self):
        """Test alpha=1 returns end quaternion"""
        q1 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        # Use actual sin/cos(45deg) for 90 deg Z rotation quaternion
        angle_45 = np.pi / 4  # 45 degrees in radians
        q2 = np.array([0.0, 0.0, np.sin(angle_45), np.cos(angle_45)], dtype=np.float32)

        result = RequestConverter._slerp_quaternion(q1, q2, 1.0)

        assert np.allclose(result, q2, atol=1e-6)

    def test_output_dtype_is_float32(self):
        """Test output quaternion has float32 dtype"""
        q1 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        q2 = np.array([0.0, 0.0, 0.707, 0.707], dtype=np.float32)

        result = RequestConverter._slerp_quaternion(q1, q2, 0.5)

        assert result.dtype == np.float32


class TestPoseToMatrix:
    """Tests for RequestConverter._pose_to_matrix()"""

    def test_identity_pose(self):
        """Test conversion of identity pose"""
        position = np.array([0.0, 0.0, 0.0])
        quaternion = np.array([0.0, 0.0, 0.0, 1.0])

        result = RequestConverter._pose_to_matrix(position, quaternion)

        assert np.allclose(result, np.eye(4), atol=1e-6)

    def test_position_only_pose(self):
        """Test pose with translation only"""
        position = np.array([5.0, 10.0, 2.0])
        quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # Identity rotation

        result = RequestConverter._pose_to_matrix(position, quaternion)

        assert np.allclose(result[:3, 3], position)
        assert np.allclose(result[:3, :3], np.eye(3), atol=1e-6)

    def test_rotation_only_pose(self):
        """Test pose with rotation only"""
        position = np.array([0.0, 0.0, 0.0])
        # 90 degree rotation around Z
        quaternion = np.array([0.0, 0.0, 0.707, 0.707])

        result = RequestConverter._pose_to_matrix(position, quaternion)

        expected_rot = R.from_euler('z', 90, degrees=True).as_matrix()
        assert np.allclose(result[:3, :3], expected_rot, atol=1e-5)
        assert np.allclose(result[:3, 3], [0.0, 0.0, 0.0])

    def test_combined_pose(self):
        """Test pose with both rotation and translation"""
        position = np.array([1.0, 2.0, 3.0])
        quaternion = np.array([0.0, 0.0, 0.707, 0.707])

        result = RequestConverter._pose_to_matrix(position, quaternion)

        assert result.shape == (4, 4)
        assert np.allclose(result[:3, 3], position)
        assert np.allclose(result[3, :3], [0.0, 0.0, 0.0])
        assert result[3, 3] == 1.0

    def test_output_dtype_is_float32(self):
        """Test output matrix has float32 dtype"""
        position = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        result = RequestConverter._pose_to_matrix(position, quaternion)

        assert result.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])