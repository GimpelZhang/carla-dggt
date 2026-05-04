# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Pytest Configuration and Fixtures

Provides shared fixtures for all test modules.

Fixtures:
- TEST_DATA_DIR: Path to test data directory
- test_scene_path: Path to test scene (scene_mini)
- test_scene_id: Test scene identifier
- sample_ego_frame: Sample ego pose frame data
- sample_dynamic_objects: Sample dynamic objects data
- Protobuf fixtures: sample_pose, sample_pose_pair, sample_camera_spec, etc.
"""

import pytest
import numpy as np
import os
from pathlib import Path
from unittest.mock import Mock

# Import protobuf types
from nre.grpc.protos import common_pb2, sensorsim_pb2

# Import DGGT server modules for fixtures
try:
    from dggt_server.scene_metadata import DGGTSceneMetadata, TrackIDMapping
    from dggt_server.frame_index_mapper import FrameIndexMapper
except ImportError:
    # Add parent directory to path for imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dggt_server.scene_metadata import DGGTSceneMetadata, TrackIDMapping
    from dggt_server.frame_index_mapper import FrameIndexMapper


# ============================================================================
# Directory and Path Fixtures
# ============================================================================

@pytest.fixture
def TEST_DATA_DIR():
    """Path to test data directory"""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def test_scene_path(TEST_DATA_DIR):
    """Path to test scene directory (scene_mini)"""
    return TEST_DATA_DIR / "scene_mini"


@pytest.fixture
def test_scene_id():
    """Test scene identifier"""
    return "scene_mini"


# ============================================================================
# Scene Metadata Fixtures
# ============================================================================

@pytest.fixture
def sample_scene_metadata():
    """Sample DGGTSceneMetadata for testing"""
    return DGGTSceneMetadata(
        scene_id="scene_mini",
        scene_path="/test/path/scene_mini",
        num_frames=5,
        fps=10.0,
        start_timestamp_us=0,
        end_timestamp_us=400_000,  # 4 frames at 10Hz (0, 100k, 200k, 300k, 400k)
        camera_width=1036,
        camera_height=700,
        intrinsic_matrix=np.array([
            [1000.0, 0.0, 518.0],
            [0.0, 1000.0, 350.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32),
        intrinsics_vary=False,
        has_static_scene=True,
        has_sky_scene=True,
        dynamic_object_ids=[0, 1, 2],
        static_scene_size_mb=50.0,
        total_size_mb=60.0
    )


@pytest.fixture
def sample_track_mapping():
    """Sample TrackIDMapping for testing"""
    mapping = TrackIDMapping(auto_prefix="dggt_obj")
    mapping.register("vehicle_0", 0)
    mapping.register("vehicle_1", 1)
    mapping.register("vehicle_2", 2)
    return mapping


@pytest.fixture
def sample_frame_index_mapper(sample_scene_metadata):
    """Sample FrameIndexMapper for testing"""
    return FrameIndexMapper(sample_scene_metadata)


# ============================================================================
# Ego Frame and Dynamic Objects Fixtures
# ============================================================================

@pytest.fixture
def sample_ego_frame():
    """Sample ego pose frame data (4x4 transformation matrix)"""
    # Identity pose at origin
    return np.eye(4, dtype=np.float32)


@pytest.fixture
def sample_ego_frame_with_pose():
    """Sample ego pose with non-identity transformation"""
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [10.0, 5.0, 1.5]  # Translation
    # Add 45 degree rotation around Z
    from scipy.spatial.transform import Rotation as R
    rotation = R.from_euler('z', 45, degrees=True).as_matrix()
    pose[:3, :3] = rotation.astype(np.float32)
    return pose


@pytest.fixture
def sample_dynamic_objects():
    """Sample dynamic objects data for testing"""
    return [
        {
            "object_id": 0,
            "track_id": "vehicle_0",
            "pose": np.eye(4, dtype=np.float32),
            "dimensions": np.array([4.5, 2.0, 1.5], dtype=np.float32)
        },
        {
            "object_id": 1,
            "track_id": "vehicle_1",
            "pose": np.eye(4, dtype=np.float32),
            "dimensions": np.array([4.0, 1.8, 1.4], dtype=np.float32)
        }
    ]


# ============================================================================
# Protobuf Fixtures
# ============================================================================

@pytest.fixture
def sample_pose():
    """Sample Pose protobuf (identity pose at origin)"""
    pose = common_pb2.Pose()
    pose.vec.x = 0.0
    pose.vec.y = 0.0
    pose.vec.z = 0.0
    pose.quat.w = 1.0  # Identity quaternion
    pose.quat.x = 0.0
    pose.quat.y = 0.0
    pose.quat.z = 0.0
    return pose


@pytest.fixture
def sample_pose_with_translation():
    """Sample Pose protobuf with translation"""
    pose = common_pb2.Pose()
    pose.vec.x = 10.0
    pose.vec.y = 5.0
    pose.vec.z = 2.0
    pose.quat.w = 1.0
    pose.quat.x = 0.0
    pose.quat.y = 0.0
    pose.quat.z = 0.0
    return pose


@pytest.fixture
def sample_pose_with_rotation():
    """Sample Pose protobuf with 90 degree rotation around Z"""
    pose = common_pb2.Pose()
    pose.vec.x = 0.0
    pose.vec.y = 0.0
    pose.vec.z = 0.0
    # Quaternion for 90 deg rotation around Z: w=cos(45deg)=0.707, z=sin(45deg)=0.707
    import math
    angle_45 = math.pi / 4
    pose.quat.w = math.cos(angle_45)
    pose.quat.x = 0.0
    pose.quat.y = 0.0
    pose.quat.z = math.sin(angle_45)
    return pose


@pytest.fixture
def sample_pose_pair():
    """Sample PosePair protobuf (start and end poses for interpolation)"""
    pose_pair = sensorsim_pb2.PosePair()

    # Start pose at origin
    pose_pair.start_pose.vec.x = 0.0
    pose_pair.start_pose.vec.y = 0.0
    pose_pair.start_pose.vec.z = 0.0
    pose_pair.start_pose.quat.w = 1.0
    pose_pair.start_pose.quat.x = 0.0
    pose_pair.start_pose.quat.y = 0.0
    pose_pair.start_pose.quat.z = 0.0

    # End pose with translation and rotation
    pose_pair.end_pose.vec.x = 10.0
    pose_pair.end_pose.vec.y = 5.0
    pose_pair.end_pose.vec.z = 2.0
    pose_pair.end_pose.quat.w = 0.707
    pose_pair.end_pose.quat.x = 0.0
    pose_pair.end_pose.quat.y = 0.0
    pose_pair.end_pose.quat.z = 0.707

    return pose_pair


@pytest.fixture
def sample_pose_pair_identity():
    """Sample PosePair with identical start and end poses"""
    pose_pair = sensorsim_pb2.PosePair()

    # Both poses are identity
    for pose_attr in ['start_pose', 'end_pose']:
        pose = getattr(pose_pair, pose_attr)
        pose.vec.x = 0.0
        pose.vec.y = 0.0
        pose.vec.z = 0.0
        pose.quat.w = 1.0
        pose.quat.x = 0.0
        pose.quat.y = 0.0
        pose.quat.z = 0.0

    return pose_pair


@pytest.fixture
def sample_camera_spec():
    """Sample CameraSpec protobuf with OpenCV pinhole parameters"""
    camera_spec = sensorsim_pb2.CameraSpec()
    camera_spec.resolution_w = 1036
    camera_spec.resolution_h = 700

    # Set pinhole camera parameters
    pinhole = camera_spec.opencv_pinhole_param
    pinhole.focal_length_x = 1000.0
    pinhole.focal_length_y = 1000.0
    pinhole.principal_point_x = 518.0
    pinhole.principal_point_y = 350.0

    return camera_spec


@pytest.fixture
def sample_fisheye_camera_spec():
    """Sample CameraSpec protobuf with fisheye parameters"""
    camera_spec = sensorsim_pb2.CameraSpec()
    camera_spec.resolution_w = 1200
    camera_spec.resolution_h = 800

    # Set fisheye camera parameters
    fisheye = camera_spec.opencv_fisheye_param
    fisheye.focal_length_x = 1200.0
    fisheye.focal_length_y = 1200.0
    fisheye.principal_point_x = 600.0
    fisheye.principal_point_y = 400.0
    fisheye.max_angle = 1.57  # ~90 degrees in radians

    return camera_spec


@pytest.fixture
def sample_ftheta_camera_spec():
    """Sample CameraSpec protobuf with f-theta parameters"""
    camera_spec = sensorsim_pb2.CameraSpec()
    camera_spec.resolution_w = 1920
    camera_spec.resolution_h = 1080

    # Set f-theta camera parameters
    ftheta = camera_spec.ftheta_param
    ftheta.principal_point_x = 960.0
    ftheta.principal_point_y = 540.0
    ftheta.max_angle = 2.09  # ~120 degrees in radians
    # Simple polynomial: pixel_dist = focal * angle (approximated)
    ftheta.pixeldist_to_angle_poly.extend([0.0, 1.0 / 960.0])

    return camera_spec


@pytest.fixture
def sample_dynamic_object():
    """Sample DynamicObject protobuf"""
    dyn_obj = sensorsim_pb2.DynamicObject()
    dyn_obj.track_id = "vehicle_0"

    # Set pose pair
    dyn_obj.pose_pair.start_pose.vec.x = 0.0
    dyn_obj.pose_pair.start_pose.vec.y = 0.0
    dyn_obj.pose_pair.start_pose.vec.z = 0.0
    dyn_obj.pose_pair.start_pose.quat.w = 1.0
    dyn_obj.pose_pair.start_pose.quat.x = 0.0
    dyn_obj.pose_pair.start_pose.quat.y = 0.0
    dyn_obj.pose_pair.start_pose.quat.z = 0.0

    dyn_obj.pose_pair.end_pose.vec.x = 10.0
    dyn_obj.pose_pair.end_pose.vec.y = 5.0
    dyn_obj.pose_pair.end_pose.vec.z = 2.0
    dyn_obj.pose_pair.end_pose.quat.w = 1.0
    dyn_obj.pose_pair.end_pose.quat.x = 0.0
    dyn_obj.pose_pair.end_pose.quat.y = 0.0
    dyn_obj.pose_pair.end_pose.quat.z = 0.0

    return dyn_obj


# ============================================================================
# Helper Functions for Tests
# ============================================================================

def create_mock_pose(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    """Helper to create mock Pose object"""
    pose = Mock()
    pose.vec = Mock()
    pose.vec.x = x
    pose.vec.y = y
    pose.vec.z = z
    pose.quat = Mock()
    pose.quat.w = qw
    pose.quat.x = qx
    pose.quat.y = qy
    pose.quat.z = qz
    return pose


def create_mock_pose_pair(start_pose=None, end_pose=None):
    """Helper to create mock PosePair object"""
    pose_pair = Mock()
    pose_pair.start_pose = start_pose or create_mock_pose()
    pose_pair.end_pose = end_pose or create_mock_pose()
    return pose_pair


def assert_valid_transformation_matrix(matrix):
    """Assert that a 4x4 transformation matrix is valid"""
    assert matrix.shape == (4, 4)
    assert matrix.dtype == np.float32
    # Check orthogonality of rotation part
    rot = matrix[:3, :3]
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-5)
    # Check determinant is 1
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-5)
    # Check last row is [0, 0, 0, 1]
    assert np.allclose(matrix[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6)


def assert_valid_intrinsic_matrix(K):
    """Assert that a 3x3 intrinsic matrix is valid"""
    assert K.shape == (3, 3)
    assert K.dtype == np.float32
    # Check structure: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    assert K[0, 1] == 0.0
    assert K[1, 0] == 0.0
    assert K[2, 0] == 0.0
    assert K[2, 1] == 0.0
    assert K[2, 2] == 1.0
    # Check focal lengths are positive
    assert K[0, 0] > 0
    assert K[1, 1] > 0