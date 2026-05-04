# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unit tests for Dynamic Objects Conversion

Tests the convert_dynamic_objects method from RequestConverter:
- Single object conversion
- Multiple objects conversion
- Pose interpolation in objects
- Empty objects list handling
- Unknown track_id handling

CRITICAL: Coordinate transform uses utils.py undo_carla_coordinate_transform.
Reference: utils.py line 146: dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
from unittest.mock import Mock

# Import the module under test
try:
    from dggt_server.request_converter import RequestConverter
    from dggt_server.scene_metadata import TrackIDMapping
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from dggt_server.request_converter import RequestConverter
    from dggt_server.scene_metadata import TrackIDMapping

# CRITICAL: Import coordinate transform from utils.py, NEVER reimplement!
try:
    from utils import undo_carla_coordinate_transform
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from utils import undo_carla_coordinate_transform

# Import fixtures helper
from tests.conftest import create_mock_pose, create_mock_pose_pair, assert_valid_transformation_matrix

# Import protobuf types
from nre.grpc.protos import sensorsim_pb2, common_pb2


def create_dynamic_object(track_id: str, start_pose=None, end_pose=None):
    """Helper to create DynamicObject protobuf"""
    dyn_obj = sensorsim_pb2.DynamicObject()
    dyn_obj.track_id = track_id

    # Set start pose
    if start_pose is None:
        start_pose = create_mock_pose()
    dyn_obj.pose_pair.start_pose.vec.x = start_pose.vec.x
    dyn_obj.pose_pair.start_pose.vec.y = start_pose.vec.y
    dyn_obj.pose_pair.start_pose.vec.z = start_pose.vec.z
    dyn_obj.pose_pair.start_pose.quat.w = start_pose.quat.w
    dyn_obj.pose_pair.start_pose.quat.x = start_pose.quat.x
    dyn_obj.pose_pair.start_pose.quat.y = start_pose.quat.y
    dyn_obj.pose_pair.start_pose.quat.z = start_pose.quat.z

    # Set end pose
    if end_pose is None:
        end_pose = create_mock_pose()
    dyn_obj.pose_pair.end_pose.vec.x = end_pose.vec.x
    dyn_obj.pose_pair.end_pose.vec.y = end_pose.vec.y
    dyn_obj.pose_pair.end_pose.vec.z = end_pose.vec.z
    dyn_obj.pose_pair.end_pose.quat.w = end_pose.quat.w
    dyn_obj.pose_pair.end_pose.quat.x = end_pose.quat.x
    dyn_obj.pose_pair.end_pose.quat.y = end_pose.quat.y
    dyn_obj.pose_pair.end_pose.quat.z = end_pose.quat.z

    return dyn_obj


@pytest.mark.unit
class TestDynamicObjectsConversion:
    """Tests for RequestConverter.convert_dynamic_objects()"""

    def test_single_object_conversion(self):
        """Test conversion of a single dynamic object"""
        # Create track mapping
        track_mapping = TrackIDMapping(auto_prefix="dggt_obj")
        track_mapping.register("vehicle_0", 0)

        # Create identity transform
        t_carla_dggt = np.eye(4, dtype=np.float32)

        # Create dynamic object
        dyn_obj = create_dynamic_object("vehicle_0")

        # Convert
        result = RequestConverter.convert_dynamic_objects(
            [dyn_obj], track_mapping, t_carla_dggt, alpha=0.5
        )

        # Verify result
        assert len(result) == 1
        object_id, pose_dggt = result[0]
        assert object_id == 0
        assert pose_dggt.shape == (4, 4)
        # Note: dtype may vary based on t_carla_dggt dtype (utils.py uses np.eye(4) float64 default)

    def test_multiple_objects_conversion(self):
        """Test conversion of multiple dynamic objects"""
        # Create track mapping
        track_mapping = TrackIDMapping(auto_prefix="dggt_obj")
        track_mapping.register("vehicle_0", 0)
        track_mapping.register("vehicle_1", 1)
        track_mapping.register("vehicle_2", 2)

        # Create identity transform
        t_carla_dggt = np.eye(4, dtype=np.float32)

        # Create dynamic objects
        objects = [
            create_dynamic_object("vehicle_0"),
            create_dynamic_object("vehicle_1"),
            create_dynamic_object("vehicle_2"),
        ]

        # Convert
        result = RequestConverter.convert_dynamic_objects(
            objects, track_mapping, t_carla_dggt, alpha=0.5
        )

        # Verify result
        assert len(result) == 3

        # Check all objects are present
        object_ids = [r[0] for r in result]
        assert sorted(object_ids) == [0, 1, 2]

        # Check all poses are valid 4x4 matrices
        for object_id, pose_dggt in result:
            assert pose_dggt.shape == (4, 4)
            # Note: dtype may vary based on implementation

    def test_pose_interpolation_in_objects(self):
        """Test that pose interpolation is applied to dynamic objects"""
        # Create track mapping
        track_mapping = TrackIDMapping(auto_prefix="dggt_obj")
        track_mapping.register("vehicle_0", 0)

        # Create identity transform
        t_carla_dggt = np.eye(4, dtype=np.float32)

        # Create object with different start/end poses
        start_pose = create_mock_pose(x=0.0, y=0.0, z=0.0)
        end_pose = create_mock_pose(x=10.0, y=5.0, z=2.0)
        dyn_obj = create_dynamic_object("vehicle_0", start_pose, end_pose)

        # Convert with alpha=0.5 (should be at midpoint)
        result = RequestConverter.convert_dynamic_objects(
            [dyn_obj], track_mapping, t_carla_dggt, alpha=0.5
        )

        assert len(result) == 1
        object_id, pose_dggt = result[0]

        # The pose should be interpolated at midpoint
        # After coordinate transform, position is [x, -y, z] for identity t_carla_dggt
        # Position interpolation: [5.0, 2.5, 1.0]
        # After undo_carla_coordinate_transform: position becomes [5.0, -2.5, 1.0]
        expected_pos = np.array([5.0, -2.5, 1.0])
        assert np.allclose(pose_dggt[:3, 3], expected_pos, atol=1e-6)

    def test_pose_interpolation_with_different_alpha(self):
        """Test pose interpolation with different alpha values"""
        # Create track mapping
        track_mapping = TrackIDMapping(auto_prefix="dggt_obj")
        track_mapping.register("vehicle_0", 0)

        # Create identity transform
        t_carla_dggt = np.eye(4, dtype=np.float32)

        # Create object with different start/end poses
        start_pose = create_mock_pose(x=0.0, y=0.0, z=0.0)
        end_pose = create_mock_pose(x=10.0, y=5.0, z=2.0)
        dyn_obj = create_dynamic_object("vehicle_0", start_pose, end_pose)

        # Test alpha=0.25
        result_025 = RequestConverter.convert_dynamic_objects(
            [dyn_obj], track_mapping, t_carla_dggt, alpha=0.25
        )
        # Position at alpha=0.25: [2.5, 1.25, 0.5]
        # After undo_carla_coordinate_transform: [2.5, -1.25, 0.5]
        assert np.allclose(result_025[0][1][:3, 3], [2.5, -1.25, 0.5], atol=1e-6)

        # Test alpha=0.75
        result_075 = RequestConverter.convert_dynamic_objects(
            [dyn_obj], track_mapping, t_carla_dggt, alpha=0.75
        )
        # Position at alpha=0.75: [7.5, 3.75, 1.5]
        # After undo_carla_coordinate_transform: [7.5, -3.75, 1.5]
        assert np.allclose(result_075[0][1][:3, 3], [7.5, -3.75, 1.5], atol=1e-6)

    def test_empty_objects_list_handling(self):
        """Test handling of empty dynamic objects list"""
        # Create track mapping
        track_mapping = TrackIDMapping(auto_prefix="dggt_obj")

        # Create identity transform
        t_carla_dggt = np.eye(4, dtype=np.float32)

        # Convert empty list
        result = RequestConverter.convert_dynamic_objects(
            [], track_mapping, t_carla_dggt, alpha=0.5
        )

        # Result should be empty list
        assert result == []
        assert len(result) == 0

    def test_unknown_track_id_handling(self):
        """Test handling of unknown track_id (should be skipped with warning)"""
        # Create track mapping with limited mappings
        track_mapping = TrackIDMapping(auto_prefix="dggt_obj")
        track_mapping.register("vehicle_0", 0)

        # Create identity transform
        t_carla_dggt = np.eye(4, dtype=np.float32)

        # Create objects: one known, one unknown
        known_obj = create_dynamic_object("vehicle_0")
        unknown_obj = create_dynamic_object("unknown_vehicle_xyz")  # No numeric component

        # Convert
        result = RequestConverter.convert_dynamic_objects(
            [known_obj, unknown_obj], track_mapping, t_carla_dggt, alpha=0.5
        )

        # Only the known object should be converted
        assert len(result) == 1
        assert result[0][0] == 0

    def test_track_id_with_numeric_extraction(self):
        """Test track_id conversion using numeric extraction fallback"""
        # Create track mapping without explicit mapping for "car_123"
        track_mapping = TrackIDMapping(auto_prefix="dggt_obj")
        # No explicit mapping registered

        # Create identity transform
        t_carla_dggt = np.eye(4, dtype=np.float32)

        # Create object with track_id containing number
        dyn_obj = create_dynamic_object("car_123")

        # Convert - should extract 123 from track_id
        result = RequestConverter.convert_dynamic_objects(
            [dyn_obj], track_mapping, t_carla_dggt, alpha=0.5
        )

        assert len(result) == 1
        object_id, pose_dggt = result[0]
        assert object_id == 123

    def test_coordinate_transform_chain(self):
        """Test that coordinate transform chain matches utils.py pattern"""
        # Create track mapping
        track_mapping = TrackIDMapping(auto_prefix="dggt_obj")
        track_mapping.register("vehicle_0", 0)

        # Create non-identity CARLA to DGGT transform
        t_carla_dggt = np.eye(4, dtype=np.float32)
        t_carla_dggt[:3, 3] = [100.0, 50.0, 10.0]  # Translation offset

        # Create object with known pose
        start_pose = create_mock_pose(x=0.0, y=0.0, z=0.0)
        end_pose = create_mock_pose(x=10.0, y=5.0, z=2.0)
        dyn_obj = create_dynamic_object("vehicle_0", start_pose, end_pose)

        # Convert
        result = RequestConverter.convert_dynamic_objects(
            [dyn_obj], track_mapping, t_carla_dggt, alpha=0.5
        )

        assert len(result) == 1
        object_id, pose_dggt = result[0]

        # Verify the transform chain was applied:
        # dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(interpolated_pose)
        # Interpolated position: [5.0, 2.5, 1.0]
        # undo_carla_coordinate_transform: [5.0, -2.5, 1.0]
        # t_carla_dggt translation added: [105.0, 47.5, 11.0]
        expected_pos = np.array([105.0, 47.5, 11.0])
        assert np.allclose(pose_dggt[:3, 3], expected_pos, atol=1e-6)

    def test_rotation_interpolation_in_objects(self):
        """Test that rotation interpolation (SLERP) is applied correctly"""
        # Create track mapping
        track_mapping = TrackIDMapping(auto_prefix="dggt_obj")
        track_mapping.register("vehicle_0", 0)

        # Create identity transform
        t_carla_dggt = np.eye(4, dtype=np.float32)

        # Create object with rotation: identity to 90 deg Z rotation
        angle_45 = np.pi / 4
        start_pose = create_mock_pose(qw=1.0)  # Identity
        end_pose = create_mock_pose(qw=np.cos(angle_45), qz=np.sin(angle_45))  # 90 deg Z

        dyn_obj = create_dynamic_object("vehicle_0", start_pose, end_pose)

        # Convert with alpha=0.5 (should be ~45 deg rotation)
        result = RequestConverter.convert_dynamic_objects(
            [dyn_obj], track_mapping, t_carla_dggt, alpha=0.5
        )

        assert len(result) == 1
        object_id, pose_dggt = result[0]

        # Verify the rotation matrix is valid (orthogonal, proper shape)
        assert pose_dggt.shape == (4, 4)
        rot = pose_dggt[:3, :3]
        # Check orthogonality
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-5)
        # Check determinant is ~1
        assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-5)


@pytest.mark.unit
class TestDynamicObjectsEdgeCases:
    """Edge case tests for dynamic objects conversion"""

    def test_object_with_negative_position(self):
        """Test object with negative position coordinates"""
        track_mapping = TrackIDMapping(auto_prefix="dggt_obj")
        track_mapping.register("vehicle_0", 0)

        t_carla_dggt = np.eye(4, dtype=np.float32)

        start_pose = create_mock_pose(x=-5.0, y=-10.0, z=-2.0)
        end_pose = create_mock_pose(x=5.0, y=10.0, z=2.0)
        dyn_obj = create_dynamic_object("vehicle_0", start_pose, end_pose)

        result = RequestConverter.convert_dynamic_objects(
            [dyn_obj], track_mapping, t_carla_dggt, alpha=0.5
        )

        # Interpolated: [0.0, 0.0, 0.0]
        # After undo: [0.0, 0.0, 0.0]
        assert np.allclose(result[0][1][:3, 3], [0.0, 0.0, 0.0], atol=1e-6)

    def test_object_with_large_position(self):
        """Test object with large position values"""
        track_mapping = TrackIDMapping(auto_prefix="dggt_obj")
        track_mapping.register("vehicle_0", 0)

        t_carla_dggt = np.eye(4, dtype=np.float32)

        start_pose = create_mock_pose(x=0.0, y=0.0, z=0.0)
        end_pose = create_mock_pose(x=1000.0, y=500.0, z=200.0)
        dyn_obj = create_dynamic_object("vehicle_0", start_pose, end_pose)

        result = RequestConverter.convert_dynamic_objects(
            [dyn_obj], track_mapping, t_carla_dggt, alpha=0.5
        )

        # Interpolated: [500.0, 250.0, 100.0]
        # After undo: [500.0, -250.0, 100.0]
        assert np.allclose(result[0][1][:3, 3], [500.0, -250.0, 100.0], atol=1e-6)

    def test_explicit_mapping_priority_over_numeric(self):
        """Test that explicit mapping takes priority over numeric extraction"""
        # Create mapping with explicit entry that differs from numeric
        track_mapping = TrackIDMapping(auto_prefix="dggt_obj")
        track_mapping.register("vehicle_999", 42)  # Explicit: 999 -> 42

        t_carla_dggt = np.eye(4, dtype=np.float32)

        dyn_obj = create_dynamic_object("vehicle_999")

        result = RequestConverter.convert_dynamic_objects(
            [dyn_obj], track_mapping, t_carla_dggt, alpha=0.5
        )

        # Should use explicit mapping (42), not numeric extraction (999)
        assert result[0][0] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])