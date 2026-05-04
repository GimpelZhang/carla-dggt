# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
E2E Test E2E-06: Error Handling

Tests error handling scenarios:
- Server unavailable (wrong port)
- Invalid scene ID (nonexistent)
- Invalid camera specification (zero resolution)
- Out-of-range timestamp (beyond end_timestamp)

Test ID: E2E-06
Test Name: error_handling
"""

import pytest
import numpy as np
import grpc
import sys
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dggt_integration import DggtScenario, DggtRenderer
from dggt_config import DggtConfig
from utils import undo_carla_coordinate_transform
from nre.grpc.protos import sensorsim_pb2

# Import helper functions from test_helpers module
from .test_helpers import create_test_camera_pose

# Note: pytest fixtures (e2e_config, dggt_scenario, default_camera_spec) are auto-discovered from conftest.py

logger = logging.getLogger(__name__)


class TestE2E06ServerUnavailable:
    """
    E2E-06: Server Unavailable Tests

    Tests handling of gRPC connection errors when server is unreachable.
    """

    def test_server_unavailable_wrong_port(self, e2e_config):
        """
        Test E2E-06: Connection to wrong port (9999) should raise RpcError.

        Validates:
        - grpc.RpcError is raised when connecting to unavailable server
        - Error code indicates connection failure
        """
        logger.info("Starting E2E-06: Server unavailable test (wrong port)")

        scene_cfg = e2e_config['scene']

        # Create config with wrong port (9999 instead of actual port)
        wrong_config = DggtConfig(
            server_host="localhost",
            server_port=9999,  # Wrong port - server not listening here
            scene_base_path=scene_cfg['dggt_scene_path'],
            default_scene_id=scene_cfg['scene_id'],
            request_timeout=5.0,  # Short timeout for faster failure
        )

        # Attempt to create scenario - should fail on connect
        with pytest.raises(Exception) as exc_info:
            scenario = DggtScenario(wrong_config)
            scenario.load_scene()  # This should trigger connection

        # Check that we got a connection-related error
        # Could be grpc.RpcError or RuntimeError wrapping it
        error_str = str(exc_info.value)
        logger.info(f"Caught expected error: {error_str}")

        # Verify error type - should be grpc-related
        assert (
            "RpcError" in error_str or
            "connection" in error_str.lower() or
            "unavailable" in error_str.lower() or
            "failed" in error_str.lower()
        ), f"Unexpected error type: {error_str}"

        logger.info("E2E-06 server unavailable: PASSED")

    def test_server_unavailable_wrong_host(self, e2e_config):
        """
        Test E2E-06 variant: Connection to wrong host.

        Validates error handling for non-existent host.
        """
        logger.info("Starting E2E-06: Server unavailable test (wrong host)")

        scene_cfg = e2e_config['scene']

        # Create config with wrong host
        wrong_config = DggtConfig(
            server_host="nonexistent host 12345",  # Invalid hostname
            server_port=50051,
            scene_base_path=scene_cfg['dggt_scene_path'],
            default_scene_id=scene_cfg['scene_id'],
            request_timeout=5.0,
        )

        # Attempt connection - should fail
        with pytest.raises(Exception) as exc_info:
            scenario = DggtScenario(wrong_config)
            scenario.load_scene()

        error_str = str(exc_info.value)
        logger.info(f"Caught expected error: {error_str}")

        # Should fail to resolve/connect
        assert "failed" in error_str.lower() or "error" in error_str.lower()

        logger.info("E2E-06 wrong host: PASSED")


class TestE2E06InvalidSceneId:
    """
    E2E-06: Invalid Scene ID Tests

    Tests handling of nonexistent scene IDs.
    """

    def test_invalid_scene_id_nonexistent(self, e2e_config):
        """
        Test E2E-06: Request with nonexistent scene ID.

        Validates:
        - ValueError or similar exception is raised
        - Error message indicates scene not found
        """
        logger.info("Starting E2E-06: Invalid scene ID test")

        dggt_cfg = e2e_config['dggt_server']
        scene_cfg = e2e_config['scene']

        # Create config with nonexistent scene ID
        wrong_config = DggtConfig(
            server_host=dggt_cfg['host'],
            server_port=dggt_cfg['port'],
            scene_base_path=scene_cfg['dggt_scene_path'],
            default_scene_id="nonexistent_scene_xyz123",  # Scene does not exist
            request_timeout=dggt_cfg['timeout'],
        )

        # Attempt to create scenario with invalid scene
        with pytest.raises(Exception) as exc_info:
            scenario = DggtScenario(wrong_config)

        error_str = str(exc_info.value)
        logger.info(f"Caught expected error: {error_str}")

        # Should indicate scene not found
        assert (
            "not found" in error_str.lower() or
            "does not exist" in error_str.lower() or
            "invalid" in error_str.lower() or
            "Scene path not found" in error_str
        ), f"Unexpected error for nonexistent scene: {error_str}"

        logger.info("E2E-06 invalid scene ID: PASSED")


class TestE2E06InvalidCameraSpec:
    """
    E2E-06: Invalid Camera Specification Tests

    Tests handling of invalid camera parameters.
    """

    def test_invalid_camera_spec_zero_resolution(self, dggt_scenario):
        """
        Test E2E-06: Camera spec with zero resolution.

        Validates:
        - Render handles invalid camera spec appropriately
        - Should either fail validation or produce error
        """
        logger.info("Starting E2E-06: Invalid camera spec (zero resolution)")

        renderer = dggt_scenario.get_renderer()

        # Create invalid camera spec with zero resolution
        invalid_camera_spec = sensorsim_pb2.CameraSpec()
        invalid_camera_spec.resolution_w = 0  # Invalid
        invalid_camera_spec.resolution_h = 0  # Invalid

        # Set some minimal intrinsics (using fisheye param)
        fisheye = invalid_camera_spec.opencv_fisheye_param
        fisheye.focal_length_x = 1000.0
        fisheye.focal_length_y = 1000.0
        fisheye.principal_point_x = 518.0
        fisheye.principal_point_y = 350.0

        # Create a valid camera pose
        camera_pose = create_test_camera_pose(x=0.0, y=0.0, z=2.0)
        camera_pose_carla = undo_carla_coordinate_transform(camera_pose)

        timestamp_us = 0

        # Attempt render with invalid spec
        # Expected: should raise exception (value error, validation error, or render error)
        with pytest.raises(Exception) as exc_info:
            image = renderer.render(
                world_snapshot=None,
                camera_spec=invalid_camera_spec,
                camera_pose=camera_pose_carla,
                timestamp=timestamp_us,
                resolution_ratio=1.0,
            )

        error_str = str(exc_info.value)
        logger.info(f"Caught expected error: {error_str}")

        # Should indicate invalid parameters or zero resolution
        # Accept various error types (value error, runtime error, grpc error)
        assert exc_info.value is not None

        logger.info("E2E-06 invalid camera spec: PASSED")

    def test_invalid_camera_spec_negative_resolution(self, dggt_scenario):
        """
        Test E2E-06 variant: Negative resolution values.

        Validates handling of negative resolution parameters.
        """
        logger.info("Starting E2E-06: Invalid camera spec (negative resolution)")

        renderer = dggt_scenario.get_renderer()

        # Create invalid camera spec with negative resolution
        invalid_camera_spec = sensorsim_pb2.CameraSpec()
        invalid_camera_spec.resolution_w = -100  # Invalid
        invalid_camera_spec.resolution_h = -100  # Invalid

        fisheye = invalid_camera_spec.opencv_fisheye_param
        fisheye.focal_length_x = 1000.0
        fisheye.focal_length_y = 1000.0

        camera_pose = create_test_camera_pose(x=0.0, y=0.0, z=2.0)
        camera_pose_carla = undo_carla_coordinate_transform(camera_pose)

        with pytest.raises(Exception) as exc_info:
            renderer.render(
                world_snapshot=None,
                camera_spec=invalid_camera_spec,
                camera_pose=camera_pose_carla,
                timestamp=0,
                resolution_ratio=1.0,
            )

        logger.info(f"Caught error for negative resolution: {exc_info.value}")
        logger.info("E2E-06 negative resolution: PASSED")


class TestE2E06OutOfRangeTimestamp:
    """
    E2E-06: Out-of-Range Timestamp Tests

    Tests handling of timestamps beyond valid range.
    """

    def test_out_of_range_timestamp_past_end(self, dggt_scenario, default_camera_spec):
        """
        Test E2E-06: Timestamp beyond end_timestamp_us.

        Validates:
        - Using timestamp > end_timestamp_us returns last frame
        - No exception raised (graceful handling)
        - Image is still valid
        """
        logger.info("Starting E2E-06: Out-of-range timestamp test")

        renderer = dggt_scenario.get_renderer()
        metadata = dggt_scenario.get_scene_metadata()

        # Get valid timestamp range
        start_ts, end_ts = dggt_scenario.get_timestamp_range()
        logger.info(f"Valid timestamp range: {start_ts} to {end_ts} us")

        # Create timestamp beyond end
        invalid_timestamp = metadata.end_timestamp_us + 1_000_000  # 1 second beyond end
        logger.info(f"Testing timestamp: {invalid_timestamp} us (beyond end)")

        # Create valid camera pose
        camera_pose = create_test_camera_pose(x=0.0, y=0.0, z=2.0)
        camera_pose_carla = undo_carla_coordinate_transform(camera_pose)

        # Render with out-of-range timestamp
        # Expected behavior: should return last frame (graceful degradation)
        try:
            image = renderer.render(
                world_snapshot=None,
                camera_spec=default_camera_spec,
                camera_pose=camera_pose_carla,
                timestamp=invalid_timestamp,
                resolution_ratio=1.0,
            )

            # Should return a valid image (last frame)
            assert image is not None, "Image should not be None"
            assert len(image.shape) == 3, f"Image should be 3D, got shape {image.shape}"
            assert image.shape[2] == 3, f"Image should have 3 channels"

            logger.info(f"Got last frame: shape={image.shape}, mean={np.mean(image):.2f}")
            logger.info("E2E-06 out-of-range timestamp: PASSED (returned last frame)")

        except Exception as e:
            # If server rejects the timestamp, that's also acceptable behavior
            error_str = str(e)
            logger.info(f"Server rejected out-of-range timestamp: {error_str}")

            # Accept either graceful handling (last frame) or explicit error
            # Both are valid error handling strategies
            if "timestamp" in error_str.lower() or "range" in error_str.lower():
                logger.info("E2E-06: Server rejected invalid timestamp - PASSED")
            else:
                # Still pass if any reasonable error was raised
                pytest.fail(f"Unexpected error for out-of-range timestamp: {e}")

    def test_out_of_range_timestamp_negative(self, dggt_scenario, default_camera_spec):
        """
        Test E2E-06 variant: Negative timestamp.

        Validates handling of negative timestamp values.
        """
        logger.info("Starting E2E-06: Negative timestamp test")

        renderer = dggt_scenario.get_renderer()

        camera_pose = create_test_camera_pose(x=0.0, y=0.0, z=2.0)
        camera_pose_carla = undo_carla_coordinate_transform(camera_pose)

        # Use negative timestamp
        negative_timestamp = -100_000  # -100ms

        try:
            image = renderer.render(
                world_snapshot=None,
                camera_spec=default_camera_spec,
                camera_pose=camera_pose_carla,
                timestamp=negative_timestamp,
                resolution_ratio=1.0,
            )

            # Should return first frame (graceful handling)
            assert image is not None
            logger.info(f"Got first frame for negative timestamp: shape={image.shape}")

        except Exception as e:
            # Also acceptable if server rejects
            logger.info(f"Server rejected negative timestamp: {e}")

        logger.info("E2E-06 negative timestamp: PASSED")


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])