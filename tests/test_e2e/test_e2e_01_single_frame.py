# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
E2E Test E2E-01: Single Frame Render

Tests basic rendering functionality:
- CARLA client connection
- DGGT scenario loading
- Single frame render request
- Image validation

Test ID: E2E-01
Test Name: single_frame_render
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dggt_integration import DggtScenario, DggtRenderer
from dggt_config import DggtConfig
from utils import undo_carla_coordinate_transform, se3_to_grpc_pose
from nre.grpc.protos import sensorsim_pb2

# Import helper functions from test_helpers module
from .test_helpers import (
    create_test_camera_pose,
    validate_rendered_image,
    save_image,
)

# Note: pytest fixtures (e2e_config, carla_client, dggt_world, dggt_scenario,
# default_camera_spec, test_output_dir) are auto-discovered from conftest.py
# and do not need to be imported explicitly

logger = logging.getLogger(__name__)


class TestE2E01SingleFrame:
    """
    E2E-01: Single Frame Render Test

    Tests basic rendering pipeline from CARLA to DGGT.
    """

    @pytest.fixture(autouse=True)
    def setup(self, dggt_scenario, default_camera_spec, test_output_dir, e2e_config):
        """Setup test fixtures."""
        self.scenario = dggt_scenario
        self.renderer = self.scenario.get_renderer()
        self.camera_spec = default_camera_spec
        self.output_dir = test_output_dir
        self.save_images = e2e_config['test']['save_images']

        logger.info(f"E2E-01 setup complete")

    def test_single_frame_render(self):
        """
        Test E2E-01: Render single frame from default camera.

        Validates:
        - Image is not None
        - Image shape matches camera spec resolution
        - Image dtype is uint8
        - Image mean value > 0 (not all black)
        """
        logger.info("Starting E2E-01: Single frame render test")

        # Step 1: Create camera pose
        # Position camera at height 2m, looking forward
        camera_pose = create_test_camera_pose(
            x=0.0,
            y=0.0,
            z=2.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0
        )

        # Apply CARLA coordinate transform (undo)
        # This matches NurecSensor.on_world_tick pattern
        camera_pose_carla = undo_carla_coordinate_transform(camera_pose)

        logger.info(f"Camera pose created: position [{0.0}, {0.0}, {2.0}]")

        # Step 2: Render frame
        timestamp_us = 0  # Start of timeline

        try:
            image = self.renderer.render(
                world_snapshot=None,  # No dynamic objects
                camera_spec=self.camera_spec,
                camera_pose=camera_pose_carla,
                timestamp=timestamp_us,
                resolution_ratio=1.0,
            )
        except Exception as e:
            pytest.fail(f"Render failed: {e}")

        logger.info(f"Render completed: image shape {image.shape if image is not None else 'None'}")

        # Step 3: Validate image
        # Expected shape from camera spec
        expected_shape = (
            self.camera_spec.resolution_h,
            self.camera_spec.resolution_w,
            3
        )

        # Validate basic properties
        validate_rendered_image(image, expected_shape)

        logger.info(f"Image validation passed: shape={image.shape}, dtype={image.dtype}, mean={np.mean(image):.2f}")

        # Step 4: Save image if configured
        if self.save_images:
            filename = f"e2e_01_single_frame.jpg"
            save_path = save_image(image, self.output_dir, filename)
            logger.info(f"Image saved: {save_path}")

        logger.info("E2E-01: Test PASSED")

    def test_single_frame_with_timestamp(self):
        """
        Test E2E-01 variant: Render at specific timestamp.

        Validates rendering at non-zero timestamp.
        """
        logger.info("Starting E2E-01 variant: Timestamp render")

        # Get timestamp range from scenario
        start_ts, end_ts = self.scenario.get_timestamp_range()

        # Use middle timestamp
        mid_timestamp = (start_ts + end_ts) // 2

        logger.info(f"Using timestamp: {mid_timestamp} us")

        # Create camera pose
        camera_pose = create_test_camera_pose(x=0.0, y=0.0, z=2.0)
        camera_pose_carla = undo_carla_coordinate_transform(camera_pose)

        # Render
        try:
            image = self.renderer.render(
                world_snapshot=None,
                camera_spec=self.camera_spec,
                camera_pose=camera_pose_carla,
                timestamp=mid_timestamp,
                resolution_ratio=1.0,
            )
        except Exception as e:
            pytest.fail(f"Render at timestamp {mid_timestamp} failed: {e}")

        # Validate
        validate_rendered_image(image)

        logger.info(f"E2E-01 timestamp variant: PASSED (mean={np.mean(image):.2f})")


class TestE2E01Connection:
    """
    E2E-01: Connection Tests

    Validates CARLA and DGGT server connectivity.
    """

    def test_carla_connection(self, carla_client):
        """Test CARLA server connection."""
        logger.info("Testing CARLA connection")

        # Get world
        world = carla_client.get_world()
        assert world is not None, "CARLA world is None"

        # Get world settings
        settings = world.get_settings()
        logger.info(f"CARLA world settings: fixed_delta_seconds={settings.fixed_delta_seconds}")

        logger.info("CARLA connection: PASSED")

    def test_dggt_connection(self, dggt_scenario):
        """Test DGGT server connection."""
        logger.info("Testing DGGT connection")

        # Get renderer
        renderer = dggt_scenario.get_renderer()

        # Check available cameras
        cameras = renderer.get_available_cameras()
        assert len(cameras) > 0, "No cameras available in DGGT scenario"

        logger.info(f"DGGT connection: PASSED ({len(cameras)} cameras available)")

    def test_dggt_camera_specs(self, dggt_scenario):
        """Test DGGT camera specifications."""
        logger.info("Testing DGGT camera specs")

        renderer = dggt_scenario.get_renderer()
        cameras = renderer.get_available_cameras()

        for camera_id, camera_spec in cameras.items():
            # Validate camera spec has required fields
            assert camera_spec.resolution_h > 0, f"Camera {camera_id} has invalid height"
            assert camera_spec.resolution_w > 0, f"Camera {camera_id} has invalid width"

            logger.info(f"Camera {camera_id}: {camera_spec.resolution_w}x{camera_spec.resolution_h}")

        logger.info("DGGT camera specs: PASSED")


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])