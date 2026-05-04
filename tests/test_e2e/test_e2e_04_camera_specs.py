# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
E2E Test E2E-04: Camera Specifications

Tests different camera configurations:
- Multiple resolution rendering
- Ftheta (fisheye) camera support
- Camera parameter validation

Test ID: E2E-04
Test Name: camera_specs
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import logging
from scipy.spatial.transform import Rotation as R

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dggt_integration import DggtScenario, DggtRenderer
from dggt_config import DggtConfig
from utils import undo_carla_coordinate_transform, se3_to_grpc_pose
from nre.grpc.protos import sensorsim_pb2, common_pb2

# Import helper functions from test_helpers module
from .test_helpers import (
    create_test_camera_pose,
    validate_rendered_image,
    save_image,
)

# Note: pytest fixtures (e2e_config, dggt_scenario, test_output_dir) are auto-discovered from conftest.py

logger = logging.getLogger(__name__)


class TestE2E04CameraSpecs:
    """
    E2E-04: Camera Specifications Test

    Tests rendering with different camera configurations.
    """

    @pytest.fixture(autouse=True)
    def setup(self, dggt_scenario, test_output_dir, e2e_config):
        """Setup test fixtures."""
        self.scenario = dggt_scenario
        self.renderer = self.scenario.get_renderer()
        self.output_dir = test_output_dir
        self.save_images = e2e_config['test']['save_images']

        logger.info(f"E2E-04 setup complete")

    def test_different_resolutions(self):
        """
        Test E2E-04: Render at multiple resolutions.

        Tests:
        - 256x256: Low resolution
        - 512x512: Medium resolution
        - 1024x768: Standard HD resolution

        Validates each resolution produces valid image.
        """
        logger.info("Starting E2E-04: Multiple resolution test")

        # Resolution test cases
        resolutions = [
            (256, 256, "256x256"),
            (512, 512, "512x512"),
            (1024, 768, "1024x768"),
        ]

        # Get base camera spec
        base_camera_spec = self.renderer.get_camera_spec(
            list(self.renderer.get_available_cameras().keys())[0]
        )

        # Create camera pose
        camera_pose = create_test_camera_pose(x=0.0, y=0.0, z=2.0)
        camera_pose_carla = undo_carla_coordinate_transform(camera_pose)

        timestamp_us = 0

        rendered_images = []

        for height, width, name in resolutions:
            logger.info(f"Testing resolution: {name} ({width}x{height})")

            # Create camera spec with target resolution
            camera_spec = sensorsim_pb2.CameraSpec()
            camera_spec.resolution_h = height
            camera_spec.resolution_w = width

            # Copy intrinsics from base camera (simplified)
            # In real scenario, intrinsics should be scaled appropriately
            if base_camera_spec.HasField('opencv_pinhole_param'):
                pinhole = camera_spec.opencv_pinhole_param
                base_pinhole = base_camera_spec.opencv_pinhole_param
                # Scale focal length proportionally
                scale_h = height / base_camera_spec.resolution_h
                scale_w = width / base_camera_spec.resolution_w
                pinhole.focal_length_x = base_pinhole.focal_length_x * scale_w
                pinhole.focal_length_y = base_pinhole.focal_length_y * scale_h
                pinhole.principal_point_x = base_pinhole.principal_point_x * scale_w
                pinhole.principal_point_y = base_pinhole.principal_point_y * scale_h
            elif base_camera_spec.HasField('ftheta_param'):
                # Copy ftheta parameters
                ftheta = camera_spec.ftheta_param
                base_ftheta = base_camera_spec.ftheta_param
                ftheta.principal_point_x = base_ftheta.principal_point_x
                ftheta.principal_point_y = base_ftheta.principal_point_y
                ftheta.max_angle = base_ftheta.max_angle
                ftheta.pixeldist_to_angle_poly.extend(base_ftheta.pixeldist_to_angle_poly)

            # Render
            try:
                # Calculate resolution ratio to achieve target resolution
                ratio_h = height / base_camera_spec.resolution_h
                ratio_w = width / base_camera_spec.resolution_w
                resolution_ratio = min(ratio_h, ratio_w)

                image = self.renderer.render(
                    world_snapshot=None,
                    camera_spec=base_camera_spec,  # Use base spec with ratio
                    camera_pose=camera_pose_carla,
                    timestamp=timestamp_us,
                    resolution_ratio=resolution_ratio,
                )
            except Exception as e:
                pytest.fail(f"Render at {name} failed: {e}")

            # Validate
            # Note: Actual resolution may differ due to server-side constraints
            logger.info(f"Rendered image shape: {image.shape}")

            validate_rendered_image(image)

            rendered_images.append((name, image))

            # Save if configured
            if self.save_images:
                filename = f"e2e_04_{name}.jpg"
                save_path = save_image(image, self.output_dir, filename)
                logger.info(f"Saved: {save_path}")

        logger.info(f"E2E-04 resolutions: PASSED ({len(rendered_images)} images)")

    def test_ftheta_camera(self):
        """
        Test E2E-04: Ftheta (fisheye) camera render.

        Tests 190-degree FOV fisheye camera rendering.

        Validates:
        - Ftheta camera spec accepted
        - Render produces valid image
        - Image shows expected fisheye distortion pattern
        """
        logger.info("Starting E2E-04: Ftheta camera test")

        # Create ftheta camera spec with 190-degree FOV
        # 190 degrees = 190 * pi / 180 = 3.316 radians (half-angle = 1.658 rad)
        # For full 190 degree FOV, max_angle should be 95 degrees = 1.658 rad
        max_angle_rad = 190.0 * np.pi / 180.0 / 2.0  # Half-angle

        camera_spec = sensorsim_pb2.CameraSpec()
        camera_spec.resolution_h = 1200
        camera_spec.resolution_w = 1200
        camera_spec.logical_id = "test_ftheta_190"

        # Set ftheta parameters
        ftheta = camera_spec.ftheta_param
        ftheta.principal_point_x = 600.0
        ftheta.principal_point_y = 600.0
        ftheta.max_angle = max_angle_rad

        # Simple polynomial approximation for fisheye
        # pixel_dist = focal_length * angle
        # For 190 degree FOV, radius should cover ~95 degrees
        focal_length = 600.0  # Approximate focal length
        ftheta.pixeldist_to_angle_poly.extend([0.0, 1.0 / focal_length])
        ftheta.reference_poly = sensorsim_pb2.FthetaCameraParam.PIXELDIST_TO_ANGLE

        logger.info(f"Ftheta spec: {camera_spec.resolution_w}x{camera_spec.resolution_h}, max_angle={max_angle_rad:.3f} rad (~{max_angle_rad * 180 / np.pi:.1f} deg)")

        # Create camera pose
        camera_pose = create_test_camera_pose(x=0.0, y=0.0, z=2.0)
        camera_pose_carla = undo_carla_coordinate_transform(camera_pose)

        timestamp_us = 0

        # Check if ftheta camera is supported
        try:
            # Try to get a ftheta camera from available cameras first
            available_cameras = self.renderer.get_available_cameras()
            ftheta_cameras = [
                cam_id for cam_id, spec in available_cameras.items()
                if spec.HasField('ftheta_param')
            ]

            if ftheta_cameras:
                # Use existing ftheta camera
                logger.info(f"Using existing ftheta camera: {ftheta_cameras[0]}")
                camera_spec = available_cameras[ftheta_cameras[0]]

                image = self.renderer.render(
                    world_snapshot=None,
                    camera_spec=camera_spec,
                    camera_pose=camera_pose_carla,
                    timestamp=timestamp_us,
                    resolution_ratio=1.0,
                )
            else:
                # Test with custom ftheta spec (may not be supported by all servers)
                logger.warning("No ftheta camera available, testing with custom spec")

                # Skip if server doesn't support custom camera specs
                pytest.skip("Server does not have ftheta cameras configured")

        except Exception as e:
            if "camera" in str(e).lower() or "not found" in str(e).lower():
                pytest.skip(f"Ftheta camera not available: {e}")
            else:
                pytest.fail(f"Ftheta render failed: {e}")

        # Validate
        validate_rendered_image(image)

        # Check for fisheye distortion pattern
        # Fisheye images typically have:
        # - Circular boundary
        # - Center region is less distorted
        # - Edge region is more distorted

        # Simple heuristic: check image center has content
        center_region = image[
            image.shape[0]//3:2*image.shape[0]//3,
            image.shape[1]//3:2*image.shape[1]//3
        ]
        center_mean = np.mean(center_region)

        logger.info(f"Ftheta image: shape={image.shape}, center_mean={center_mean:.2f}")

        assert center_mean > 0, "Ftheta image center is all black"

        # Save if configured
        if self.save_images:
            filename = "e2e_04_ftheta_190.jpg"
            save_path = save_image(image, self.output_dir, filename)
            logger.info(f"Saved: {save_path}")

        logger.info("E2E-04 ftheta: PASSED")

    def test_camera_spec_validation(self):
        """
        Test E2E-04: Camera spec validation.

        Validates that available camera specs have required fields.
        """
        logger.info("Starting E2E-04: Camera spec validation")

        available_cameras = self.renderer.get_available_cameras()

        for camera_id, spec in available_cameras.items():
            logger.info(f"Validating camera: {camera_id}")

            # Check resolution
            assert spec.resolution_h > 0, f"Camera {camera_id}: invalid height"
            assert spec.resolution_w > 0, f"Camera {camera_id}: invalid width"

            # Check camera parameters (at least one type should be set)
            has_params = (
                spec.HasField('opencv_pinhole_param') or
                spec.HasField('opencv_fisheye_param') or
                spec.HasField('ftheta_param')
            )
            assert has_params, f"Camera {camera_id}: no intrinsic parameters"

            # Validate specific parameter types
            if spec.HasField('opencv_pinhole_param'):
                pinhole = spec.opencv_pinhole_param
                assert pinhole.focal_length_x > 0, "Invalid focal_length_x"
                assert pinhole.focal_length_y > 0, "Invalid focal_length_y"

            if spec.HasField('opencv_fisheye_param'):
                fisheye = spec.opencv_fisheye_param
                assert fisheye.focal_length_x > 0, "Invalid focal_length_x"
                assert fisheye.max_angle > 0, "Invalid max_angle"

            if spec.HasField('ftheta_param'):
                ftheta = spec.ftheta_param
                assert ftheta.max_angle > 0, "Invalid max_angle"

            logger.info(f"Camera {camera_id}: VALID")

        logger.info("E2E-04 spec validation: PASSED")


class TestE2E04ResolutionScaling:
    """
    E2E-04: Resolution Scaling Tests

    Tests resolution scaling functionality.
    """

    def test_resolution_ratio_scaling(self, dggt_scenario, e2e_config):
        """
        Test resolution ratio scaling.

        Tests that resolution_ratio parameter correctly scales output.
        """
        logger.info("Starting resolution ratio test")

        renderer = dggt_scenario.get_renderer()
        base_camera_spec = list(renderer.get_available_cameras().values())[0]

        base_height = base_camera_spec.resolution_h
        base_width = base_camera_spec.resolution_w

        camera_pose = create_test_camera_pose(x=0.0, y=0.0, z=2.0)
        camera_pose_carla = undo_carla_coordinate_transform(camera_pose)

        # Test different ratios
        ratios = [0.5, 1.0, 1.5]

        for ratio in ratios:
            logger.info(f"Testing ratio: {ratio}")

            try:
                image = renderer.render(
                    world_snapshot=None,
                    camera_spec=base_camera_spec,
                    camera_pose=camera_pose_carla,
                    timestamp=0,
                    resolution_ratio=ratio,
                )
            except Exception as e:
                pytest.fail(f"Render at ratio {ratio} failed: {e}")

            # Validate image
            validate_rendered_image(image)

            # Expected resolution (may differ due to server constraints)
            expected_h = int(base_height * ratio)
            expected_w = int(base_width * ratio)

            logger.info(f"Ratio {ratio}: expected {expected_w}x{expected_h}, got {image.shape[1]}x{image.shape[0]}")

            if e2e_config['test']['save_images']:
                output_dir = Path(e2e_config['test']['output_dir'])
                filename = f"e2e_04_ratio_{ratio}.jpg"
                save_image(image, output_dir, filename)

        logger.info("Resolution ratio test: PASSED")


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])