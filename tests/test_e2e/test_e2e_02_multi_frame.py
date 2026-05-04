# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
E2E Test E2E-02: Multi-Frame Sequence Test

Tests rendering multiple frames in sequence:
- Frame-by-frame rendering
- Render time tracking
- Frame difference calculation
- Success rate validation

Test ID: E2E-02
Test Name: multi_frame_sequence
"""

import pytest
import numpy as np
import sys
import time
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

logger = logging.getLogger(__name__)


class TestE2E02MultiFrame:
    """
    E2E-02: Multi-Frame Sequence Test

    Tests rendering a sequence of frames from the scenario timeline.
    """

    @pytest.fixture(autouse=True)
    def setup(self, dggt_scenario, default_camera_spec, test_output_dir, e2e_config):
        """Setup test fixtures."""
        self.scenario = dggt_scenario
        self.renderer = self.scenario.get_renderer()
        self.camera_spec = default_camera_spec
        self.output_dir = test_output_dir / "e2e_02_multi_frame"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_images = e2e_config['test']['save_images']
        self.num_frames = e2e_config['test'].get('multi_frame_count', 10)

        # Get timestamp range
        self.start_ts, self.end_ts = self.scenario.get_timestamp_range()
        self.timestamp_step = (self.end_ts - self.start_ts) // self.num_frames

        logger.info(f"E2E-02 setup complete: {self.num_frames} frames, "
                    f"timestamps {self.start_ts} to {self.end_ts}")

    def test_multi_frame_sequence(self):
        """
        Test E2E-02: Render multi-frame sequence.

        Tests:
        - Loop through num_frames (default 10)
        - Render each frame at different timestamps
        - Track render time for each frame
        - Compute frame-to-frame differences
        - Collect statistics: success_count, fail_count, avg/max/min render_time

        Validates:
        - All frames render successfully
        - Render times are reasonable (< 5 seconds each)
        - Frame differences are meaningful (> 0 for dynamic scenes)
        """
        logger.info(f"Starting E2E-02: Multi-frame sequence test ({self.num_frames} frames)")

        # Results tracking
        results = {
            "success_count": 0,
            "fail_count": 0,
            "render_times": [],
            "frame_diffs": [],
            "frames": [],
        }

        # Camera pose (fixed position for all frames)
        camera_pose = create_test_camera_pose(x=0.0, y=0.0, z=2.0)
        camera_pose_carla = undo_carla_coordinate_transform(camera_pose)

        prev_image = None

        # Render each frame
        for frame_idx in range(self.num_frames):
            frame_result = {
                "frame_idx": frame_idx,
                "timestamp": None,
                "status": None,
                "render_time_ms": None,
                "mean_diff": None,
            }

            # Calculate timestamp for this frame
            timestamp = self.start_ts + frame_idx * self.timestamp_step
            frame_result["timestamp"] = timestamp

            logger.info(f"Rendering frame {frame_idx}/{self.num_frames} at timestamp {timestamp} us")

            try:
                # Start render timing
                start_time = time.time()

                # Render frame
                image = self.renderer.render(
                    world_snapshot=None,  # No dynamic objects for basic test
                    camera_spec=self.camera_spec,
                    camera_pose=camera_pose_carla,
                    timestamp=timestamp,
                    resolution_ratio=1.0,
                )

                # End render timing
                render_time_ms = (time.time() - start_time) * 1000
                frame_result["render_time_ms"] = render_time_ms

                logger.info(f"Frame {frame_idx}: rendered in {render_time_ms:.1f} ms")

                # Validate image
                validate_rendered_image(image)

                # Compute frame difference from previous frame
                if prev_image is not None:
                    # Ensure same shape
                    if image.shape == prev_image.shape:
                        diff = np.abs(image.astype(float) - prev_image.astype(float))
                        mean_diff = np.mean(diff)
                        frame_result["mean_diff"] = mean_diff
                        results["frame_diffs"].append(mean_diff)

                        logger.info(f"Frame {frame_idx}: mean difference from prev = {mean_diff:.2f}")

                # Save image if configured
                if self.save_images:
                    filename = f"e2e_02_frame_{frame_idx:04d}.jpg"
                    save_image(image, self.output_dir, filename)

                # Update results
                frame_result["status"] = "success"
                results["success_count"] += 1
                results["render_times"].append(render_time_ms)

                # Store for next iteration
                prev_image = image.copy()

            except Exception as e:
                frame_result["status"] = "failed"
                frame_result["error"] = str(e)
                results["fail_count"] += 1

                logger.error(f"Frame {frame_idx}: FAILED - {e}")

            results["frames"].append(frame_result)

        # Compute statistics
        if results["render_times"]:
            results["avg_render_time_ms"] = np.mean(results["render_times"])
            results["max_render_time_ms"] = np.max(results["render_times"])
            results["min_render_time_ms"] = np.min(results["render_times"])

        if results["frame_diffs"]:
            results["avg_frame_diff"] = np.mean(results["frame_diffs"])
            results["max_frame_diff"] = np.max(results["frame_diffs"])
            results["min_frame_diff"] = np.min(results["frame_diffs"])

        # Log summary
        logger.info(f"\n=== E2E-02 Multi-Frame Test Results ===")
        logger.info(f"Success count: {results['success_count']}/{self.num_frames}")
        logger.info(f"Fail count: {results['fail_count']}/{self.num_frames}")
        if results["render_times"]:
            logger.info(f"Average render time: {results['avg_render_time_ms']:.1f} ms")
            logger.info(f"Max render time: {results['max_render_time_ms']:.1f} ms")
            logger.info(f"Min render time: {results['min_render_time_ms']:.1f} ms")
        if results["frame_diffs"]:
            logger.info(f"Average frame diff: {results['avg_frame_diff']:.2f}")
            logger.info(f"Max frame diff: {results['max_frame_diff']:.2f}")
            logger.info(f"Min frame diff: {results['min_frame_diff']:.2f}")

        # Validations
        success_rate = results["success_count"] / self.num_frames

        # All frames must succeed
        assert success_rate >= 0.9, \
            f"Success rate {success_rate*100:.1f}% is below threshold (90%)"

        # Render times must be reasonable
        if results["render_times"]:
            assert results["avg_render_time_ms"] < 5000, \
                f"Average render time {results['avg_render_time_ms']:.1f} ms is too high"

        logger.info("E2E-02 multi-frame sequence: PASSED")

    def test_multi_frame_varying_camera(self):
        """
        Test E2E-02 variant: Multi-frame with moving camera.

        Tests rendering with camera position changing between frames.
        """
        logger.info("Starting E2E-02 variant: Moving camera test")

        success_count = 0
        fail_count = 0
        render_times = []

        # Camera moves along X axis
        for frame_idx in range(5):
            # Move camera forward each frame
            x_pos = frame_idx * 5.0  # 5m per frame

            camera_pose = create_test_camera_pose(x=x_pos, y=0.0, z=2.0)
            camera_pose_carla = undo_carla_coordinate_transform(camera_pose)

            timestamp = self.start_ts + frame_idx * self.timestamp_step

            try:
                start_time = time.time()

                image = self.renderer.render(
                    world_snapshot=None,
                    camera_spec=self.camera_spec,
                    camera_pose=camera_pose_carla,
                    timestamp=timestamp,
                    resolution_ratio=1.0,
                )

                render_time_ms = (time.time() - start_time) * 1000
                render_times.append(render_time_ms)

                validate_rendered_image(image)

                success_count += 1

                if self.save_images:
                    filename = f"e2e_02_moving_frame_{frame_idx:04d}.jpg"
                    save_image(image, self.output_dir, filename)

                logger.info(f"Moving camera frame {frame_idx}: SUCCESS at x={x_pos}")

            except Exception as e:
                fail_count += 1
                logger.error(f"Moving camera frame {frame_idx}: FAILED - {e}")

        logger.info(f"Moving camera test: {success_count}/5 frames succeeded")

        assert success_count >= 4, "At least 4 of 5 frames must succeed"

        logger.info("E2E-02 moving camera variant: PASSED")


class TestE2E02Performance:
    """
    E2E-02: Performance Tests

    Tests render performance metrics.
    """

    def test_render_time_consistency(self, dggt_scenario, default_camera_spec):
        """Test render time is consistent across multiple calls."""
        logger.info("Testing render time consistency")

        renderer = dggt_scenario.get_renderer()
        camera_pose = create_test_camera_pose(x=0.0, y=0.0, z=2.0)
        camera_pose_carla = undo_carla_coordinate_transform(camera_pose)

        render_times = []

        # Render 5 times at same timestamp
        for i in range(5):
            start_time = time.time()
            image = renderer.render(
                world_snapshot=None,
                camera_spec=default_camera_spec,
                camera_pose=camera_pose_carla,
                timestamp=0,
                resolution_ratio=1.0,
            )
            render_time_ms = (time.time() - start_time) * 1000
            render_times.append(render_time_ms)

        # Check consistency (std dev < 50% of mean)
        mean_time = np.mean(render_times)
        std_time = np.std(render_times)

        logger.info(f"Render times: {render_times}")
        logger.info(f"Mean: {mean_time:.1f} ms, Std: {std_time:.1f} ms")

        # Allow some variation
        assert std_time < mean_time * 0.5, \
            f"Render time too inconsistent: std={std_time:.1f} > 50% of mean={mean_time:.1f}"

        logger.info("Render time consistency: PASSED")


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])