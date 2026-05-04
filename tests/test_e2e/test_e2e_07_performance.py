# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
E2E Test E2E-07: Performance Measurements

Tests performance metrics:
- Render latency (P50/P95/P99 percentile analysis)
- FPS measurement (frame rate throughput)

Test ID: E2E-07
Test Name: performance_measurements
"""

import pytest
import numpy as np
import sys
import json
import time
from pathlib import Path
import logging
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dggt_integration import DggtScenario, DggtRenderer
from dggt_config import DggtConfig
from utils import undo_carla_coordinate_transform
from nre.grpc.protos import sensorsim_pb2

# Import helper functions from test_helpers module
from .test_helpers import (
    create_test_camera_pose,
    save_image,
)

# Note: pytest fixtures (e2e_config, carla_client, dggt_world, dggt_scenario,
# default_camera_spec, test_output_dir) are auto-discovered from conftest.py

logger = logging.getLogger(__name__)


class TestE2E07Performance:
    """
    E2E-07: Performance Measurement Tests

    Measures and reports DGGT rendering performance metrics.
    """

    @pytest.fixture(autouse=True)
    def setup(self, dggt_scenario, default_camera_spec, test_output_dir, e2e_config):
        """Setup test fixtures."""
        self.scenario = dggt_scenario
        self.renderer = self.scenario.get_renderer()
        self.camera_spec = default_camera_spec
        self.output_dir = test_output_dir
        self.test_cfg = e2e_config['test']

        logger.info("E2E-07 setup complete")

    def test_render_latency(self):
        """
        Test E2E-07: Measure render latency across 30 samples.

        Computes P50, P95, P99 percentile latencies and saves JSON results.

        Metrics:
        - Latency: time from render call start to image returned
        - P50: median latency (typical performance)
        - P95: 95th percentile (worst 5% cases)
        - P99: 99th percentile (worst 1% cases)
        """
        logger.info("Starting E2E-07: Render latency measurement")

        num_samples = 30
        latencies = []

        # Get timestamp range
        start_ts, end_ts = self.scenario.get_timestamp_range()
        timestamp_step = (end_ts - start_ts) // num_samples

        # Create fixed camera pose
        camera_pose = create_test_camera_pose(x=0.0, y=0.0, z=2.0)
        camera_pose_carla = undo_carla_coordinate_transform(camera_pose)

        logger.info(f"Collecting {num_samples} latency samples...")

        for i in range(num_samples):
            timestamp_us = start_ts + i * timestamp_step

            # Measure render time
            start_time = time.perf_counter()

            try:
                image = self.renderer.render(
                    world_snapshot=None,
                    camera_spec=self.camera_spec,
                    camera_pose=camera_pose_carla,
                    timestamp=timestamp_us,
                    resolution_ratio=1.0,
                )
            except Exception as e:
                logger.warning(f"Render failed at sample {i}: {e}")
                continue

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds

            latencies.append(latency_ms)
            logger.debug(f"Sample {i}: latency={latency_ms:.2f}ms")

        if len(latencies) < num_samples // 2:
            pytest.fail(f"Too few successful renders: {len(latencies)}/{num_samples}")

        # Compute percentile statistics
        latencies_sorted = sorted(latencies)
        p50 = np.percentile(latencies_sorted, 50)
        p95 = np.percentile(latencies_sorted, 95)
        p99 = np.percentile(latencies_sorted, 99)
        mean_latency = np.mean(latencies_sorted)
        min_latency = np.min(latencies_sorted)
        max_latency = np.max(latencies_sorted)

        logger.info(f"Latency statistics:")
        logger.info(f"  Min:   {min_latency:.2f}ms")
        logger.info(f"  Mean:  {mean_latency:.2f}ms")
        logger.info(f"  P50:   {p50:.2f}ms")
        logger.info(f"  P95:   {p95:.2f}ms")
        logger.info(f"  P99:   {p99:.2f}ms")
        logger.info(f"  Max:   {max_latency:.2f}ms")

        # Save results to JSON
        results = {
            "test": "E2E-07_render_latency",
            "num_samples": len(latencies),
            "latencies_ms": latencies_sorted,
            "statistics": {
                "min_ms": float(min_latency),
                "mean_ms": float(mean_latency),
                "p50_ms": float(p50),
                "p95_ms": float(p95),
                "p99_ms": float(p99),
                "max_ms": float(max_latency),
            },
            "timestamp_range_us": {
                "start": start_ts,
                "end": end_ts,
            },
            "camera_resolution": {
                "width": self.camera_spec.resolution_w,
                "height": self.camera_spec.resolution_h,
            },
        }

        results_path = self.output_dir / "e2e_07_latency_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Latency results saved to: {results_path}")

        # Basic validation: latencies should be reasonable
        # Allow wide range since DGGT rendering can vary significantly
        assert p50 > 0, "P50 latency should be positive"
        assert p99 < 60000, "P99 latency should be under 60 seconds"  # Reasonable timeout

        logger.info("E2E-07 render latency: PASSED")

    def test_fps_measurement(self):
        """
        Test E2E-07: Measure frame rate (FPS) over 60 frames.

        Measures throughput by rendering 60 consecutive frames.

        Metrics:
        - FPS: frames per second (num_frames / total_time)
        - Average frame time: total_time / num_frames
        """
        logger.info("Starting E2E-07: FPS measurement")

        num_frames = 60

        # Get timestamp range
        start_ts, end_ts = self.scenario.get_timestamp_range()
        timestamp_step = (end_ts - start_ts) // num_frames

        # Create fixed camera pose
        camera_pose = create_test_camera_pose(x=0.0, y=0.0, z=2.0)
        camera_pose_carla = undo_carla_coordinate_transform(camera_pose)

        logger.info(f"Measuring FPS over {num_frames} frames...")

        # Warm-up: render a few frames first to ensure steady state
        warmup_frames = 5
        for i in range(warmup_frames):
            timestamp_us = start_ts + i * timestamp_step
            try:
                self.renderer.render(
                    world_snapshot=None,
                    camera_spec=self.camera_spec,
                    camera_pose=camera_pose_carla,
                    timestamp=timestamp_us,
                    resolution_ratio=1.0,
                )
            except Exception:
                pass

        # Start measurement
        total_start_time = time.perf_counter()
        successful_frames = 0

        for i in range(num_frames):
            timestamp_us = start_ts + (warmup_frames + i) * timestamp_step

            try:
                image = self.renderer.render(
                    world_snapshot=None,
                    camera_spec=self.camera_spec,
                    camera_pose=camera_pose_carla,
                    timestamp=timestamp_us,
                    resolution_ratio=1.0,
                )
                successful_frames += 1
            except Exception as e:
                logger.warning(f"Frame {i} failed: {e}")

        total_end_time = time.perf_counter()
        total_time = total_end_time - total_start_time

        # Compute FPS
        fps = successful_frames / total_time
        avg_frame_time_ms = (total_time / successful_frames) * 1000 if successful_frames > 0 else 0

        logger.info(f"FPS measurement results:")
        logger.info(f"  Total frames attempted: {num_frames}")
        logger.info(f"  Successful frames: {successful_frames}")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  FPS: {fps:.2f}")
        logger.info(f"  Average frame time: {avg_frame_time_ms:.2f}ms")

        # Save results to JSON
        results = {
            "test": "E2E-07_fps_measurement",
            "num_frames_attempted": num_frames,
            "successful_frames": successful_frames,
            "total_time_seconds": total_time,
            "fps": fps,
            "avg_frame_time_ms": avg_frame_time_ms,
            "timestamp_range_us": {
                "start": start_ts,
                "end": end_ts,
            },
            "camera_resolution": {
                "width": self.camera_spec.resolution_w,
                "height": self.camera_spec.resolution_h,
            },
        }

        results_path = self.output_dir / "e2e_07_fps_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"FPS results saved to: {results_path}")

        # Basic validation
        assert successful_frames >= num_frames * 0.8, \
            f"Too few successful frames: {successful_frames}/{num_frames}"
        assert fps > 0, "FPS should be positive"

        logger.info("E2E-07 FPS measurement: PASSED")

    def test_render_consistency(self):
        """
        Test E2E-07: Verify render consistency over repeated frames.

        Validates that repeated renders at same timestamp produce similar results.
        """
        logger.info("Starting E2E-07: Render consistency test")

        num_samples = 10
        timestamp_us = self.scenario.get_scene_metadata().start_timestamp_us

        camera_pose = create_test_camera_pose(x=0.0, y=0.0, z=2.0)
        camera_pose_carla = undo_carla_coordinate_transform(camera_pose)

        images = []
        for i in range(num_samples):
            try:
                image = self.renderer.render(
                    world_snapshot=None,
                    camera_spec=self.camera_spec,
                    camera_pose=camera_pose_carla,
                    timestamp=timestamp_us,  # Same timestamp for all
                    resolution_ratio=1.0,
                )
                images.append(image)
            except Exception as e:
                logger.warning(f"Render {i} failed: {e}")

        if len(images) < num_samples * 0.8:
            pytest.fail(f"Too few successful renders: {len(images)}/{num_samples}")

        # Compute mean values for each image
        means = [np.mean(img) for img in images]
        mean_std = np.std(means)

        logger.info(f"Render consistency: mean_std={mean_std:.4f}")

        # Consistent renders should have similar mean values
        # Allow some variance due to floating point/compression differences
        assert mean_std < 5.0, f"Render means too inconsistent: std={mean_std}"

        logger.info("E2E-07 render consistency: PASSED")


class TestE2E07PerformanceSummary:
    """
    E2E-07: Performance Summary Generator

    Aggregates performance metrics into summary report.
    """

    def test_performance_summary(self, test_output_dir):
        """
        Generate combined performance summary from latency and FPS results.
        """
        logger.info("Starting E2E-07: Performance summary generation")

        latency_path = test_output_dir / "e2e_07_latency_results.json"
        fps_path = test_output_dir / "e2e_07_fps_results.json"

        summary = {
            "test": "E2E-07_performance_summary",
            "tests_completed": [],
        }

        # Load latency results if available
        if latency_path.exists():
            with open(latency_path, 'r') as f:
                latency_results = json.load(f)
            summary["latency"] = latency_results["statistics"]
            summary["tests_completed"].append("render_latency")
            logger.info(f"Loaded latency results: P50={latency_results['statistics']['p50_ms']}ms")

        # Load FPS results if available
        if fps_path.exists():
            with open(fps_path, 'r') as f:
                fps_results = json.load(f)
            summary["fps"] = {
                "fps": fps_results["fps"],
                "avg_frame_time_ms": fps_results["avg_frame_time_ms"],
            }
            summary["tests_completed"].append("fps_measurement")
            logger.info(f"Loaded FPS results: {fps_results['fps']:.2f} FPS")

        # Save summary
        summary_path = test_output_dir / "e2e_07_performance_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Performance summary saved to: {summary_path}")

        # Pass if at least one test completed
        assert len(summary["tests_completed"]) >= 1, \
            "No performance tests completed"

        logger.info("E2E-07 performance summary: PASSED")


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])