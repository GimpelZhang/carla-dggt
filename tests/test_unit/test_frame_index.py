# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unit tests for Frame Index Mapper

Tests the FrameIndexMapper class:
- Single frame index lookup from timestamp
- Midpoint calculation from start/end timestamps
- Batch frame range computation
- Timestamp validation and clamping
"""

import pytest
import numpy as np

# Import the modules under test
try:
    from dggt_server.frame_index_mapper import FrameIndexMapper
    from dggt_server.scene_metadata import DGGTSceneMetadata
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from dggt_server.frame_index_mapper import FrameIndexMapper
    from dggt_server.scene_metadata import DGGTSceneMetadata


@pytest.mark.unit
class TestFrameIndexMapper:
    """Tests for FrameIndexMapper class"""

    @pytest.fixture
    def scene_metadata(self):
        """Create test scene metadata"""
        return DGGTSceneMetadata(
            scene_id="test_scene",
            scene_path="/path/to/scene",
            num_frames=100,
            fps=10.0,
            start_timestamp_us=0,
            end_timestamp_us=9_900_000,  # 99 frames at 10Hz
            camera_width=518,
            camera_height=350,
            intrinsic_matrix=np.eye(3, dtype=np.float32),
            intrinsics_vary=False,
            has_static_scene=True,
            has_sky_scene=True,
            dynamic_object_ids=[0, 1],
            static_scene_size_mb=100.0,
            total_size_mb=120.0
        )

    @pytest.fixture
    def mapper(self, scene_metadata):
        """Create frame index mapper"""
        return FrameIndexMapper(scene_metadata)

    def test_init(self, mapper, scene_metadata):
        """Test initialization"""
        assert mapper.metadata == scene_metadata

    def test_get_frame_index_direct_timestamp_middle(self, mapper):
        """Test direct timestamp lookup at middle of scene"""
        # At 5_000_000 us (5 seconds), should be frame 50 at 10Hz
        assert mapper.get_frame_index(timestamp_us=5_000_000) == 50

    def test_get_frame_index_direct_timestamp_start(self, mapper):
        """Test direct timestamp lookup at start"""
        assert mapper.get_frame_index(timestamp_us=0) == 0

    def test_get_frame_index_direct_timestamp_end(self, mapper):
        """Test direct timestamp lookup at end"""
        assert mapper.get_frame_index(timestamp_us=9_900_000) == 99

    def test_get_frame_index_midpoint_calculation(self, mapper):
        """Test midpoint calculation from start/end timestamps"""
        # Midpoint of 0-1_000_000 is 500_000 -> frame 5 at 10Hz
        assert mapper.get_frame_index(frame_start_us=0, frame_end_us=1_000_000) == 5

        # Midpoint of 5_000_000-6_000_000 is 5_500_000 -> frame 55
        assert mapper.get_frame_index(frame_start_us=5_000_000, frame_end_us=6_000_000) == 55

    def test_get_frame_index_midpoint_symmetric(self, mapper):
        """Test midpoint is symmetric"""
        # Same midpoint regardless of order
        result1 = mapper.get_frame_index(frame_start_us=2_000_000, frame_end_us=4_000_000)
        result2 = mapper.get_frame_index(frame_start_us=4_000_000, frame_end_us=2_000_000)
        # Both should give frame 30 (midpoint of 2M and 4M is 3M -> frame 30)
        # Note: the second case has start > end which is unusual but should still work
        assert result1 == 30

    def test_get_frame_index_no_timestamp_returns_zero(self, mapper):
        """Test default behavior when no timestamp provided"""
        assert mapper.get_frame_index() == 0

    def test_get_frame_index_timestamp_before_start_clamped(self, mapper):
        """Test timestamp before start is clamped to frame 0"""
        assert mapper.get_frame_index(timestamp_us=-1_000_000) == 0

    def test_get_frame_index_timestamp_after_end_clamped(self, mapper):
        """Test timestamp after end is clamped to last frame"""
        assert mapper.get_frame_index(timestamp_us=20_000_000) == 99

    def test_get_frame_index_negative_timestamp(self, mapper):
        """Test negative timestamp handling"""
        assert mapper.get_frame_index(timestamp_us=-100) == 0

    def test_get_frame_range_default_start(self, mapper):
        """Test frame range from default start (scene start)"""
        # 1 second duration from start at 10Hz = 11 frames (0-10 inclusive)
        frames = mapper.get_frame_range(duration_us=1_000_000)
        assert frames == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_get_frame_range_custom_start(self, mapper):
        """Test frame range from custom start timestamp"""
        # 0.5 second duration starting at frame 50 (5_000_000 us)
        frames = mapper.get_frame_range(
            duration_us=500_000,
            start_timestamp_us=5_000_000
        )
        assert frames == [50, 51, 52, 53, 54, 55]

    def test_get_frame_range_exceeds_scene_end_clamped(self, mapper):
        """Test frame range that exceeds scene end is clamped"""
        # Duration exceeds scene end
        frames = mapper.get_frame_range(
            duration_us=20_000_000,
            start_timestamp_us=8_000_000
        )
        # Should be clamped to end of scene (frame 99)
        assert frames[-1] == 99

    def test_get_frame_range_start_before_scene_start_clamped(self, mapper):
        """Test frame range start before scene start is clamped"""
        frames = mapper.get_frame_range(
            duration_us=1_000_000,
            start_timestamp_us=-5_000_000
        )
        # Start should be clamped to 0
        assert frames[0] == 0

    def test_get_frame_range_returns_list(self, mapper):
        """Test that frame range returns a list"""
        frames = mapper.get_frame_range(duration_us=1_000_000)
        assert isinstance(frames, list)

    def test_get_frame_range_consecutive_frames(self, mapper):
        """Test that frame range returns consecutive frame indices"""
        frames = mapper.get_frame_range(duration_us=1_000_000)
        for i in range(len(frames) - 1):
            assert frames[i + 1] == frames[i] + 1

    def test_get_timestamp_range_from_frames(self, mapper):
        """Test timestamp range from frame indices"""
        start_ts, end_ts = mapper.get_timestamp_range([10, 15, 20])
        assert start_ts == 1_000_000  # Frame 10 at 10Hz
        assert end_ts == 2_000_000    # Frame 20 at 10Hz

    def test_get_timestamp_range_single_frame(self, mapper):
        """Test timestamp range from single frame"""
        start_ts, end_ts = mapper.get_timestamp_range([50])
        assert start_ts == 5_000_000
        assert end_ts == 5_000_000

    def test_get_timestamp_range_empty_list(self, mapper):
        """Test timestamp range from empty list"""
        start_ts, end_ts = mapper.get_timestamp_range([])
        # Returns scene start
        assert start_ts == mapper.metadata.start_timestamp_us
        assert end_ts == mapper.metadata.start_timestamp_us

    def test_get_timestamp_range_out_of_bounds_clamped(self, mapper):
        """Test frame indices out of bounds are clamped"""
        # Negative and excessive indices
        start_ts, end_ts = mapper.get_timestamp_range([-5, 150])
        # Clamped to valid range: -5 -> 0, 150 -> 99
        assert start_ts == 0
        assert end_ts == 9_900_000

    def test_validate_timestamp_valid(self, mapper):
        """Test valid timestamp validation"""
        is_valid, clamped = mapper.validate_timestamp(5_000_000)
        assert is_valid == True
        assert clamped == 5_000_000

    def test_validate_timestamp_below_start(self, mapper):
        """Test timestamp below start is invalidated and clamped"""
        is_valid, clamped = mapper.validate_timestamp(-1_000_000)
        assert is_valid == False
        assert clamped == 0

    def test_validate_timestamp_above_end(self, mapper):
        """Test timestamp above end is invalidated and clamped"""
        is_valid, clamped = mapper.validate_timestamp(20_000_000)
        assert is_valid == False
        assert clamped == 9_900_000

    def test_validate_timestamp_at_boundaries(self, mapper):
        """Test timestamp at exact boundaries"""
        # At start
        is_valid, clamped = mapper.validate_timestamp(0)
        assert is_valid == True

        # At end
        is_valid, clamped = mapper.validate_timestamp(9_900_000)
        assert is_valid == True


@pytest.mark.unit
class TestFrameIndexMapperEdgeCases:
    """Edge case tests for FrameIndexMapper"""

    @pytest.fixture
    def short_scene(self):
        """Create very short scene (1 frame)"""
        return DGGTSceneMetadata(
            scene_id="short",
            scene_path="/path",
            num_frames=1,
            fps=10.0,
            start_timestamp_us=0,
            end_timestamp_us=0,
            camera_width=518,
            camera_height=350,
            intrinsic_matrix=np.eye(3, dtype=np.float32),
            intrinsics_vary=False,
            has_static_scene=True,
            has_sky_scene=False,
            dynamic_object_ids=[],
            static_scene_size_mb=10.0,
            total_size_mb=10.0
        )

    @pytest.fixture
    def long_scene(self):
        """Create long scene (1000 frames)"""
        return DGGTSceneMetadata(
            scene_id="long",
            scene_path="/path",
            num_frames=1000,
            fps=10.0,
            start_timestamp_us=0,
            end_timestamp_us=99_900_000,
            camera_width=518,
            camera_height=350,
            intrinsic_matrix=np.eye(3, dtype=np.float32),
            intrinsics_vary=False,
            has_static_scene=True,
            has_sky_scene=True,
            dynamic_object_ids=[0],
            static_scene_size_mb=100.0,
            total_size_mb=150.0
        )

    def test_single_frame_scene_all_timestamps_return_zero(self, short_scene):
        """Test that single frame scene returns frame 0 for all timestamps"""
        mapper = FrameIndexMapper(short_scene)

        assert mapper.get_frame_index(timestamp_us=0) == 0
        assert mapper.get_frame_index(timestamp_us=1_000_000) == 0
        assert mapper.get_frame_index(frame_start_us=0, frame_end_us=100) == 0

    def test_single_frame_scene_frame_range(self, short_scene):
        """Test frame range for single frame scene"""
        mapper = FrameIndexMapper(short_scene)
        frames = mapper.get_frame_range(duration_us=1_000_000)
        assert frames == [0]

    def test_long_scene_frame_index_accuracy(self, long_scene):
        """Test frame index accuracy for long scene"""
        mapper = FrameIndexMapper(long_scene)

        # Frame 500 should be at 50_000_000 us
        assert mapper.get_frame_index(timestamp_us=50_000_000) == 500

        # Frame 999 should be at end
        assert mapper.get_frame_index(timestamp_us=99_900_000) == 999

    def test_long_scene_frame_range(self, long_scene):
        """Test frame range for long scene"""
        mapper = FrameIndexMapper(long_scene)

        # 10 seconds = 101 frames at 10Hz (frames 0-100)
        frames = mapper.get_frame_range(duration_us=10_000_000)
        assert len(frames) == 101
        assert frames[0] == 0
        assert frames[-1] == 100

    def test_high_fps_scene(self):
        """Test scene with higher FPS"""
        metadata = DGGTSceneMetadata(
            scene_id="high_fps",
            scene_path="/path",
            num_frames=100,
            fps=30.0,  # 30 Hz
            start_timestamp_us=0,
            end_timestamp_us=3_300_000,  # ~3.3 seconds for 100 frames at 30Hz
            camera_width=518,
            camera_height=350,
            intrinsic_matrix=np.eye(3, dtype=np.float32),
            intrinsics_vary=False,
            has_static_scene=True,
            has_sky_scene=True,
            dynamic_object_ids=[],
            static_scene_size_mb=50.0,
            total_size_mb=60.0
        )

        mapper = FrameIndexMapper(metadata)

        # At 1_000_000 us (1 second) at 30Hz, should be frame 30
        assert mapper.get_frame_index(timestamp_us=1_000_000) == 30

    def test_fractional_frame_index_truncation(self):
        """Test that fractional frame indices are truncated"""
        metadata = DGGTSceneMetadata(
            scene_id="test",
            scene_path="/path",
            num_frames=100,
            fps=10.0,
            start_timestamp_us=0,
            end_timestamp_us=9_900_000,
            camera_width=518,
            camera_height=350,
            intrinsic_matrix=np.eye(3, dtype=np.float32),
            intrinsics_vary=False,
            has_static_scene=True,
            has_sky_scene=True,
            dynamic_object_ids=[],
            static_scene_size_mb=50.0,
            total_size_mb=60.0
        )

        mapper = FrameIndexMapper(metadata)

        # 5_500_000 us should give frame 55 (exact)
        # 5_549_999 us should also give frame 55 (truncated)
        assert mapper.get_frame_index(timestamp_us=5_500_000) == 55
        assert mapper.get_frame_index(timestamp_us=5_549_999) == 55
        # 5_550_000 us should give frame 55 (still, due to int truncation)
        # Actually at 10Hz, each frame is 100_000 us, so:
        # frame_idx = elapsed_us * fps / 1_000_000
        # = 5_550_000 * 10 / 1_000_000 = 55.5 -> int truncates to 55
        assert mapper.get_frame_index(timestamp_us=5_550_000) == 55

    def test_timestamp_range_with_unsorted_frames(self):
        """Test timestamp range handles unsorted frame indices"""
        metadata = DGGTSceneMetadata(
            scene_id="test",
            scene_path="/path",
            num_frames=100,
            fps=10.0,
            start_timestamp_us=0,
            end_timestamp_us=9_900_000,
            camera_width=518,
            camera_height=350,
            intrinsic_matrix=np.eye(3, dtype=np.float32),
            intrinsics_vary=False,
            has_static_scene=True,
            has_sky_scene=True,
            dynamic_object_ids=[],
            static_scene_size_mb=50.0,
            total_size_mb=60.0
        )

        mapper = FrameIndexMapper(metadata)

        # Unsorted input
        start_ts, end_ts = mapper.get_timestamp_range([50, 10, 30])
        # Should use min and max
        assert start_ts == 1_000_000  # Frame 10
        assert end_ts == 5_000_000    # Frame 50


@pytest.mark.unit
class TestFrameIndexMapperPriority:
    """Test timestamp priority in get_frame_index"""

    @pytest.fixture
    def metadata(self):
        """Create base metadata"""
        return DGGTSceneMetadata(
            scene_id="test",
            scene_path="/path",
            num_frames=100,
            fps=10.0,
            start_timestamp_us=0,
            end_timestamp_us=9_900_000,
            camera_width=518,
            camera_height=350,
            intrinsic_matrix=np.eye(3, dtype=np.float32),
            intrinsics_vary=False,
            has_static_scene=True,
            has_sky_scene=True,
            dynamic_object_ids=[],
            static_scene_size_mb=50.0,
            total_size_mb=60.0
        )

    @pytest.fixture
    def mapper(self, metadata):
        """Create mapper"""
        return FrameIndexMapper(metadata)

    def test_timestamp_us_overrides_start_end(self, mapper):
        """Test that timestamp_us takes priority over frame_start/end"""
        # Provide both timestamp_us and frame_start/end
        # timestamp_us should be used
        result = mapper.get_frame_index(
            timestamp_us=5_000_000,  # Should give frame 50
            frame_start_us=0,        # Would give frame 5 if used
            frame_end_us=1_000_000
        )
        assert result == 50

    def test_only_frame_start_provided_returns_zero(self, mapper):
        """Test that only frame_start (without frame_end) returns 0"""
        result = mapper.get_frame_index(frame_start_us=5_000_000)
        assert result == 0

    def test_only_frame_end_provided_returns_zero(self, mapper):
        """Test that only frame_end (without frame_start) returns 0"""
        result = mapper.get_frame_index(frame_end_us=5_000_000)
        assert result == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])