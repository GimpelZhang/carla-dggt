# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unit tests for frame_index_mapper module
"""

import pytest
import numpy as np
from dggt_server.frame_index_mapper import FrameIndexMapper
from dggt_server.scene_metadata import DGGTSceneMetadata


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
            intrinsic_matrix=np.eye(3),
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

    def test_get_frame_index_direct_timestamp(self, mapper):
        """Test direct timestamp lookup"""
        # Middle of scene
        assert mapper.get_frame_index(timestamp_us=5_000_000) == 50

        # Start of scene
        assert mapper.get_frame_index(timestamp_us=0) == 0

        # End of scene
        assert mapper.get_frame_index(timestamp_us=9_900_000) == 99

    def test_get_frame_index_midpoint(self, mapper):
        """Test midpoint calculation from start/end"""
        # Midpoint of 0-1_000_000 is 500_000 -> frame 5
        assert mapper.get_frame_index(frame_start_us=0, frame_end_us=1_000_000) == 5

        # Midpoint of 5_000_000-6_000_000 is 5_500_000 -> frame 55
        assert mapper.get_frame_index(frame_start_us=5_000_000, frame_end_us=6_000_000) == 55

    def test_get_frame_index_no_timestamp(self, mapper):
        """Test default behavior when no timestamp provided"""
        assert mapper.get_frame_index() == 0

    def test_get_frame_index_clamping(self, mapper):
        """Test timestamp clamping to valid range"""
        # Before start -> clamped to frame 0
        assert mapper.get_frame_index(timestamp_us=-1_000_000) == 0

        # After end -> clamped to last frame
        assert mapper.get_frame_index(timestamp_us=20_000_000) == 99

    def test_get_frame_range_default_start(self, mapper):
        """Test frame range from default start"""
        # 1 second duration from start
        frames = mapper.get_frame_range(duration_us=1_000_000)
        assert frames == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_get_frame_range_custom_start(self, mapper):
        """Test frame range from custom start"""
        # 0.5 second duration starting at frame 50
        frames = mapper.get_frame_range(
            duration_us=500_000,
            start_timestamp_us=5_000_000
        )
        assert frames == [50, 51, 52, 53, 54, 55]

    def test_get_frame_range_clamping(self, mapper):
        """Test frame range clamping to scene bounds"""
        # Duration exceeds scene end
        frames = mapper.get_frame_range(
            duration_us=20_000_000,
            start_timestamp_us=8_000_000
        )
        # Should be clamped to end of scene
        assert frames[-1] == 99

    def test_get_timestamp_range(self, mapper):
        """Test timestamp range from frame indices"""
        start_ts, end_ts = mapper.get_timestamp_range([10, 15, 20])
        assert start_ts == 1_000_000
        assert end_ts == 2_000_000

    def test_get_timestamp_range_empty(self, mapper):
        """Test timestamp range from empty list"""
        start_ts, end_ts = mapper.get_timestamp_range([])
        # Returns scene start
        assert start_ts == mapper.metadata.start_timestamp_us

    def test_get_timestamp_range_clamping(self, mapper):
        """Test frame index clamping in timestamp range"""
        # Out of range indices
        start_ts, end_ts = mapper.get_timestamp_range([-5, 150])
        # Clamped to valid range
        assert start_ts == 0
        assert end_ts == 9_900_000

    def test_validate_timestamp_valid(self, mapper):
        """Test valid timestamp validation"""
        is_valid, clamped = mapper.validate_timestamp(5_000_000)
        assert is_valid == True
        assert clamped == 5_000_000

    def test_validate_timestamp_below_start(self, mapper):
        """Test timestamp below start"""
        is_valid, clamped = mapper.validate_timestamp(-1_000_000)
        assert is_valid == False
        assert clamped == 0

    def test_validate_timestamp_above_end(self, mapper):
        """Test timestamp above end"""
        is_valid, clamped = mapper.validate_timestamp(20_000_000)
        assert is_valid == False
        assert clamped == 9_900_000


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
            intrinsic_matrix=np.eye(3),
            intrinsics_vary=False,
            has_static_scene=True,
            has_sky_scene=False,
            dynamic_object_ids=[],
            static_scene_size_mb=10.0,
            total_size_mb=10.0
        )

    def test_single_frame_scene(self, short_scene):
        """Test mapper with single frame scene"""
        mapper = FrameIndexMapper(short_scene)

        # All timestamps should return frame 0
        assert mapper.get_frame_index(timestamp_us=0) == 0
        assert mapper.get_frame_index(timestamp_us=1_000_000) == 0
        assert mapper.get_frame_index(frame_start_us=0, frame_end_us=100) == 0

        # Frame range should return [0]
        frames = mapper.get_frame_range(duration_us=1_000_000)
        assert frames == [0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])