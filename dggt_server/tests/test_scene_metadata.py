# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unit tests for scene_metadata module
"""

import pytest
import numpy as np
from dggt_server.scene_metadata import (
    DGGTSceneMetadata,
    FrameMetadata,
    ObjectMetadata,
    TrackIDMapping
)


class TestDGGTSceneMetadata:
    """Tests for DGGTSceneMetadata dataclass"""

    def test_timestamp_to_frame_conversion(self):
        """Test timestamp to frame index conversion"""
        meta = DGGTSceneMetadata(
            scene_id="test",
            scene_path="/path/to/scene",
            num_frames=20,
            fps=10.0,
            start_timestamp_us=0,
            end_timestamp_us=2_000_000,
            camera_width=518,
            camera_height=350,
            intrinsic_matrix=np.eye(3),
            intrinsics_vary=False,
            has_static_scene=True,
            has_sky_scene=True,
            dynamic_object_ids=[0],
            static_scene_size_mb=100.0,
            total_size_mb=120.0
        )

        assert meta.get_frame_index_from_timestamp(0) == 0
        assert meta.get_frame_index_from_timestamp(1_900_000) == 19
        assert meta.get_frame_index_from_timestamp(-100) == 0
        assert meta.get_frame_index_from_timestamp(5_000_000) == 19
        assert meta.get_frame_index_from_timestamp(500_000) == 5

    def test_frame_to_timestamp_conversion(self):
        """Test frame index to timestamp conversion"""
        meta = DGGTSceneMetadata(
            scene_id="test",
            scene_path="/path/to/scene",
            num_frames=20,
            fps=10.0,
            start_timestamp_us=0,
            end_timestamp_us=1_900_000,
            camera_width=518,
            camera_height=350,
            intrinsic_matrix=np.eye(3),
            intrinsics_vary=False,
            has_static_scene=True,
            has_sky_scene=True,
            dynamic_object_ids=[0],
            static_scene_size_mb=100.0,
            total_size_mb=120.0
        )

        assert meta.get_timestamp_from_frame_index(0) == 0
        assert meta.get_timestamp_from_frame_index(10) == 1_000_000
        assert meta.get_timestamp_from_frame_index(19) == 1_900_000


class TestTrackIDMapping:
    """Tests for TrackIDMapping class"""

    def test_auto_generate_track_id(self):
        mapper = TrackIDMapping(auto_prefix="dggt_obj")
        assert mapper.to_track_id(0) == "dggt_obj_0000"
        assert mapper.to_track_id(5) == "dggt_obj_0005"

    def test_extract_object_id(self):
        mapper = TrackIDMapping(auto_prefix="dggt_obj")
        assert mapper.to_object_id("dggt_obj_0003") == 3
        assert mapper.to_object_id("vehicle-001") == 1

    def test_explicit_mapping(self):
        mapper = TrackIDMapping(auto_prefix="dggt_obj")
        mapper.register("ego_vehicle", 0)
        assert mapper.to_object_id("ego_vehicle") == 0
        assert mapper.to_track_id(0) == "ego_vehicle"

    def test_invalid_track_id(self):
        mapper = TrackIDMapping(auto_prefix="dggt_obj")
        with pytest.raises(ValueError):
            mapper.to_object_id("no_numeric_id")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
