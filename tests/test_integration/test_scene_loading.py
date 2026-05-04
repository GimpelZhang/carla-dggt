# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Integration tests for Scene Loading

Tests the DGGTSceneLoader and DGGTSceneManager classes:
- Load scene metadata from disk
- Load frame ego data
- Load frame objects
- Scene manager caching
- Scene metadata methods (timestamp <-> frame index)

Uses mocked scene data fixtures for testing.
"""

import pytest
import numpy as np
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules under test
try:
    from dggt_server.scene_loader import DGGTSceneLoader
    from dggt_server.scene_manager import DGGTSceneManager
    from dggt_server.scene_metadata import DGGTSceneMetadata, FrameMetadata
    from dggt_server.exceptions import SceneNotFoundError, InvalidJSONError
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from dggt_server.scene_loader import DGGTSceneLoader
    from dggt_server.scene_manager import DGGTSceneManager
    from dggt_server.scene_metadata import DGGTSceneMetadata, FrameMetadata
    from dggt_server.exceptions import SceneNotFoundError, InvalidJSONError


@pytest.fixture
def mock_scene_dir():
    """Create a temporary mock scene directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        scene_path = Path(tmpdir) / "test_scene"
        scene_path.mkdir()

        # Create gaussians directory with static_scene.ply
        gaussians_dir = scene_path / "gaussians"
        gaussians_dir.mkdir()
        static_ply = gaussians_dir / "static_scene.ply"
        static_ply.write_bytes(b"mock_ply_content")

        # Create ego_pose directory with frame files
        ego_dir = scene_path / "ego_pose"
        ego_dir.mkdir()

        # Create 5 frames of ego pose data
        for i in range(5):
            ego_data = {
                "camera": {
                    "width": 1036,
                    "height": 700
                },
                "camera_intrinsics": [
                    [1000.0, 0.0, 518.0],
                    [0.0, 1000.0, 350.0],
                    [0.0, 0.0, 1.0]
                ],
                "camera_extrinsics_world": np.eye(4).tolist()
            }
            ego_file = ego_dir / f"frame_{i:04d}_ego.json"
            with open(ego_file, 'w') as f:
                json.dump(ego_data, f)

        # Create dynamic_objects directory with frame files
        objects_dir = scene_path / "dynamic_objects"
        objects_dir.mkdir()

        # Create dynamic objects for first frame
        objects_data = [
            {
                "object_id": 0,
                "track_id": "vehicle_0",
                "pose_world": np.eye(4).tolist(),
                "dimensions": [4.5, 2.0, 1.5]
            },
            {
                "object_id": 1,
                "track_id": "vehicle_1",
                "pose_world": np.eye(4).tolist(),
                "dimensions": [4.0, 1.8, 1.4]
            }
        ]
        objects_file = objects_dir / "frame_0000_objects.json"
        with open(objects_file, 'w') as f:
            json.dump(objects_data, f)

        yield tmpdir


@pytest.fixture
def mock_scene_loader(mock_scene_dir):
    """Create a DGGTSceneLoader with mock scene directory"""
    return DGGTSceneLoader(mock_scene_dir, default_fps=10.0)


@pytest.fixture
def mock_scene_manager(mock_scene_dir):
    """Create a DGGTSceneManager with mock scene directory"""
    manager = DGGTSceneManager(mock_scene_dir, default_fps=10.0)
    return manager


@pytest.mark.integration
class TestSceneLoading:
    """Integration tests for DGGTSceneLoader"""

    def test_discover_scenes(self, mock_scene_loader):
        """Test that scene discovery finds test_scene"""
        scenes = mock_scene_loader.discover_scenes()
        assert len(scenes) == 1
        assert scenes[0] == "test_scene"

    def test_load_scene_metadata(self, mock_scene_loader):
        """Test loading scene metadata from disk"""
        metadata = mock_scene_loader.load_scene("test_scene")

        assert metadata.scene_id == "test_scene"
        assert metadata.num_frames == 5
        assert metadata.fps == 10.0
        assert metadata.camera_width == 1036
        assert metadata.camera_height == 700
        assert metadata.has_static_scene == True
        assert len(metadata.dynamic_object_ids) == 2
        assert metadata.dynamic_object_ids == [0, 1]

    def test_load_scene_not_found(self, mock_scene_loader):
        """Test loading non-existent scene raises SceneNotFoundError"""
        with pytest.raises(SceneNotFoundError):
            mock_scene_loader.load_scene("non_existent_scene")

    def test_load_frame_ego_data(self, mock_scene_loader, mock_scene_dir):
        """Test loading frame ego pose data"""
        scene_path = os.path.join(mock_scene_dir, "test_scene")
        ego_data = mock_scene_loader.load_frame_ego(scene_path, 0)

        assert "camera" in ego_data
        assert ego_data["camera"]["width"] == 1036
        assert ego_data["camera"]["height"] == 700
        assert "camera_intrinsics" in ego_data
        assert "camera_extrinsics_world" in ego_data

    def test_load_frame_ego_all_frames(self, mock_scene_loader, mock_scene_dir):
        """Test loading ego data for all frames"""
        scene_path = os.path.join(mock_scene_dir, "test_scene")

        for i in range(5):
            ego_data = mock_scene_loader.load_frame_ego(scene_path, i)
            assert ego_data is not None
            assert "camera" in ego_data

    def test_load_frame_ego_out_of_range(self, mock_scene_loader, mock_scene_dir):
        """Test loading ego data for out-of-range frame index"""
        scene_path = os.path.join(mock_scene_dir, "test_scene")

        # Frame index 10 is out of range (only 5 frames)
        with pytest.raises(SceneNotFoundError):
            mock_scene_loader.load_frame_ego(scene_path, 10)

    def test_load_frame_objects(self, mock_scene_loader, mock_scene_dir):
        """Test loading frame dynamic objects"""
        scene_path = os.path.join(mock_scene_dir, "test_scene")
        objects = mock_scene_loader.load_frame_objects(scene_path, 0)

        assert len(objects) == 2
        assert objects[0]["object_id"] == 0
        assert objects[1]["object_id"] == 1

    def test_load_frame_objects_empty_frame(self, mock_scene_loader, mock_scene_dir):
        """Test loading objects for frame with no objects file"""
        scene_path = os.path.join(mock_scene_dir, "test_scene")

        # Frame 1 has no objects file
        objects = mock_scene_loader.load_frame_objects(scene_path, 1)
        assert objects == []

    def test_load_frame_metadata(self, mock_scene_loader, mock_scene_dir):
        """Test loading complete frame metadata"""
        scene_path = os.path.join(mock_scene_dir, "test_scene")
        frame_meta = mock_scene_loader.load_frame_metadata(scene_path, 0, 5)

        assert frame_meta.frame_idx == 0
        assert frame_meta.timestamp_us == 0
        assert frame_meta.width == 1036
        assert frame_meta.height == 700
        assert frame_meta.c2w_matrix.shape == (4, 4)
        assert frame_meta.intrinsic_matrix.shape == (3, 3)
        assert len(frame_meta.objects) == 2

    def test_intrinsic_matrix_values(self, mock_scene_loader, mock_scene_dir):
        """Test that intrinsic matrix has correct values"""
        scene_path = os.path.join(mock_scene_dir, "test_scene")
        frame_meta = mock_scene_loader.load_frame_metadata(scene_path, 0, 5)

        K = frame_meta.intrinsic_matrix
        assert K[0, 0] == 1000.0  # fx
        assert K[1, 1] == 1000.0  # fy
        assert K[0, 2] == 518.0   # cx
        assert K[1, 2] == 350.0   # cy


@pytest.mark.integration
class TestSceneManager:
    """Integration tests for DGGTSceneManager"""

    def test_scene_manager_initialization(self, mock_scene_manager):
        """Test scene manager initialization"""
        assert mock_scene_manager.loader is not None
        assert mock_scene_manager._initialized == False

    def test_scene_manager_initialize_auto_discover(self, mock_scene_manager):
        """Test scene manager auto-discovery initialization"""
        mock_scene_manager.initialize(auto_discover=True)

        assert mock_scene_manager._initialized == True
        scenes = mock_scene_manager.list_scenes()
        assert len(scenes) == 1
        assert "test_scene" in scenes

    def test_scene_manager_get_scene(self, mock_scene_manager):
        """Test getting scene metadata from manager"""
        mock_scene_manager.initialize(auto_discover=True)

        metadata = mock_scene_manager.get_scene("test_scene")
        assert metadata.scene_id == "test_scene"
        assert metadata.num_frames == 5

    def test_scene_manager_get_scene_not_found(self, mock_scene_manager):
        """Test getting non-existent scene raises error"""
        mock_scene_manager.initialize(auto_discover=True)

        with pytest.raises(SceneNotFoundError):
            mock_scene_manager.get_scene("non_existent_scene")

    def test_scene_manager_caching(self, mock_scene_manager):
        """Test that scene metadata is cached"""
        mock_scene_manager.initialize(auto_discover=True)

        # First call loads from disk
        metadata1 = mock_scene_manager.get_scene("test_scene")

        # Second call should return cached version
        metadata2 = mock_scene_manager.get_scene("test_scene")

        # Both should be the same object (cached)
        assert metadata1 is metadata2
        assert "test_scene" in mock_scene_manager._metadata_cache

    def test_scene_manager_clear_cache(self, mock_scene_manager):
        """Test clearing scene manager cache"""
        mock_scene_manager.initialize(auto_discover=True)

        # Load scene to cache it
        mock_scene_manager.get_scene("test_scene")
        assert len(mock_scene_manager._metadata_cache) > 0

        # Clear cache
        mock_scene_manager.clear_cache()
        assert len(mock_scene_manager._metadata_cache) == 0

    def test_scene_manager_register_scene_mapping(self, mock_scene_manager):
        """Test registering scene ID mapping"""
        mock_scene_manager.initialize(auto_discover=True)

        # Register external ID mapping
        mock_scene_manager.register_scene("external_scene_id", "test_scene")

        # Get scene using external ID
        metadata = mock_scene_manager.get_scene("external_scene_id")
        assert metadata.scene_id == "test_scene"

    def test_scene_manager_get_frame_metadata(self, mock_scene_manager):
        """Test getting frame metadata from manager"""
        mock_scene_manager.initialize(auto_discover=True)

        frame_meta = mock_scene_manager.get_frame_metadata("test_scene", 0)
        assert frame_meta.frame_idx == 0
        assert frame_meta.timestamp_us == 0

    def test_scene_manager_get_frame_metadata_by_timestamp(self, mock_scene_manager):
        """Test getting frame metadata by timestamp"""
        mock_scene_manager.initialize(auto_discover=True)

        # At timestamp 0, should be frame 0
        frame_meta = mock_scene_manager.get_frame_metadata_by_timestamp("test_scene", 0)
        assert frame_meta.frame_idx == 0

        # At timestamp 200_000 (200ms), should be frame 2 at 10Hz
        frame_meta = mock_scene_manager.get_frame_metadata_by_timestamp("test_scene", 200_000)
        assert frame_meta.frame_idx == 2

    def test_scene_manager_on_demand_loading(self, mock_scene_dir):
        """Test on-demand scene loading without auto-discovery"""
        manager = DGGTSceneManager(mock_scene_dir, default_fps=10.0)
        manager.initialize(auto_discover=False)

        # No scenes loaded initially
        assert len(manager.list_scenes()) == 0

        # On-demand load
        metadata = manager.get_scene("test_scene")
        assert metadata.scene_id == "test_scene"
        assert len(manager.list_scenes()) == 1


@pytest.mark.integration
class TestSceneMetadataMethods:
    """Integration tests for scene metadata timestamp/frame methods"""

    def test_get_frame_index_from_timestamp(self, mock_scene_loader):
        """Test frame index calculation from timestamp"""
        metadata = mock_scene_loader.load_scene("test_scene")

        # At timestamp 0, frame 0
        assert metadata.get_frame_index_from_timestamp(0) == 0

        # At timestamp 100_000 (100ms), frame 1 at 10Hz
        assert metadata.get_frame_index_from_timestamp(100_000) == 1

        # At timestamp 400_000 (400ms), frame 4 at 10Hz
        assert metadata.get_frame_index_from_timestamp(400_000) == 4

    def test_get_frame_index_from_timestamp_clamping(self, mock_scene_loader):
        """Test frame index clamping for out-of-range timestamps"""
        metadata = mock_scene_loader.load_scene("test_scene")

        # Negative timestamp -> clamped to 0
        assert metadata.get_frame_index_from_timestamp(-100_000) == 0

        # Excessive timestamp -> clamped to last frame (4)
        assert metadata.get_frame_index_from_timestamp(1_000_000) == 4

    def test_get_timestamp_from_frame_index(self, mock_scene_loader):
        """Test timestamp calculation from frame index"""
        metadata = mock_scene_loader.load_scene("test_scene")

        # Frame 0 -> timestamp 0
        assert metadata.get_timestamp_from_frame_index(0) == 0

        # Frame 1 -> timestamp 100_000 (100ms at 10Hz)
        assert metadata.get_timestamp_from_frame_index(1) == 100_000

        # Frame 4 -> timestamp 400_000 (400ms at 10Hz)
        assert metadata.get_timestamp_from_frame_index(4) == 400_000

    def test_get_timestamp_from_frame_index_clamping(self, mock_scene_loader):
        """Test timestamp clamping for out-of-range frame indices"""
        metadata = mock_scene_loader.load_scene("test_scene")

        # Negative frame index -> clamped to 0
        assert metadata.get_timestamp_from_frame_index(-5) == 0

        # Excessive frame index -> clamped to last frame (4)
        assert metadata.get_timestamp_from_frame_index(100) == 400_000

    def test_timestamp_frame_roundtrip(self, mock_scene_loader):
        """Test roundtrip conversion between timestamp and frame index"""
        metadata = mock_scene_loader.load_scene("test_scene")

        # For each frame, verify roundtrip
        for frame_idx in range(metadata.num_frames):
            timestamp = metadata.get_timestamp_from_frame_index(frame_idx)
            recovered_idx = metadata.get_frame_index_from_timestamp(timestamp)
            assert recovered_idx == frame_idx


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])