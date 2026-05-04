# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unit tests for scene_manager module
"""

import pytest
import os
from dggt_server.scene_manager import DGGTSceneManager
from dggt_server.exceptions import SceneNotFoundError

TEST_SCENE_BASE = "/home/junchuan/e2e/dggt/output/waymo/training/scene1"
TEST_SCENE_ID = "0328/001"


@pytest.fixture
def scene_manager():
    """Fixture for scene manager with auto-discovery"""
    manager = DGGTSceneManager(TEST_SCENE_BASE)
    manager.initialize(auto_discover=True)
    return manager


@pytest.fixture
def scene_manager_no_discover():
    """Fixture for scene manager without auto-discovery"""
    manager = DGGTSceneManager(TEST_SCENE_BASE)
    manager.initialize(auto_discover=False)
    return manager


class TestDGGTSceneManager:
    def test_initialize_auto_discover(self):
        """Test auto-discovery of scenes"""
        manager = DGGTSceneManager(TEST_SCENE_BASE)
        manager.initialize(auto_discover=True)
        scenes = manager.list_scenes()
        assert len(scenes) > 0
        assert TEST_SCENE_ID in scenes

    def test_initialize_manual(self, scene_manager_no_discover):
        """Test manual initialization without auto-discovery"""
        # Should not have any scenes cached initially
        scenes = scene_manager_no_discover.list_scenes()
        assert len(scenes) == 0

    def test_register_scene_mapping(self, scene_manager):
        """Test external ID to internal ID mapping"""
        external_id = "clipgt-7f360cc2-test"
        internal_id = TEST_SCENE_ID

        scene_manager.register_scene(external_id, internal_id)

        # Verify mapping works - get_scene should resolve external_id
        meta = scene_manager.get_scene(external_id)
        assert meta.scene_id == internal_id

    def test_get_scene_cached(self, scene_manager):
        """Test cache hit when getting scene"""
        # Scene should already be cached from auto-discover
        assert TEST_SCENE_ID in scene_manager._metadata_cache

        # Get scene - should return cached metadata
        meta = scene_manager.get_scene(TEST_SCENE_ID)
        assert meta.scene_id == TEST_SCENE_ID
        assert meta.num_frames == 20
        assert meta.has_static_scene == True
        assert meta.camera_width == 518
        assert meta.camera_height == 350

    def test_get_scene_on_demand(self, scene_manager_no_discover):
        """Test on-demand loading when scene not cached"""
        # Manager initialized without auto-discover, so cache is empty
        assert len(scene_manager_no_discover._metadata_cache) == 0

        # Get scene should load on-demand
        meta = scene_manager_no_discover.get_scene(TEST_SCENE_ID)
        assert meta.scene_id == TEST_SCENE_ID
        assert meta.num_frames == 20

        # Now it should be cached
        assert TEST_SCENE_ID in scene_manager_no_discover._metadata_cache

    def test_get_frame_metadata(self, scene_manager):
        """Test frame metadata retrieval"""
        frame_meta = scene_manager.get_frame_metadata(TEST_SCENE_ID, 0)

        assert frame_meta.frame_idx == 0
        assert frame_meta.c2w_matrix.shape == (4, 4)
        assert frame_meta.intrinsic_matrix.shape == (3, 3)
        assert frame_meta.width == 518
        assert frame_meta.height == 350

    def test_get_frame_metadata_by_timestamp(self, scene_manager):
        """Test frame metadata retrieval by timestamp"""
        # Get scene metadata to find valid timestamp range
        scene_meta = scene_manager.get_scene(TEST_SCENE_ID)

        # Use a timestamp in the middle of the scene
        mid_timestamp = scene_meta.start_timestamp_us + 500000  # 0.5 seconds after start

        frame_meta = scene_manager.get_frame_metadata_by_timestamp(TEST_SCENE_ID, mid_timestamp)

        # Should return frame 5 (at 10 fps, 0.5s = frame 5)
        assert frame_meta.frame_idx == 5
        assert frame_meta.timestamp_us == mid_timestamp
        assert frame_meta.c2w_matrix.shape == (4, 4)

    def test_list_scenes(self, scene_manager):
        """Test scene list (verify deduplication)"""
        scenes = scene_manager.list_scenes()

        # Should have at least one scene
        assert len(scenes) > 0
        assert TEST_SCENE_ID in scenes

        # Verify no duplicates (list_scenes combines cache and mapping keys)
        # After auto-discover, all scenes are in cache, no mappings yet
        unique_scenes = set(scenes)
        assert len(unique_scenes) == len(scenes)

    def test_list_scenes_with_mapping(self, scene_manager):
        """Test scene list including mapped external IDs"""
        external_id = "clipgt-external-test"
        scene_manager.register_scene(external_id, TEST_SCENE_ID)

        scenes = scene_manager.list_scenes()

        # Should include both internal ID and external ID
        assert TEST_SCENE_ID in scenes
        assert external_id in scenes

        # External ID maps to same scene, so both should resolve
        meta_internal = scene_manager.get_scene(TEST_SCENE_ID)
        meta_external = scene_manager.get_scene(external_id)
        assert meta_internal.scene_id == meta_external.scene_id

    def test_clear_cache(self, scene_manager):
        """Test cache clearing"""
        # Verify cache has content
        assert len(scene_manager._metadata_cache) > 0

        # Clear cache
        scene_manager.clear_cache()

        # Cache should be empty
        assert len(scene_manager._metadata_cache) == 0

        # But should still be able to load on-demand
        meta = scene_manager.get_scene(TEST_SCENE_ID)
        assert meta.scene_id == TEST_SCENE_ID

        # And now cached again
        assert TEST_SCENE_ID in scene_manager._metadata_cache

    def test_get_scene_not_found(self, scene_manager_no_discover):
        """Test SceneNotFoundError for nonexistent scene"""
        with pytest.raises(SceneNotFoundError):
            scene_manager_no_discover.get_scene("nonexistent/scene")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])