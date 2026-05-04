# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Integration tests for scene_loader module
"""

import pytest
import os
import numpy as np
from dggt_server.scene_loader import DGGTSceneLoader
from dggt_server.exceptions import SceneNotFoundError


TEST_SCENE_BASE = "/home/junchuan/e2e/dggt/output/waymo/training/scene1"
TEST_SCENE_ID = "0328/001"


@pytest.fixture
def scene_loader():
    return DGGTSceneLoader(TEST_SCENE_BASE)


@pytest.fixture
def test_scene_path():
    return os.path.join(TEST_SCENE_BASE, TEST_SCENE_ID)


class TestDGGTSceneLoader:
    def test_discover_scenes(self, scene_loader):
        scenes = scene_loader.discover_scenes()
        assert len(scenes) > 0
        assert TEST_SCENE_ID in scenes

    def test_load_scene_metadata(self, scene_loader):
        meta = scene_loader.load_scene(TEST_SCENE_ID)
        assert meta.scene_id == TEST_SCENE_ID
        assert meta.num_frames == 20
        assert meta.has_static_scene == True
        assert meta.camera_width == 518
        assert meta.camera_height == 350

    def test_load_frame_ego(self, scene_loader, test_scene_path):
        ego_data = scene_loader.load_frame_ego(test_scene_path, 0)
        assert "camera_intrinsics" in ego_data
        assert "camera_extrinsics_world" in ego_data

    def test_load_frame_objects(self, scene_loader, test_scene_path):
        objects = scene_loader.load_frame_objects(test_scene_path, 0)
        assert len(objects) > 0
        assert "object_id" in objects[0]
        assert "pose_world" in objects[0]

    def test_load_frame_metadata(self, scene_loader, test_scene_path):
        meta = scene_loader.load_frame_metadata(test_scene_path, 0, 20)
        assert meta.frame_idx == 0
        assert meta.c2w_matrix.shape == (4, 4)
        assert meta.intrinsic_matrix.shape == (3, 3)

    def test_scene_not_found(self, scene_loader):
        with pytest.raises(SceneNotFoundError):
            scene_loader.load_scene("nonexistent/scene")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
