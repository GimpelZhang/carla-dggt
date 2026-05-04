# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Scene Manager

Scene cache manager for managing multiple scenes, on-demand loading, metadata caching.
"""

import logging
from typing import Dict, List, Optional

from .scene_loader import DGGTSceneLoader
from .scene_metadata import DGGTSceneMetadata, FrameMetadata

logger = logging.getLogger(__name__)


class DGGTSceneManager:
    """
    Scene Manager

    Responsibilities:
    - Manage scene_id -> DGGTSceneMetadata mapping
    - Scene metadata caching
    - On-demand scene loading
    - Support flexible scene_id mapping
    """

    def __init__(self, scene_base_path: str, default_fps: float = 10.0):
        """
        Args:
            scene_base_path: Scene root directory path
            default_fps: Default frame rate
        """
        self.loader = DGGTSceneLoader(scene_base_path, default_fps)
        self._metadata_cache: Dict[str, DGGTSceneMetadata] = {}
        self._id_mapping: Dict[str, str] = {}  # External ID -> Internal ID
        self._initialized = False

    def initialize(self, auto_discover: bool = True) -> None:
        """
        Initialize scene manager

        Args:
            auto_discover: Whether to auto-discover all scenes
        """
        if auto_discover:
            discovered = self.loader.discover_scenes()
            for scene_id in discovered:
                try:
                    self._metadata_cache[scene_id] = self.loader.load_scene(scene_id)
                    logger.info(f"Loaded scene: {scene_id}")
                except Exception as e:
                    logger.warning(f"Failed to load scene {scene_id}: {e}")
        self._initialized = True

    def register_scene(self, external_id: str, internal_id: str) -> None:
        """
        Register scene ID mapping

        Used when external system uses different ID naming

        Args:
            external_id: ID used by external system
            internal_id: DGGT internal scene ID

            e.g.:
            register_scene("clipgt-7f360cc2-xxx", "0328/001")
        """
        self._id_mapping[external_id] = internal_id
        logger.info(f"Registered scene mapping: {external_id} -> {internal_id}")

    def get_scene(self, scene_id: str) -> DGGTSceneMetadata:
        """
        Get scene metadata

        Args:
            scene_id: Scene ID (can be internal ID or mapped external ID)

        Returns:
            DGGTSceneMetadata

        Raises:
            SceneNotFoundError: Scene not found
        """
        # 1. Check mapping
        actual_id = self._id_mapping.get(scene_id, scene_id)

        # 2. Check cache
        if actual_id in self._metadata_cache:
            return self._metadata_cache[actual_id]

        # 3. On-demand load
        metadata = self.loader.load_scene(actual_id)
        self._metadata_cache[actual_id] = metadata
        return metadata

    def get_frame_metadata(self, scene_id: str, frame_idx: int) -> FrameMetadata:
        """
        Get single frame metadata

        Args:
            scene_id: Scene ID
            frame_idx: Frame index

        Returns:
            FrameMetadata
        """
        meta = self.get_scene(scene_id)
        return self.loader.load_frame_metadata(meta.scene_path, frame_idx, meta.num_frames)

    def get_frame_metadata_by_timestamp(self, scene_id: str, timestamp_us: int) -> FrameMetadata:
        """
        Get frame metadata by timestamp

        Args:
            scene_id: Scene ID
            timestamp_us: Timestamp in microseconds

        Returns:
            FrameMetadata
        """
        meta = self.get_scene(scene_id)
        frame_idx = meta.get_frame_index_from_timestamp(timestamp_us)
        return self.get_frame_metadata(scene_id, frame_idx)

    def list_scenes(self) -> List[str]:
        """List all available scene IDs"""
        return list(set(self._metadata_cache.keys()) | set(self._id_mapping.keys()))

    def clear_cache(self) -> None:
        """Clear cache"""
        self._metadata_cache.clear()
        logger.info("Scene cache cleared")
