# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Scene Loader

Scene loader implementation for loading DGGT scene data from disk.

IMPORTANT: Must import coordinate transform utilities from utils.py (lines 70-89, 98-116, 119-159)
and NEVER reimplement them locally.
"""

import os
import glob
import json
import logging
from typing import List, Dict

import numpy as np

from .exceptions import (
    SceneNotFoundError,
    MissingStaticSceneError,
    EmptyEgoPoseError,
    InvalidJSONError,
    FrameIndexOutOfRangeError,
)
from .scene_metadata import (
    DGGTSceneMetadata,
    FrameMetadata,
    ObjectMetadata,
)
from .validators import validate_scene_integrity

logger = logging.getLogger(__name__)


class DGGTSceneLoader:
    """
    DGGT Scene Loader

    Responsibilities:
    - Load scene from directory path
    - Parse scene metadata
    - Load single frame data (ego_pose, dynamic_objects)
    """

    def __init__(self, scene_base_path: str, default_fps: float = 10.0):
        """
        Args:
            scene_base_path: Scene root directory
                e.g.: "/home/junchuan/e2e/dggt/output/waymo/training/scene1"
            default_fps: Default frame rate (Waymo = 10 Hz)
        """
        self.scene_base_path = scene_base_path
        self.default_fps = default_fps

    def discover_scenes(self) -> List[str]:
        """
        Discover all scenes in the scene directory

        Returns:
            scene_id list, e.g. ["0328/001", "0328/002"]
        """
        scenes = []
        # Walk directory, find directories containing gaussians/static_scene.ply
        for root, dirs, files in os.walk(self.scene_base_path):
            gaussians_path = os.path.join(root, "gaussians")
            if os.path.isdir(gaussians_path):
                static_ply = os.path.join(gaussians_path, "static_scene.ply")
                if os.path.exists(static_ply):
                    # Extract relative path as scene_id
                    rel_path = os.path.relpath(root, self.scene_base_path)
                    scenes.append(rel_path)
        return sorted(scenes)

    def load_scene(self, scene_id: str) -> DGGTSceneMetadata:
        """
        Load scene metadata

        Args:
            scene_id: Scene ID, e.g. "0328/001"

        Returns:
            DGGTSceneMetadata complete metadata

        Raises:
            SceneNotFoundError: Scene directory not found
            MissingStaticSceneError: Static scene PLY missing
            EmptyEgoPoseError: No ego pose data
        """
        scene_path = os.path.join(self.scene_base_path, scene_id)

        # 1. Check scene directory exists (Critical)
        if not os.path.exists(scene_path):
            raise SceneNotFoundError(scene_id, scene_path)

        # 2. Check static_scene.ply (Critical)
        static_ply = os.path.join(scene_path, "gaussians", "static_scene.ply")
        if not os.path.exists(static_ply):
            raise MissingStaticSceneError(scene_id)

        # 2.5 Validate scene integrity
        is_valid, validation_issues = validate_scene_integrity(scene_path)
        if not is_valid:
            logger.warning(f"Scene {scene_id} validation issues: {validation_issues}")
        elif validation_issues:
            logger.info(f"Scene {scene_id} optional validation notes: {validation_issues}")

        # 3. Count frames from ego_pose directory
        ego_dir = os.path.join(scene_path, "ego_pose")
        ego_files = glob.glob(os.path.join(ego_dir, "frame_*_ego.json"))
        if not ego_files:
            raise EmptyEgoPoseError(scene_id)
        num_frames = len(ego_files)

        # 4. Load first frame to get camera info
        first_ego = self.load_frame_ego(scene_path, 0)

        # 5. Load first frame dynamic objects
        first_objects = self.load_frame_objects(scene_path, 0)

        # 6. Detect intrinsics variation (sample all frames)
        all_intrinsics = []
        for frame_idx in range(num_frames):
            try:
                ego_data = self.load_frame_ego(scene_path, frame_idx)
                all_intrinsics.append(np.array(ego_data['camera_intrinsics']))
            except (SceneNotFoundError, InvalidJSONError):
                logger.warning(f"Could not load ego data for frame {frame_idx}")
                continue

        # Calculate focal length variance
        intrinsics_vary = False
        if len(all_intrinsics) > 1:
            fx_values = [K[0, 0] for K in all_intrinsics]
            fy_values = [K[1, 1] for K in all_intrinsics]
            fx_mean = np.mean(fx_values) if np.mean(fx_values) != 0 else 1.0
            fy_mean = np.mean(fy_values) if np.mean(fy_values) != 0 else 1.0
            fx_variance = np.var(fx_values) / (fx_mean ** 2)
            fy_variance = np.var(fy_values) / (fy_mean ** 2)
            intrinsics_vary = (fx_variance > 0.0001) or (fy_variance > 0.0001)  # > 1%

            if intrinsics_vary:
                logger.warning(
                    f"Scene {scene_id} has varying camera intrinsics "
                    f"(fx variance: {fx_variance:.6f}, fy variance: {fy_variance:.6f}). "
                    f"Per-frame intrinsics will be used in FrameMetadata."
                )

        # 7. Calculate file sizes
        static_size_mb = os.path.getsize(static_ply) / (1024 * 1024) if os.path.exists(static_ply) else 0

        # 8. Check for sky scene (optional)
        sky_ply = os.path.join(scene_path, "gaussians", "sky_scene.ply")
        has_sky_scene = os.path.exists(sky_ply)
        if not has_sky_scene:
            logger.info(f"Scene '{scene_id}' has no sky scene PLY")

        # 9. Build metadata
        metadata = DGGTSceneMetadata(
            scene_id=scene_id,
            scene_path=scene_path,
            num_frames=num_frames,
            fps=self.default_fps,
            start_timestamp_us=0,  # DGGT default start is 0
            end_timestamp_us=int((num_frames - 1) * 1_000_000 / self.default_fps),
            camera_width=first_ego['camera']['width'],
            camera_height=first_ego['camera']['height'],
            intrinsic_matrix=np.array(first_ego['camera_intrinsics']),
            intrinsics_vary=intrinsics_vary,
            has_static_scene=True,
            has_sky_scene=has_sky_scene,
            dynamic_object_ids=[obj['object_id'] for obj in first_objects],
            static_scene_size_mb=static_size_mb,
            total_size_mb=self._calculate_total_size(scene_path),
        )

        return metadata

    def load_frame_ego(self, scene_path: str, frame_idx: int) -> Dict:
        """
        Load single frame ego_pose JSON

        Args:
            scene_path: Scene directory absolute path
            frame_idx: Frame index

        Returns:
            ego_data: Dictionary with camera data

        Raises:
            SceneNotFoundError: Frame file not found
            InvalidJSONError: JSON parsing failed
        """
        ego_file = os.path.join(scene_path, "ego_pose", f"frame_{frame_idx:04d}_ego.json")

        try:
            with open(ego_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise SceneNotFoundError(f"frame_{frame_idx:04d}", ego_file)
        except json.JSONDecodeError as e:
            raise InvalidJSONError(ego_file, e)

    def load_frame_objects(self, scene_path: str, frame_idx: int) -> List[Dict]:
        """
        Load single frame dynamic_objects JSON

        Args:
            scene_path: Scene directory absolute path
            frame_idx: Frame index

        Returns:
            objects_data: List of object dictionaries (empty list if no file)

        Raises:
            InvalidJSONError: JSON parsing failed
        """
        obj_file = os.path.join(scene_path, "dynamic_objects", f"frame_{frame_idx:04d}_objects.json")

        # File not exists -> return empty list (Info - normal case)
        if not os.path.exists(obj_file):
            logger.debug(f"No dynamic objects file for frame {frame_idx}")
            return []

        try:
            with open(obj_file, 'r') as f:
                data = json.load(f)
                return data if data else []
        except json.JSONDecodeError as e:
            raise InvalidJSONError(obj_file, e)

    def load_frame_metadata(self, scene_path: str, frame_idx: int, num_frames: int) -> FrameMetadata:
        """
        Load complete frame metadata

        Args:
            scene_path: Scene directory absolute path
            frame_idx: Frame index
            num_frames: Total number of frames (for range checking)

        Returns:
            FrameMetadata complete frame metadata
        """
        # Frame index range check (Warning - auto-correct)
        if frame_idx < 0 or frame_idx >= num_frames:
            original_idx = frame_idx
            frame_idx = max(0, min(frame_idx, num_frames - 1))
            logger.warning(
                f"Frame index {original_idx} out of range [0, {num_frames-1}], "
                f"corrected to {frame_idx}"
            )

        ego_data = self.load_frame_ego(scene_path, frame_idx)
        objects_data = self.load_frame_objects(scene_path, frame_idx)

        objects_meta = [
            ObjectMetadata(
                object_id=obj['object_id'],
                pose_world=np.array(obj['pose_world']),
                dimensions=np.array(obj['dimensions'])
            )
            for obj in objects_data
        ]

        return FrameMetadata(
            frame_idx=frame_idx,
            timestamp_us=int(frame_idx * 1_000_000 / self.default_fps),
            c2w_matrix=np.array(ego_data['camera_extrinsics_world']),
            intrinsic_matrix=np.array(ego_data['camera_intrinsics']),  # Current frame actual intrinsics
            width=ego_data['camera']['width'],
            height=ego_data['camera']['height'],
            objects=objects_meta
        )

    def _calculate_total_size(self, scene_path: str) -> float:
        """Calculate total size of scene directory in MB"""
        total_size = 0
        for root, dirs, files in os.walk(scene_path):
            for f in files:
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)
