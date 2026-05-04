# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Scene Analyzer

Scene structure analysis tool for analyzing DGGT scene assets and validating data integrity.
"""

import os
import glob
import json
import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SceneAnalysisResult:
    """Scene analysis result"""

    # Basic info
    scene_path: str
    num_frames: int
    frame_indices: List[int]

    # Scene content
    has_static: bool
    has_sky: bool
    static_size_mb: float
    sky_size_mb: float

    # Camera specs (from first frame)
    camera_width: int
    camera_height: int
    intrinsic_matrix: np.ndarray  # 3x3

    # Dynamic objects
    object_ids: List[int]
    object_dimensions: Dict[int, np.ndarray]  # object_id -> [L, W, H]

    # Total size
    total_size_mb: float

    # Status
    is_valid: bool
    issues: List[str]


def analyze_scene(scene_path: str) -> SceneAnalysisResult:
    """
    Analyze DGGT scene directory structure

    Args:
        scene_path: Scene directory absolute path

    Returns:
        SceneAnalysisResult with complete analysis
    """
    issues = []

    # Check scene directory exists
    if not os.path.exists(scene_path):
        return SceneAnalysisResult(
            scene_path=scene_path,
            num_frames=0,
            frame_indices=[],
            has_static=False,
            has_sky=False,
            static_size_mb=0,
            sky_size_mb=0,
            camera_width=0,
            camera_height=0,
            intrinsic_matrix=np.eye(3),
            object_ids=[],
            object_dimensions={},
            total_size_mb=0,
            is_valid=False,
            issues=[f"Scene directory not found: {scene_path}"]
        )

    # 1. Check gaussians/static_scene.ply
    static_ply = os.path.join(scene_path, "gaussians", "static_scene.ply")
    has_static = os.path.exists(static_ply)
    static_size_mb = os.path.getsize(static_ply) / (1024 * 1024) if has_static else 0
    if not has_static:
        issues.append("Missing required file: gaussians/static_scene.ply")

    # 2. Check sky_scene.ply (optional)
    sky_ply = os.path.join(scene_path, "gaussians", "sky_scene.ply")
    has_sky = os.path.exists(sky_ply)
    sky_size_mb = os.path.getsize(sky_ply) / (1024 * 1024) if has_sky else 0

    # 3. Count ego_pose frames
    ego_dir = os.path.join(scene_path, "ego_pose")
    ego_files = sorted(glob.glob(os.path.join(ego_dir, "frame_*_ego.json")))
    num_frames = len(ego_files)

    # Extract frame indices
    frame_indices = []
    for f in ego_files:
        match = re.search(r'frame_(\d+)_ego\.json', f)
        if match:
            frame_indices.append(int(match.group(1)))

    if not ego_files:
        issues.append("No ego_pose files found")
    else:
        # Check frame sequence contiguity
        expected = list(range(min(frame_indices), max(frame_indices) + 1))
        if frame_indices != expected:
            issues.append(f"Frame indices not contiguous: expected {expected}, found {frame_indices}")

    # 4. Load first frame for camera specs
    camera_width = 0
    camera_height = 0
    intrinsic_matrix = np.eye(3)
    object_ids = []
    object_dimensions = {}

    if ego_files:
        try:
            with open(ego_files[0], 'r') as f:
                first_ego = json.load(f)
            camera_width = first_ego.get('camera', {}).get('width', 0)
            camera_height = first_ego.get('camera', {}).get('height', 0)
            intrinsic_matrix = np.array(first_ego.get('camera_intrinsics', np.eye(3)))
        except (json.JSONDecodeError, Exception) as e:
            issues.append(f"Failed to parse first ego file: {e}")

    # 5. Load dynamic objects from first frame
    obj_dir = os.path.join(scene_path, "dynamic_objects")
    obj_files = glob.glob(os.path.join(obj_dir, "frame_*_objects.json"))
    
    if obj_files:
        try:
            with open(sorted(obj_files)[0], 'r') as f:
                first_objects = json.load(f)
            object_ids = [obj['object_id'] for obj in first_objects]
            object_dimensions = {
                obj['object_id']: np.array(obj['dimensions'])
                for obj in first_objects
            }
        except (json.JSONDecodeError, Exception) as e:
            issues.append(f"Failed to parse first objects file: {e}")

    # 6. Calculate total size
    total_size = 0
    for root, dirs, files in os.walk(scene_path):
        for f in files:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)
    total_size_mb = total_size / (1024 * 1024)

    # Determine validity
    is_valid = has_static and num_frames > 0 and not any("Missing required" in i for i in issues)

    return SceneAnalysisResult(
        scene_path=scene_path,
        num_frames=num_frames,
        frame_indices=frame_indices,
        has_static=has_static,
        has_sky=has_sky,
        static_size_mb=static_size_mb,
        sky_size_mb=sky_size_mb,
        camera_width=camera_width,
        camera_height=camera_height,
        intrinsic_matrix=intrinsic_matrix,
        object_ids=object_ids,
        object_dimensions=object_dimensions,
        total_size_mb=total_size_mb,
        is_valid=is_valid,
        issues=issues
    )


def print_scene_report(result: SceneAnalysisResult) -> None:
    """
    Print scene analysis report

    Args:
        result: SceneAnalysisResult to print
    """
    print(f"Scene Analysis Report for: {result.scene_path}")
    print("-" * 50)
    print(f"Frames: {result.num_frames} (frame_{min(result.frame_indices):04d} to frame_{max(result.frame_indices):04d})" if result.frame_indices else "Frames: 0")
    
    static_status = "OK" if result.has_static else "MISSING"
    print(f"Static Scene: {result.static_size_mb:.1f} MB ({static_status})")
    
    sky_status = "OK" if result.has_sky else "N/A"
    print(f"Sky Scene: {result.sky_size_mb:.1f} MB ({sky_status})")
    
    print(f"\nCamera Specs (Frame 0):")
    print(f"  Resolution: {result.camera_width} x {result.camera_height}")
    print(f"  Intrinsics K: {result.intrinsic_matrix.tolist()}")
    
    print(f"\nDynamic Object IDs: {result.object_ids}")
    for obj_id, dims in result.object_dimensions.items():
        print(f"  Object {obj_id} dimensions: {dims.tolist()} (meters)")
    
    print(f"\nTotal Size: {result.total_size_mb:.1f} MB")
    
    status = "VALID" if result.is_valid else "INVALID"
    print(f"Status: {status}")
    
    if result.issues:
        print("\nIssues:")
        for issue in result.issues:
            print(f"  - {issue}")
