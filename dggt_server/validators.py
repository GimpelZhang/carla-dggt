# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Scene Validators

Scene validation functions for checking data integrity.
"""

import os
import glob
import re
import json
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


def validate_scene_integrity(scene_path: str) -> Tuple[bool, List[str]]:
    """
    Validate scene data integrity.

    Args:
        scene_path: Absolute path to scene directory

    Returns:
        (is_valid, issues): Whether valid, and list of issues found
    """
    issues = []

    # Required files check
    required_files = [
        "gaussians/static_scene.ply",
    ]

    for rel_path in required_files:
        full_path = os.path.join(scene_path, rel_path)
        if not os.path.exists(full_path):
            issues.append(f"Missing required file: {rel_path}")

    # Optional files warning
    optional_files = [
        "gaussians/sky_scene.ply",
    ]

    for rel_path in optional_files:
        full_path = os.path.join(scene_path, rel_path)
        if not os.path.exists(full_path):
            issues.append(f"Missing optional file: {rel_path} (will use default)")

    # Frame integrity check
    ego_dir = os.path.join(scene_path, "ego_pose")
    ego_files = sorted(glob.glob(os.path.join(ego_dir, "frame_*_ego.json")))

    if not ego_files:
        issues.append("No ego pose files found in ego_pose directory")
    else:
        # Extract frame indices and check contiguity
        indices = []
        for f in ego_files:
            filename = os.path.basename(f)  # Extract filename first
            match = re.search(r'frame_(\d+)_ego\.json', filename)
            if match:
                indices.append(int(match.group(1)))

        if indices:
            expected = list(range(min(indices), max(indices) + 1))
            if indices != expected:
                issues.append(
                    f"Frame indices are not contiguous: expected {expected}, "
                    f"found {indices}"
                )

        # JSON format validation - sample first ego file
        sample_file = ego_files[0]
        try:
            with open(sample_file) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON format in {os.path.basename(sample_file)}: {e}")
        except Exception as e:
            issues.append(f"Failed to read {os.path.basename(sample_file)}: {e}")

    # Dynamic objects check (optional)
    obj_dir = os.path.join(scene_path, "dynamic_objects")
    obj_files = glob.glob(os.path.join(obj_dir, "frame_*_objects.json"))
    if not obj_files:
        logger.info(f"No dynamic objects files found for scene at {scene_path}")

    is_valid = not any("required" in i.lower() for i in issues)
    return is_valid, issues
