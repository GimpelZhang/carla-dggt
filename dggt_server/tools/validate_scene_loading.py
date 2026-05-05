# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Scene Loading Validation Script

Validates scene loading functionality.

Usage:
    python validate_scene_loading.py --scene-base-path /path/to/output
    python validate_scene_loading.py --config config.yaml
"""

import argparse
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dggt_server.config import DGGTServerConfig
from dggt_server.scene_manager import DGGTSceneManager
from dggt_server.scene_loader import DGGTSceneLoader
from dggt_server.tools.scene_analyzer import analyze_scene, print_scene_report

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validate DGGT scene loading")
    parser.add_argument(
        "--scene-base-path",
        type=str,
        default="/home/junchuan/e2e/dggt/output/waymo/training/scene1",
        help="Scene base directory path"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (optional)"
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default="0328/001",
        help="Specific scene ID to validate (optional)"
    )
    
    args = parser.parse_args()

    # Load configuration
    if args.config and os.path.exists(args.config):
        config = DGGTServerConfig.from_yaml(args.config)
        scene_base_path = config.scene_base_path
    else:
        scene_base_path = args.scene_base_path
        config = DGGTServerConfig(scene_base_path=scene_base_path)

    print("=== DGGT Scene Loading Validation ===")
    print(f"Scene Base Path: {scene_base_path}")

    # 1. Initialize scene manager
    manager = DGGTSceneManager(scene_base_path, default_fps=config.default_fps)
    manager.initialize(auto_discover=config.auto_discover)

    print(f"Discovered Scenes: {len(manager.list_scenes())}")

    # 2. Validate each scene
    for scene_id in manager.list_scenes():
        try:
            meta = manager.get_scene(scene_id)
            print(f"\n--- Scene: {scene_id} ---")
            print(f"  Frames: {meta.num_frames}")
            print(f"  Resolution: {meta.camera_width} x {meta.camera_height}")
            print(f"  Dynamic Objects: {len(meta.dynamic_object_ids)}")
            print(f"  Static Size: {meta.static_scene_size_mb:.1f} MB")
            print(f"  Intrinsics Vary: {meta.intrinsics_vary}")

            # Validate frame loading
            if meta.num_frames > 0:
                frame0 = manager.get_frame_metadata(scene_id, 0)
                print(f"  Frame 0 C2W position: {frame0.c2w_matrix[:3, 3].tolist()}")
                print(f"  Frame 0 Objects: {len(frame0.objects)}")

                # Test last frame
                last_frame = manager.get_frame_metadata(scene_id, meta.num_frames - 1)
                print(f"  Last frame ({meta.num_frames - 1}) loaded successfully")

        except Exception as e:
            logger.error(f"Failed to validate scene {scene_id}: {e}")

    # 3. Run scene analyzer on specific scene
    if args.scene_id:
        full_scene_path = os.path.join(scene_base_path, args.scene_id)
        print(f"\n=== Scene Analyzer Report ===")
        result = analyze_scene(full_scene_path)
        print_scene_report(result)

    print("\n=== Validation Complete ===")


if __name__ == "__main__":
    main()
