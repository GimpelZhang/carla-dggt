# Utility functions for loading GT pose data from datasets.
import os
import numpy as np
import torch
from typing import List


def load_ego_pose_translations(
    scene_dir: str,
    frame_indices: List[int],
    dataset: str = "waymo",
) -> torch.Tensor:
    """
    Load GT ego_pose translation vectors for scale recovery.

    Args:
        scene_dir: Scene directory path (e.g., "data/waymo/processed/validation/000")
        frame_indices: List of frame indices (e.g., [0, 5, 10, 15])
        dataset: Dataset type ("waymo" or "nuscenes"), determines file path convention.

    Returns:
        torch.Tensor: (S, 3) translation vector sequence from ego_pose matrices

    Notes:
        - Both Waymo and nuScenes store 4x4 Vehicle-to-World transformation matrices.
        - Translation component [:3, 3] represents vehicle position in world coordinates.
        - World coordinates are large values (~km level), but scale_factor calculation
          uses relative frame-to-frame movement, eliminating absolute position bias.
        - Path conventions:
            - waymo:    {scene_dir}/ego_pose/{frame_idx:03d}.txt
            - nuscenes: {scene_dir}/extrinsics/{frame_idx:03d}_0.txt
    """
    translations = []

    for frame_idx in frame_indices:
        if dataset == "waymo":
            ego_path = os.path.join(scene_dir, "ego_pose", f"{frame_idx:03d}.txt")
        elif dataset == "nuscenes":
            ego_path = os.path.join(scene_dir, "extrinsics", f"{frame_idx:03d}_0.txt")
        else:
            raise ValueError(f"Unsupported dataset: {dataset}. Must be 'waymo' or 'nuscenes'.")

        if not os.path.exists(ego_path):
            raise FileNotFoundError(f"ego_pose file not found: {ego_path}")

        # Load 4x4 transformation matrix
        matrix = np.loadtxt(ego_path)

        # Extract translation component ([:3, 3])
        translation = matrix[:3, 3]
        translations.append(translation)

    # Stack into (S, 3) tensor
    translations_array = np.stack(translations, axis=0)
    return torch.tensor(translations_array, dtype=torch.float32)