# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
E2E Test Helper Functions

Provides utility functions for E2E tests that can be imported explicitly.
"""

import numpy as np
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_test_camera_pose(x: float = 0.0, y: float = 0.0, z: float = 2.0,
                            yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """
    Create 4x4 camera pose matrix from position and rotation.

    Args:
        x, y, z: Position in meters
        yaw, pitch, roll: Rotation angles in degrees

    Returns:
        4x4 transformation matrix (CARLA coordinate system)
    """
    from scipy.spatial.transform import Rotation as R

    pose = np.eye(4, dtype=np.float32)

    # Position
    pose[:3, 3] = [x, y, z]

    # Rotation (CARLA convention: z-up, left-handed)
    rotation = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)
    pose[:3, :3] = rotation.as_matrix().astype(np.float32)

    return pose


def validate_rendered_image(image: np.ndarray, expected_shape: tuple = None) -> bool:
    """
    Validate rendered image meets basic requirements.

    Args:
        image: Rendered image array
        expected_shape: Optional expected shape (H, W, 3)

    Returns:
        True if valid

    Raises:
        AssertionError if validation fails
    """
    # Check not None
    assert image is not None, "Image is None"

    # Check shape
    assert len(image.shape) == 3, f"Image shape must be 3D, got {image.shape}"
    assert image.shape[2] == 3, f"Image must have 3 channels (RGB), got {image.shape[2]}"

    if expected_shape:
        assert image.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {image.shape}"

    # Check dtype
    assert image.dtype == np.uint8, f"Image dtype must be uint8, got {image.dtype}"

    # Check content (not all black/white)
    mean_val = np.mean(image)
    assert mean_val > 0, f"Image appears all black (mean={mean_val})"
    assert mean_val < 255, f"Image appears all white (mean={mean_val})"

    return True


def save_image(image: np.ndarray, output_dir: Path, filename: str) -> Path:
    """
    Save image to output directory.

    Args:
        image: Image array to save
        output_dir: Output directory path
        filename: Output filename

    Returns:
        Path to saved image
    """
    try:
        import imageio
        output_path = output_dir / filename
        imageio.imwrite(output_path, image)
        logger.info(f"Saved image: {output_path}")
        return output_path
    except ImportError:
        from PIL import Image
        output_path = output_dir / filename
        Image.fromarray(image).save(output_path, quality=95)
        logger.info(f"Saved image: {output_path}")
        return output_path