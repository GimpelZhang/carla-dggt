# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Scene Metadata Classes

Core data structures for DGGT scene metadata.

IMPORTANT: Must import coordinate transform utilities from utils.py (lines 70-89, 98-116, 119-159)
and NEVER reimplement them locally.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np

# Import constants from existing constants.py
try:
    from ..constants import EGO_TRACK_ID, VEHICLE_LABELS
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from constants import EGO_TRACK_ID, VEHICLE_LABELS

logger = logging.getLogger(__name__)


@dataclass
class DGGTSceneMetadata:
    """DGGT scene metadata"""

    # Basic info (required, no defaults)
    scene_id: str                      # Scene unique identifier
    scene_path: str                    # Scene directory absolute path
    num_frames: int                    # Total frame count

    # Time info (required, no defaults)
    fps: float                         # Frame rate (default 10 Hz for Waymo)
    start_timestamp_us: int            # Start timestamp (microseconds)
    end_timestamp_us: int              # End timestamp (microseconds)

    # Camera info (required, no defaults)
    camera_width: int                  # Image width
    camera_height: int                 # Image height
    intrinsic_matrix: np.ndarray       # 3x3 intrinsic matrix (first frame as default)
    intrinsics_vary: bool              # Whether intrinsics vary across frames (variance > 1%)

    # Scene content (required, no defaults)
    has_static_scene: bool             # Has static scene
    has_sky_scene: bool                # Has sky scene
    dynamic_object_ids: List[int]      # Dynamic object ID list

    # File sizes (required, no defaults - moved before optional fields)
    static_scene_size_mb: float        # Static scene size
    total_size_mb: float               # Total size

    # Optional fields with defaults (must come after all required fields)
    georeference: Optional[Dict] = None  # OpenDRIVE georeference info (lat, lon, alt)
    t_scenario_dggt: Optional[np.ndarray] = None  # Scene to DGGT transform (4x4)

    def get_frame_index_from_timestamp(self, timestamp_us: int) -> int:
        """Calculate frame index from timestamp"""
        if timestamp_us < self.start_timestamp_us:
            return 0
        if timestamp_us >= self.end_timestamp_us:
            return self.num_frames - 1

        elapsed_us = timestamp_us - self.start_timestamp_us
        frame_idx = int(elapsed_us * self.fps / 1_000_000)
        return min(frame_idx, self.num_frames - 1)

    def get_timestamp_from_frame_index(self, frame_idx: int) -> int:
        """Calculate timestamp from frame index"""
        if frame_idx < 0:
            frame_idx = 0
        if frame_idx >= self.num_frames:
            frame_idx = self.num_frames - 1

        return self.start_timestamp_us + int(frame_idx * 1_000_000 / self.fps)


@dataclass
class FrameMetadata:
    """Single frame metadata"""

    frame_idx: int                     # Frame index
    timestamp_us: int                  # Timestamp

    # Camera data
    c2w_matrix: np.ndarray             # 4x4 Camera-to-World matrix
    intrinsic_matrix: np.ndarray       # 3x3 intrinsic matrix (actual for this frame)
    width: int
    height: int

    # Dynamic objects (optional)
    objects: List['ObjectMetadata'] = field(default_factory=list)


@dataclass
class ObjectMetadata:
    """Dynamic object metadata"""

    object_id: int                     # Object ID
    pose_world: np.ndarray             # 4x4 world pose matrix
    dimensions: np.ndarray             # [length, width, height]


@dataclass
class TrackIDMapping:
    """
    Bidirectional mapping between track_id (NuRec/CARLA string) and object_id (DGGT integer)

    Strategy:
    - Priority: explicit mapping (from config)
    - Fallback: auto-generate/extract when unmatched
    """

    auto_prefix: str = "dggt_obj"              # Prefix for auto-generated track_id
    explicit_mapping: Dict[str, int] = field(default_factory=dict)  # Explicit mapping

    def to_object_id(self, track_id: str) -> int:
        """
        NuRec track_id -> DGGT object_id

        Args:
            track_id: NuRec/CARLA string ID, e.g. "vehicle-001"

        Returns:
            DGGT integer object_id

        Raises:
            ValueError: Cannot parse track_id
        """
        # 1. Check explicit mapping first
        if track_id in self.explicit_mapping:
            return self.explicit_mapping[track_id]

        # 2. Try to extract number from string
        match = re.search(r'(\d+)', track_id)
        if match:
            return int(match.group(1))

        # 3. Cannot parse, raise error
        raise ValueError(
            f"Cannot convert track_id '{track_id}' to object_id: "
            f"no explicit mapping and no numeric component found"
        )

    def to_track_id(self, object_id: int) -> str:
        """
        DGGT object_id -> NuRec track_id

        Args:
            object_id: DGGT integer ID, e.g. 0, 1, 2

        Returns:
            NuRec/CARLA string track_id
        """
        # 1. Check reverse mapping first
        for tid, oid in self.explicit_mapping.items():
            if oid == object_id:
                return tid

        # 2. Auto-generate
        return f"{self.auto_prefix}_{object_id:04d}"

    def register(self, track_id: str, object_id: int) -> None:
        """Register explicit mapping"""
        self.explicit_mapping[track_id] = object_id

    def from_scene_objects(self, object_ids: List[int]) -> None:
        """
        Auto-build mapping from scene object_id list

        Args:
            object_ids: All object_id list in scene
        """
        for oid in object_ids:
            track_id = f"{self.auto_prefix}_{oid:04d}"
            self.explicit_mapping[track_id] = oid