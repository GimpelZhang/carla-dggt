# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Server Package

This package provides scene loading and management for DGGT (Dynamic Gaussian Ground Truth) data.

Note: scene_loader and scene_manager require scipy for coordinate transforms.
These imports are optional and will fail gracefully if scipy is not available.
"""

from .exceptions import (
    DGGTError,
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
    TrackIDMapping,
)

# Optional imports (require scipy)
try:
    from .scene_loader import DGGTSceneLoader
    from .scene_manager import DGGTSceneManager
    from .config import DGGTServerConfig
    from .frame_index_mapper import FrameIndexMapper
    from .request_converter import RequestConverter
    _SCIPY_AVAILABLE = True
except ImportError as e:
    # scipy not available - scene_loader/scene_manager require coordinate transforms
    DGGTSceneLoader = None
    DGGTSceneManager = None
    DGGTServerConfig = None
    FrameIndexMapper = None
    RequestConverter = None
    _SCIPY_AVAILABLE = False
    _IMPORT_ERROR = str(e)

# Optional imports (require grpc)
try:
    from .dggt_service import DGGTService
    _GRPC_AVAILABLE = True
except ImportError as e:
    DGGTService = None
    _GRPC_AVAILABLE = False
    _GRPC_IMPORT_ERROR = str(e)

# Optional imports (require torch/gsplat for rendering)
try:
    from .dggt_renderer import DGGTRenderer, render_dggt_frame, _get_undo_carla_coordinate_transform
    _TORCH_AVAILABLE = True
except ImportError as e:
    DGGTRenderer = None
    render_dggt_frame = None
    _get_undo_carla_coordinate_transform = None
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = str(e)

__all__ = [
    "DGGTError",
    "SceneNotFoundError",
    "MissingStaticSceneError",
    "EmptyEgoPoseError",
    "InvalidJSONError",
    "FrameIndexOutOfRangeError",
    "DGGTSceneMetadata",
    "FrameMetadata",
    "ObjectMetadata",
    "TrackIDMapping",
    "DGGTSceneLoader",
    "DGGTSceneManager",
    "DGGTServerConfig",
    "FrameIndexMapper",
    "RequestConverter",
    "DGGTService",
    "DGGTRenderer",
    "render_dggt_frame",
    "_get_undo_carla_coordinate_transform",
]