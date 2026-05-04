# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Server Configuration

Configuration dataclass for DGGT server settings.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DGGTServerConfig:
    """DGGT Server Configuration"""

    # gRPC service
    grpc_host: str = "localhost"
    grpc_port: int = 50051
    max_workers: int = 10  # ThreadPoolExecutor max workers

    # Device configuration
    device: str = "cuda"  # Device for rendering (cuda/cpu)

    # Scene configuration
    scene_base_path: str = "/home/junchuan/e2e/dggt/output/waymo/training"
    default_fps: float = 10.0
    auto_discover: bool = True

    # Scene ID mapping (optional)
    scene_id_mappings: Dict[str, str] = field(default_factory=dict)  # external_id -> internal_id

    # Render configuration
    default_image_format: str = "JPEG"
    default_image_quality: float = 95.0

    # Coordinate system configuration
    # When True, incoming poses are assumed to be in DGGT coordinates (skip CARLA transform)
    # This should be True when using dggt_client.py with DGGT scene data
    # When False, incoming poses are assumed to be in CARLA coordinates (apply transform)
    source_is_dggt: bool = True

    # Cache configuration
    max_cached_scenes: int = 10

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DGGTServerConfig":
        """
        Load configuration from YAML file

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            DGGTServerConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Parse nested structure
        grpc = config_dict.get('grpc', {})
        scene = config_dict.get('scene', {})
        render = config_dict.get('render', {})
        cache = config_dict.get('cache', {})

        return cls(
            grpc_host=grpc.get('host', 'localhost'),
            grpc_port=grpc.get('port', 50051),
            max_workers=grpc.get('max_workers', 10),
            device=config_dict.get('device', 'cuda'),
            scene_base_path=scene.get('base_path', '/home/junchuan/e2e/dggt/output/waymo/training'),
            default_fps=scene.get('default_fps', 10.0),
            auto_discover=scene.get('auto_discover', True),
            scene_id_mappings=scene.get('id_mappings', {}),
            default_image_format=render.get('default_format', 'JPEG'),
            default_image_quality=render.get('default_quality', 95.0),
            source_is_dggt=config_dict.get('source_is_dggt', True),
            max_cached_scenes=cache.get('max_cached_scenes', 10),
        )

    @classmethod
    def from_env(cls) -> "DGGTServerConfig":
        """
        Load configuration from environment variables

        Returns:
            DGGTServerConfig instance
        """
        return cls(
            grpc_host=os.getenv("DGGT_GRPC_HOST", "localhost"),
            grpc_port=int(os.getenv("DGGT_GRPC_PORT", "50051")),
            max_workers=int(os.getenv("DGGT_MAX_WORKERS", "10")),
            device=os.getenv("DGGT_DEVICE", "cuda"),
            scene_base_path=os.getenv("DGGT_SCENE_PATH", "/home/junchuan/e2e/dggt/output/waymo/training"),
            default_fps=float(os.getenv("DGGT_DEFAULT_FPS", "10.0")),
            auto_discover=os.getenv("DGGT_AUTO_DISCOVER", "true").lower() == "true",
            default_image_format=os.getenv("DGGT_IMAGE_FORMAT", "JPEG"),
            default_image_quality=float(os.getenv("DGGT_IMAGE_QUALITY", "95.0")),
            source_is_dggt=os.getenv("DGGT_SOURCE_IS_DGGT", "true").lower() == "true",
            max_cached_scenes=int(os.getenv("DGGT_MAX_CACHED_SCENES", "10")),
        )
