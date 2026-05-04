# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Configuration Module

Provides configuration management for DGGT client integration.

Classes:
- DggtConfig: Dataclass holding configuration values
- DggtConfigLoader: Static methods for loading configuration from YAML and env vars
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DggtConfig:
    """DGGT client configuration.

    Holds all configuration parameters for connecting to the DGGT server
    and controlling rendering behavior.

    Attributes:
        server_host: gRPC server hostname
        server_port: gRPC server port
        scene_base_path: Base path to DGGT scene directories
        default_scene_id: Default scene identifier
        max_message_length: Maximum gRPC message size in bytes
        request_timeout: Request timeout in seconds
        image_format: Output image format (JPEG or PNG)
        image_quality: JPEG quality (0-100)
        coordinate_transform_enabled: Whether to apply coordinate transforms
        max_dynamic_objects: Maximum number of dynamic objects per frame
        t_scenario_dggt: Optional 4x4 transform matrix (scenario to DGGT coords)
    """

    # Server configuration
    server_host: str = "localhost"
    server_port: int = 50051

    # Scene configuration
    scene_base_path: str = ""
    default_scene_id: str = ""

    # gRPC configuration
    max_message_length: int = 10 * 1024 * 1024  # 10MB
    request_timeout: float = 30.0  # seconds

    # Image configuration
    image_format: str = "JPEG"  # JPEG or PNG
    image_quality: float = 95.0  # JPEG quality 0-100

    # Coordinate transform configuration
    coordinate_transform_enabled: bool = True

    # Dynamic objects configuration
    max_dynamic_objects: int = 100

    # Transform matrix (4x4 numpy array)
    t_scenario_dggt: Optional[np.ndarray] = field(default=None)

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if not self.scene_base_path:
            raise ValueError("scene_base_path is required")
        if self.image_quality < 0 or self.image_quality > 100:
            raise ValueError("image_quality must be in [0, 100]")
        if self.image_format not in ["JPEG", "PNG"]:
            raise ValueError(f"Unsupported image format: {self.image_format}")
        if self.server_port <= 0 or self.server_port > 65535:
            raise ValueError(f"Invalid server_port: {self.server_port}")
        if self.request_timeout <= 0:
            raise ValueError(f"Invalid request_timeout: {self.request_timeout}")
        if self.max_message_length <= 0:
            raise ValueError(f"Invalid max_message_length: {self.max_message_length}")
        if self.max_dynamic_objects < 0:
            raise ValueError(f"Invalid max_dynamic_objects: {self.max_dynamic_objects}")

        # Validate transform matrix if present
        if self.t_scenario_dggt is not None:
            if self.t_scenario_dggt.shape != (4, 4):
                raise ValueError(
                    f"t_scenario_dggt must be 4x4 matrix, got shape {self.t_scenario_dggt.shape}"
                )

    def __post_init__(self):
        """Post-initialization validation (optional, can be deferred)."""
        # Validation is deferred to explicit validate() call for flexibility
        # (e.g., when loading partial config before env overrides)
        pass


class DggtConfigLoader:
    """Configuration loader with support for YAML files and environment variables.

    Provides static methods to load configuration from different sources.
    Environment variables can override YAML configuration values.

    Environment Variable Overrides:
        DGGT_SERVER_HOST: Override server hostname
        DGGT_SERVER_PORT: Override server port
        DGGT_SCENE_BASE_PATH: Override scene base path
        DGGT_SCENE_ID: Override default scene ID
    """

    @staticmethod
    def from_yaml(path: str) -> DggtConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            DggtConfig instance with loaded values.

        Raises:
            FileNotFoundError: If config file does not exist.
            ValueError: If configuration validation fails.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        config = DggtConfig()

        # Server configuration
        if 'server' in data:
            config.server_host = data['server'].get('host', config.server_host)
            config.server_port = data['server'].get('port', config.server_port)
            config.request_timeout = data['server'].get('timeout', config.request_timeout)

        # Scene configuration
        if 'scene' in data:
            config.scene_base_path = data['scene'].get('base_path', '')
            # Always convert to string to handle YAML numeric parsing (001 -> 1)
            config.default_scene_id = str(data['scene'].get('default_id', ''))

        # Image configuration
        if 'image' in data:
            config.image_format = data['image'].get('format', config.image_format)
            config.image_quality = data['image'].get('quality', config.image_quality)

        # Coordinate transform configuration
        if 'coordinate' in data:
            config.coordinate_transform_enabled = data['coordinate'].get('enabled', True)

        # gRPC configuration (optional in YAML)
        if 'grpc' in data:
            config.max_message_length = data['grpc'].get('max_message_length', config.max_message_length)

        # Dynamic objects configuration (optional in YAML)
        if 'dynamic_objects' in data:
            config.max_dynamic_objects = data['dynamic_objects'].get('max_count', config.max_dynamic_objects)

        # Transform matrix (optional in YAML)
        if 'transform' in data and 't_scenario_dggt' in data['transform']:
            matrix_data = data['transform']['t_scenario_dggt']
            if isinstance(matrix_data, list):
                config.t_scenario_dggt = np.array(matrix_data, dtype=np.float32)

        config.validate()
        logger.info(f"Loaded config from {path}")
        return config

    @staticmethod
    def from_env(base_config: Optional[DggtConfig] = None) -> DggtConfig:
        """Load configuration from environment variables.

        Environment variables override values in the base_config.

        Args:
            base_config: Optional base configuration to override.
                         If None, creates a default config first.

        Returns:
            DggtConfig instance with environment overrides applied.

        Note:
            Does NOT validate the config automatically. Caller should
            call validate() if scene_base_path is expected to be set.
        """
        config = base_config or DggtConfig()

        # Environment variable overrides
        if 'DGGT_SERVER_HOST' in os.environ:
            config.server_host = os.environ['DGGT_SERVER_HOST']
        if 'DGGT_SERVER_PORT' in os.environ:
            config.server_port = int(os.environ['DGGT_SERVER_PORT'])
        if 'DGGT_SCENE_BASE_PATH' in os.environ:
            config.scene_base_path = os.environ['DGGT_SCENE_BASE_PATH']
        if 'DGGT_SCENE_ID' in os.environ:
            config.default_scene_id = os.environ['DGGT_SCENE_ID']

        # Additional optional env vars
        if 'DGGT_REQUEST_TIMEOUT' in os.environ:
            config.request_timeout = float(os.environ['DGGT_REQUEST_TIMEOUT'])
        if 'DGGT_IMAGE_FORMAT' in os.environ:
            config.image_format = os.environ['DGGT_IMAGE_FORMAT']
        if 'DGGT_IMAGE_QUALITY' in os.environ:
            config.image_quality = float(os.environ['DGGT_IMAGE_QUALITY'])
        if 'DGGT_MAX_DYNAMIC_OBJECTS' in os.environ:
            config.max_dynamic_objects = int(os.environ['DGGT_MAX_DYNAMIC_OBJECTS'])

        return config

    @staticmethod
    def default() -> DggtConfig:
        """Get default configuration.

        Returns:
            DggtConfig instance with all default values.

        Note:
            The default config has an empty scene_base_path, which
            will fail validation. Caller must set this before use.
        """
        return DggtConfig()

    @staticmethod
    def from_yaml_with_env(path: str) -> DggtConfig:
        """Load from YAML and apply environment variable overrides.

        This is the recommended method for production use.

        Args:
            path: Path to YAML configuration file.

        Returns:
            DggtConfig with YAML values overridden by env vars.

        Raises:
            FileNotFoundError: If config file does not exist.
            ValueError: If configuration validation fails.
        """
        config = DggtConfigLoader.from_yaml(path)
        config = DggtConfigLoader.from_env(config)
        config.validate()
        return config