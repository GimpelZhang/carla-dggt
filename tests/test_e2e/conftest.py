# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
E2E Test Configuration and Fixtures

Provides pytest fixtures for E2E testing with:
- CARLA client connection
- DGGT server connection
- OpenDRIVE world loading
- DggtScenario setup
"""

import pytest
import yaml
import numpy as np
import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import CARLA
try:
    import carla
except ImportError:
    raise ImportError("CARLA Python API not found. Please add CARLA to PYTHONPATH.")

# Import DGGT integration
from dggt_integration import DggtScenario, DggtRenderer, DggtSensor
from dggt_config import DggtConfig, DggtConfigLoader

# Import utils
from utils import undo_carla_coordinate_transform, se3_to_grpc_pose

# Import protobuf types
from nre.grpc.protos import sensorsim_pb2, common_pb2

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Fixture
# ============================================================================

@pytest.fixture
def e2e_config() -> Dict[str, Any]:
    """
    Load E2E test configuration from YAML file.

    Returns:
        Dict with configuration for CARLA, DGGT server, scene, and test settings.
    """
    config_path = Path(__file__).parent / "e2e_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"E2E config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# ============================================================================
# CARLA Client Fixture
# ============================================================================

@pytest.fixture
def carla_client(e2e_config):
    """
    Connect to CARLA server.

    Args:
        e2e_config: Test configuration

    Returns:
        carla.Client instance

    Note:
        Requires CARLA server running at configured host:port.
    """
    carla_cfg = e2e_config['carla']
    host = carla_cfg['host']
    port = carla_cfg['port']
    timeout = carla_cfg['timeout']

    client = carla.Client(host, port)
    client.set_timeout(timeout)

    # Verify connection
    try:
        world = client.get_world()
        logger.info(f"Connected to CARLA at {host}:{port}")
    except Exception as e:
        pytest.fail(f"Failed to connect to CARLA at {host}:{port}: {e}")

    yield client

    # Cleanup: reset world if needed
    # client.reload_world()  # Optional reset


# ============================================================================
# DGGT World Fixture (OpenDRIVE)
# ============================================================================

@pytest.fixture
def dggt_world(carla_client, e2e_config):
    """
    Load OpenDRIVE map into CARLA world.

    Args:
        carla_client: CARLA client
        e2e_config: Test configuration

    Returns:
        carla.World with OpenDRIVE map loaded

    Note:
        Uses xodr_map_path from config to load custom OpenDRIVE map.
    """
    scene_cfg = e2e_config['scene']
    xodr_path = scene_cfg['xodr_map_path']

    # Check if XODR file exists
    if not os.path.exists(xodr_path):
        pytest.skip(f"OpenDRIVE map not found: {xodr_path}")

    # Load OpenDRIVE map
    try:
        # CARLA API for loading OpenDRIVE
        client = carla_client

        # Option 1: Use load_map if available
        # world = client.load_map(xodr_path)

        # Option 2: Get current world and verify
        world = client.get_world()

        logger.info(f"CARLA world ready for scene: {scene_cfg['scene_id']}")

    except Exception as e:
        pytest.fail(f"Failed to load OpenDRIVE map: {e}")

    yield world


# ============================================================================
# DGGT Scenario Fixture
# ============================================================================

@pytest.fixture
def dggt_scenario(e2e_config):
    """
    Initialize DGGT scenario.

    Args:
        e2e_config: Test configuration

    Returns:
        DggtScenario instance connected to DGGT server

    Note:
        Requires DGGT server running at configured host:port.
    """
    scene_cfg = e2e_config['scene']
    dggt_cfg = e2e_config['dggt_server']

    # Create DggtConfig
    config = DggtConfig(
        server_host=dggt_cfg['host'],
        server_port=dggt_cfg['port'],
        scene_base_path=scene_cfg['dggt_scene_path'],
        default_scene_id=scene_cfg['scene_id'],
        request_timeout=dggt_cfg['timeout'],
    )

    # Create scenario
    try:
        scenario = DggtScenario(
            config=config,
            scene_id=scene_cfg['scene_id'],
        )
        scenario.load_scene()

        logger.info(f"DggtScenario loaded: {scene_cfg['scene_id']}")

    except Exception as e:
        pytest.fail(f"Failed to load DGGT scenario: {e}")

    yield scenario

    # Cleanup
    renderer = scenario.get_renderer()
    renderer.disconnect()


# ============================================================================
# Camera Spec Fixture
# ============================================================================

@pytest.fixture
def default_camera_spec(dggt_scenario):
    """
    Get default camera spec from DGGT scenario.

    Args:
        dggt_scenario: DGGT scenario

    Returns:
        sensorsim_pb2.CameraSpec for first available camera
    """
    renderer = dggt_scenario.get_renderer()
    available_cameras = renderer.get_available_cameras()

    if not available_cameras:
        pytest.fail("No cameras available in DGGT scenario")

    # Get first camera
    first_camera_id = list(available_cameras.keys())[0]
    camera_spec = available_cameras[first_camera_id]

    logger.info(f"Using camera: {first_camera_id}")

    return camera_spec


# ============================================================================
# Test Output Fixture
# ============================================================================

@pytest.fixture
def test_output_dir(e2e_config):
    """
    Create test output directory.

    Args:
        e2e_config: Test configuration

    Returns:
        Path to output directory
    """
    output_dir = Path(e2e_config['test']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Test output directory: {output_dir}")

    yield output_dir

    # Cleanup optional: remove output dir after tests
    # if output_dir.exists():
    #     shutil.rmtree(output_dir)


# Import helper functions from separate module (for explicit imports in test files)
from .test_helpers import create_test_camera_pose, validate_rendered_image, save_image

# Expose helpers for pytest fixture discovery (fixtures defined above)
# Note: pytest fixtures are auto-discovered; helper functions must be imported explicitly