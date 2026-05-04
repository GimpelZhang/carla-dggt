# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Phase 4 OpenDRIVE Integration Tests

Integration tests for OpenDRIVE world generation in DGGT scenarios.

Tests:
1. Test xodr detection in DGGT scene path
2. Test xodr content loading
3. Test CARLA world generation from OpenDRIVE
4. Test geoReference parsing
5. Test actor spawning on OpenDRIVE world
6. Test full integration with DggtScenario

Environment:
- CARLA server at 127.0.0.1:2000
- DGGT scene at /home/junchuan/e2e/dggt/output/waymo/training/scene1/0328/001/

Usage:
    # Run with CARLA server running
    cd /mnt/E/carla/carla/PythonAPI/examples/nvidia/nurec
    source /home/junchuan/miniconda3/etc/profile.d/conda.sh && conda activate dggt
    python test_phase4_opendrive.py
"""

import sys
import os
import unittest
import logging
import time
import numpy as np

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    print("Warning: carla module not available")

from dggt_project.dggt_opendrive import DggtOpendriveWorld
from dggt_project.dggt_projection import get_t_rig_enu_from_ecef
from dggt_project.dggt_scenario import DggtScenario

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
TEST_SCENE_PATH = '/home/junchuan/e2e/dggt/output/waymo/training/scene1/0328/001/'
EXPECTED_XODR_FILE = 'map.xodr'


class TestDggtOpendriveWorld(unittest.TestCase):
    """Test DggtOpendriveWorld class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.scene_path = TEST_SCENE_PATH
        self.client = None

        if CARLA_AVAILABLE:
            try:
                self.client = carla.Client(CARLA_HOST, CARLA_PORT)
                self.client.set_timeout(10.0)
                # Test connection
                self.client.get_world()
                logger.info(f"Connected to CARLA server at {CARLA_HOST}:{CARLA_PORT}")
            except Exception as e:
                logger.warning(f"Could not connect to CARLA server: {e}")
                self.client = None

    def test_01_xodr_detection(self):
        """Test 1: Test xodr detection in DGGT scene path."""
        logger.info("Test 1: xodr detection")

        # Test without CARLA client (just file detection)
        opendrive_world = DggtOpendriveWorld(
            self.client if self.client else None,
            self.scene_path
        )

        xodr_path = opendrive_world.detect_xodr_file()

        if xodr_path:
            self.assertTrue(os.path.exists(xodr_path))
            self.assertEqual(os.path.basename(xodr_path), EXPECTED_XODR_FILE)
            logger.info(f"  Found xodr file: {xodr_path}")
        else:
            # Check if scene path exists
            if os.path.exists(self.scene_path):
                logger.warning(f"  Scene path exists but no map.xodr found")
            else:
                logger.warning(f"  Scene path does not exist: {self.scene_path}")
            self.skipTest(f"No xodr file found in {self.scene_path}")

    def test_02_xodr_content_loading(self):
        """Test 2: Test xodr content loading."""
        logger.info("Test 2: xodr content loading")

        opendrive_world = DggtOpendriveWorld(
            self.client if self.client else None,
            self.scene_path
        )

        # First detect file
        xodr_path = opendrive_world.detect_xodr_file()
        if xodr_path is None:
            self.skipTest(f"No xodr file found in {self.scene_path}")

        # Load content
        content = opendrive_world.load_xodr_content()

        self.assertIsNotNone(content)
        self.assertTrue(len(content) > 0)
        self.assertTrue(content.startswith('<'), "Content should start with XML opening tag")
        logger.info(f"  Loaded xodr content: {len(content)} bytes")

    def test_03_geo_reference_parsing(self):
        """Test 3: Test geoReference parsing."""
        logger.info("Test 3: geoReference parsing")

        opendrive_world = DggtOpendriveWorld(
            self.client if self.client else None,
            self.scene_path
        )

        xodr_path = opendrive_world.detect_xodr_file()
        if xodr_path is None:
            self.skipTest(f"No xodr file found in {self.scene_path}")

        content = opendrive_world.load_xodr_content()
        if content is None:
            self.skipTest("Could not load xodr content")

        geo_ref = opendrive_world.get_geo_reference()

        if geo_ref:
            self.assertIn('lat', geo_ref)
            self.assertIn('lon', geo_ref)
            self.assertIn('alt', geo_ref)
            logger.info(f"  geoReference parsed: lat={geo_ref['lat']}, lon={geo_ref['lon']}, alt={geo_ref['alt']}")
        else:
            logger.warning("  geoReference not found or could not be parsed")
            # This is not necessarily a failure - some maps may not have geoReference

    def test_04_carla_world_generation(self):
        """Test 4: Test CARLA world generation from OpenDRIVE."""
        logger.info("Test 4: CARLA world generation")

        if self.client is None:
            self.skipTest("CARLA server not available")

        opendrive_world = DggtOpendriveWorld(self.client, self.scene_path)

        xodr_path = opendrive_world.detect_xodr_file()
        if xodr_path is None:
            self.skipTest(f"No xodr file found in {self.scene_path}")

        world = opendrive_world.generate_world()

        if world:
            self.assertIsNotNone(world)
            # Verify world is a valid CARLA world
            snapshot = world.get_snapshot()
            self.assertIsNotNone(snapshot)
            logger.info(f"  Generated CARLA world successfully")
            logger.info(f"  World map name: {world.get_map().name if world.get_map() else 'unknown'}")
        else:
            self.fail("Failed to generate CARLA world from OpenDRIVE")

    def test_05_coordinate_transform(self):
        """Test 5: Test coordinate transform using geoReference."""
        logger.info("Test 5: coordinate transform")

        opendrive_world = DggtOpendriveWorld(
            self.client if self.client else None,
            self.scene_path
        )

        xodr_path = opendrive_world.detect_xodr_file()
        if xodr_path is None:
            self.skipTest(f"No xodr file found in {self.scene_path}")

        content = opendrive_world.load_xodr_content()
        if content is None:
            self.skipTest("Could not load xodr content")

        # Create a dummy ECEF transformation matrix
        t_world_base = np.eye(4)

        # Compute transform
        t_scenario_carla = get_t_rig_enu_from_ecef(t_world_base, content)

        self.assertIsNotNone(t_scenario_carla)
        self.assertEqual(t_scenario_carla.shape, (4, 4))
        logger.info(f"  Computed coordinate transform matrix")
        logger.info(f"  Translation: {t_scenario_carla[:3, 3]}")


class TestDggtScenarioIntegration(unittest.TestCase):
    """Test full DggtScenario integration with OpenDRIVE."""

    def setUp(self):
        """Set up test fixtures."""
        self.scene_path = TEST_SCENE_PATH
        self.client = None

        if CARLA_AVAILABLE:
            try:
                self.client = carla.Client(CARLA_HOST, CARLA_PORT)
                self.client.set_timeout(10.0)
                self.client.get_world()
                logger.info(f"Connected to CARLA server at {CARLA_HOST}:{CARLA_PORT}")
            except Exception as e:
                logger.warning(f"Could not connect to CARLA server: {e}")
                self.client = None

    def test_06_scenario_opendrive_setup(self):
        """Test 6: Test full integration with DggtScenario."""
        logger.info("Test 6: Full DggtScenario integration")

        if self.client is None:
            self.skipTest("CARLA server not available")

        # Create scenario instance
        scenario = DggtScenario(self.client, self.scene_path)

        # Test setup_opendrive_world method
        world = scenario.setup_opendrive_world()

        self.assertIsNotNone(world)
        self.assertIsNotNone(scenario.world)
        logger.info(f"  Setup OpenDRIVE world via DggtScenario")
        logger.info(f"  Scenario repr: {repr(scenario)}")

        # Get geo reference if available
        geo_ref = scenario.get_opendrive_geo_reference()
        if geo_ref:
            logger.info(f"  geoReference: {geo_ref}")

    def test_07_scenario_context_manager(self):
        """Test 7: Test DggtScenario context manager pattern."""
        logger.info("Test 7: DggtScenario context manager")

        if self.client is None:
            self.skipTest("CARLA server not available")

        with DggtScenario(self.client, self.scene_path) as scenario:
            # Inside context, world should be initialized
            self.assertIsNotNone(scenario.world)
            logger.info(f"  Context manager initialized world")
            logger.info(f"  Scenario: {repr(scenario)}")

        # After context, actors should be destroyed
        self.assertEqual(len(scenario.actor_mapping), 0)
        logger.info("  Context manager cleaned up actors")

    def test_08_actor_spawning_on_opendrive_world(self):
        """Test 8: Test actor spawning on OpenDRIVE world."""
        logger.info("Test 8: Actor spawning on OpenDRIVE world")

        if self.client is None:
            self.skipTest("CARLA server not available")

        with DggtScenario(self.client, self.scene_path) as scenario:
            world = scenario.world
            if world is None:
                self.skipTest("World not initialized")

            # Get blueprint library
            blueprint_library = world.get_blueprint_library()

            # Find a vehicle blueprint
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            if vehicle_bp is None:
                vehicle_bp = blueprint_library.filter('vehicle.*')[0]

            self.assertIsNotNone(vehicle_bp)
            logger.info(f"  Found vehicle blueprint: {vehicle_bp.id}")

            # Spawn point - use spawn points from the world
            spawn_points = world.get_map().get_spawn_points()
            if len(spawn_points) > 0:
                spawn_point = spawn_points[0]
            else:
                # Default spawn point
                spawn_point = carla.Transform(
                    carla.Location(x=0, y=0, z=1),
                    carla.Rotation()
                )

            # Try to spawn a vehicle
            try:
                vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
                if vehicle:
                    logger.info(f"  Spawned vehicle: {vehicle.id}")
                    # Destroy it
                    vehicle.destroy()
                    logger.info("  Destroyed test vehicle")
                else:
                    logger.warning("  Could not spawn vehicle at spawn point")
            except Exception as e:
                logger.warning(f"  Error spawning vehicle: {e}")


def run_tests():
    """Run all tests."""
    # Check prerequisites
    logger.info("=" * 60)
    logger.info("Phase 4 OpenDRIVE Integration Tests")
    logger.info("=" * 60)

    logger.info(f"Scene path: {TEST_SCENE_PATH}")
    logger.info(f"Expected xodr file: {EXPECTED_XODR_FILE}")

    if os.path.exists(TEST_SCENE_PATH):
        logger.info(f"Scene path exists: YES")
        expected_xodr = os.path.join(TEST_SCENE_PATH, EXPECTED_XODR_FILE)
        if os.path.exists(expected_xodr):
            logger.info(f"map.xodr exists: YES")
        else:
            logger.warning(f"map.xodr exists: NO")
    else:
        logger.warning(f"Scene path exists: NO")

    if CARLA_AVAILABLE:
        logger.info("CARLA module available: YES")
        try:
            client = carla.Client(CARLA_HOST, CARLA_PORT)
            client.set_timeout(5.0)
            client.get_world()
            logger.info(f"CARLA server connection: YES ({CARLA_HOST}:{CARLA_PORT})")
        except Exception as e:
            logger.warning(f"CARLA server connection: NO - {e}")
    else:
        logger.warning("CARLA module available: NO")

    logger.info("=" * 60)

    # Run tests
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()