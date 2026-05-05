# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT OpenDRIVE World Generation Module

Provides functionality to detect and load OpenDRIVE map files from DGGT scene
directories and generate CARLA worlds from them.

Classes:
    DggtOpendriveWorld: Manages OpenDRIVE world generation from scene directory
"""

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional

import carla

logger = logging.getLogger(__name__)


class DggtOpendriveWorld:
    """
    Manages OpenDRIVE world generation for DGGT scenes.

    This class detects and loads OpenDRIVE map files (map.xodr) from DGGT
    scene directories and generates CARLA worlds using the OpenDRIVE data.

    Attributes:
        client: CARLA client instance
        scene_path: Path to the DGGT scene directory
        xodr_path: Path to the detected map.xodr file
        xodr_content: XML content of the OpenDRIVE file
        world: Generated CARLA world (after generate_world() is called)

    Example:
        >>> client = carla.Client('localhost', 2000)
        >>> opendrive_world = DggtOpendriveWorld(client, '/path/to/scene')
        >>> if opendrive_world.detect_xodr_file():
        ...     world = opendrive_world.generate_world()
        ...     geo_ref = opendrive_world.get_geo_reference()
    """

    # Default OpenDRIVE generation parameters
    DEFAULT_GENERATION_PARAMS = carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        max_road_length=500.0,
        wall_height=0.0,
        additional_width=7.6,
        smooth_junctions=True,
        enable_mesh_visibility=True,
    )

    def __init__(self, client: carla.Client, scene_path: str):
        """
        Initialize the DGGT OpenDRIVE world generator.

        Args:
            client: CARLA client instance connected to the server
            scene_path: Path to the DGGT scene directory containing map.xodr
        """
        self.client = client
        self.scene_path = Path(scene_path)
        self.xodr_path: Optional[str] = None
        self.xodr_content: Optional[str] = None
        self._world: Optional[carla.World] = None
        self._geo_reference: Optional[Dict] = None

    def detect_xodr_file(self) -> Optional[str]:
        """
        Detect map.xodr file in the scene directory.

        Searches for 'map.xodr' directly in the scene directory.
        DGGT scenes store OpenDRIVE files directly (unlike NUREC which
        stores them in zip archives).

        Returns:
            str: Path to the map.xodr file if found, None otherwise
        """
        expected_path = self.scene_path / "map.xodr"

        if expected_path.exists() and expected_path.is_file():
            self.xodr_path = str(expected_path)
            logger.debug(f"Found map.xodr at: {self.xodr_path}")
            return self.xodr_path

        logger.warning(f"map.xodr not found in scene directory: {self.scene_path}")
        return None

    def load_xodr_content(self) -> Optional[str]:
        """
        Load XML content from the detected xodr file.

        Reads the OpenDRIVE XML file content into memory. Must be called
        after detect_xodr_file() has successfully found the file.

        Returns:
            str: XML content of the OpenDRIVE file if loaded successfully,
                 None if file not found or cannot be read

        Raises:
            FileNotFoundError: If xodr_path is not set or file doesn't exist
            IOError: If file cannot be read
        """
        if self.xodr_path is None:
            # Try to detect first if not already done
            if self.detect_xodr_file() is None:
                return None

        try:
            with open(self.xodr_path, 'r', encoding='utf-8') as f:
                self.xodr_content = f.read()

            logger.debug(
                f"Loaded OpenDRIVE content from {self.xodr_path} "
                f"({len(self.xodr_content)} bytes)"
            )
            return self.xodr_content

        except FileNotFoundError:
            logger.error(f"OpenDRIVE file not found: {self.xodr_path}")
            return None
        except IOError as e:
            logger.error(f"Error reading OpenDRIVE file {self.xodr_path}: {e}")
            return None

    def generate_world(
        self,
        params: Optional[carla.OpendriveGenerationParameters] = None
    ) -> Optional[carla.World]:
        """
        Generate CARLA world from OpenDRIVE data.

        Uses client.generate_opendrive_world() to create a CARLA world
        from the loaded OpenDRIVE XML content. The world is generated
        with parameters optimized for DGGT scenes.

        Args:
            params: Optional OpenDRIVE generation parameters. If not provided,
                    uses DEFAULT_GENERATION_PARAMS optimized for DGGT scenes.

        Returns:
            carla.World: Generated CARLA world if successful, None otherwise

        Raises:
            RuntimeError: If xodr_content is not loaded
        """
        # Load content if not already done
        if self.xodr_content is None:
            if self.load_xodr_content() is None:
                return None

        if params is None:
            params = self.DEFAULT_GENERATION_PARAMS

        try:
            self._world = self.client.generate_opendrive_world(
                self.xodr_content,
                params,
            )
            logger.info(
                f"Generated CARLA world from OpenDRIVE: {self.scene_path.name}"
            )
            return self._world

        except Exception as e:
            logger.error(f"Failed to generate world from OpenDRIVE: {e}")
            return None

    def get_geo_reference(self) -> Optional[Dict]:
        """
        Parse geoReference from OpenDRIVE header.

        Extracts geographic reference information (lat_0, lon_0, alt_0) from
        the OpenDRIVE XML header's geoReference element. This information
        is needed for coordinate transformations between ECEF and ENU frames.

        Returns:
            Dict: Dictionary with 'lat', 'lon', 'alt' keys if parsed successfully,
                  None if geoReference not found or cannot be parsed

        Note:
            The geoReference element contains a PROJ string like:
            "+proj=tmerc +lat_0=37.4 +lon_0=-122.1 +alt_0=0 ..."
        """
        # Load content if not already done
        if self.xodr_content is None:
            if self.load_xodr_content() is None:
                return None

        if self._geo_reference is not None:
            return self._geo_reference

        try:
            tree = ET.fromstring(self.xodr_content)
            geo_references = tree.findall(".//geoReference")

            if len(geo_references) < 1:
                logger.warning("No geoReference element found in OpenDRIVE")
                return None

            geo_reference = geo_references[0]

            if geo_reference.text is None:
                logger.warning("geoReference element has no text content")
                return None

            # Parse PROJ string parameters
            proj_parts = geo_reference.text.split(" ")

            lats = [
                float(lat[7:]) for lat in proj_parts
                if lat.find("+lat_0") != -1
            ]
            lons = [
                float(lon[7:]) for lon in proj_parts
                if lon.find("+lon_0") != -1
            ]
            alts = [
                float(alt[8:]) for alt in proj_parts
                if alt.find("+alt_0") != -1 or alt.find("+=alt_0") != -1
            ]

            if len(lats) != 1 or len(lons) != 1 or len(alts) != 1:
                logger.warning(
                    f"Unable to parse geoReference: {geo_reference.text}"
                )
                return None

            self._geo_reference = {
                'lat': lats[0],
                'lon': lons[0],
                'alt': alts[0],
            }

            logger.debug(
                f"Parsed geoReference: lat={self._geo_reference['lat']}, "
                f"lon={self._geo_reference['lon']}, "
                f"alt={self._geo_reference['alt']}"
            )
            return self._geo_reference

        except ET.ParseError as e:
            logger.error(f"Failed to parse OpenDRIVE XML: {e}")
            return None

    @property
    def world(self) -> Optional[carla.World]:
        """Get the generated CARLA world."""
        return self._world

    @property
    def geo_reference(self) -> Optional[Dict]:
        """Get the parsed geoReference dictionary."""
        return self._geo_reference

    def __repr__(self) -> str:
        """Return string representation of the instance."""
        status = []
        if self.xodr_path:
            status.append(f"xodr={self.xodr_path}")
        else:
            status.append("xodr=not_detected")
        if self._world:
            status.append("world=generated")
        else:
            status.append("world=not_generated")

        return f"DggtOpendriveWorld(scene={self.scene_path.name}, {', '.join(status)})"