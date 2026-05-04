# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unit tests for Track ID Mapping

Tests the TrackIDMapping class from scene_metadata.py:
- Auto-generate track_id with prefix
- Extract object_id from numeric string
- Explicit mapping
- from_scene_objects() method
- Unparseable track_id handling
- Priority: explicit over auto

Reference: dggt_server/scene_metadata.py TrackIDMapping class
"""

import pytest
import numpy as np

# Import the module under test
try:
    from dggt_server.scene_metadata import TrackIDMapping
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from dggt_server.scene_metadata import TrackIDMapping


@pytest.mark.unit
class TestTrackIDMapping:
    """Tests for TrackIDMapping class"""

    def test_auto_generate_track_id_with_prefix(self):
        """Test auto-generation of track_id with prefix"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")

        # Generate track_id for object_id 0
        track_id = mapping.to_track_id(0)
        assert track_id == "dggt_obj_0000"

        # Generate track_id for object_id 123
        track_id = mapping.to_track_id(123)
        assert track_id == "dggt_obj_0123"

    def test_auto_generate_custom_prefix(self):
        """Test auto-generation with custom prefix"""
        mapping = TrackIDMapping(auto_prefix="vehicle")

        track_id = mapping.to_track_id(5)
        assert track_id == "vehicle_0005"

    def test_extract_object_id_from_numeric_string(self):
        """Test extraction of object_id from track_id containing number"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")
        # No explicit mapping registered

        # Track_id "vehicle_123" should extract 123
        object_id = mapping.to_object_id("vehicle_123")
        assert object_id == 123

        # Track_id "car_999" should extract 999
        object_id = mapping.to_object_id("car_999")
        assert object_id == 999

    def test_extract_object_id_from_pure_numeric(self):
        """Test extraction from track_id that is pure number"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")

        object_id = mapping.to_object_id("42")
        assert object_id == 42

    def test_explicit_mapping(self):
        """Test explicit mapping registration"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")

        # Register explicit mapping
        mapping.register("ego_vehicle", 0)
        mapping.register("npc_1", 100)

        # Lookup should return explicit values
        assert mapping.to_object_id("ego_vehicle") == 0
        assert mapping.to_object_id("npc_1") == 100

    def test_explicit_mapping_to_track_id(self):
        """Test reverse lookup for explicit mapping"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")
        mapping.register("ego_vehicle", 0)
        mapping.register("npc_1", 100)

        # Reverse lookup should return original track_id
        assert mapping.to_track_id(0) == "ego_vehicle"
        assert mapping.to_track_id(100) == "npc_1"

    def test_from_scene_objects_method(self):
        """Test from_scene_objects() auto-build method"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")

        # Auto-build mapping from object_id list
        object_ids = [0, 1, 2, 3, 10, 20]
        mapping.from_scene_objects(object_ids)

        # Verify mappings were created
        assert mapping.to_object_id("dggt_obj_0000") == 0
        assert mapping.to_object_id("dggt_obj_0001") == 1
        assert mapping.to_object_id("dggt_obj_0002") == 2
        assert mapping.to_object_id("dggt_obj_0003") == 3
        assert mapping.to_object_id("dggt_obj_0010") == 10
        assert mapping.to_object_id("dggt_obj_0020") == 20

        # Verify reverse mapping works
        assert mapping.to_track_id(0) == "dggt_obj_0000"
        assert mapping.to_track_id(20) == "dggt_obj_0020"

    def test_unparseable_track_id_raises_error(self):
        """Test that unparseable track_id raises ValueError"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")
        # No explicit mapping registered

        # Track_id without numeric component should raise error
        with pytest.raises(ValueError) as exc_info:
            mapping.to_object_id("unknown_vehicle")

        assert "Cannot convert track_id" in str(exc_info.value)
        assert "unknown_vehicle" in str(exc_info.value)
        assert "no numeric component" in str(exc_info.value)

    def test_unparseable_track_id_no_digits(self):
        """Test track_id with no digits raises error"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")

        with pytest.raises(ValueError):
            mapping.to_object_id("abc_xyz")

    def test_priority_explicit_over_auto(self):
        """Test that explicit mapping takes priority over auto extraction"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")

        # Register explicit mapping that differs from what would be extracted
        # "vehicle_999" would normally extract 999, but we map it to 42
        mapping.register("vehicle_999", 42)

        # Explicit mapping should win
        object_id = mapping.to_object_id("vehicle_999")
        assert object_id == 42  # Explicit value, not 999

    def test_priority_explicit_over_numeric_extraction(self):
        """Test explicit mapping priority over numeric extraction"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")

        # Register: "car_123" -> 500 (explicit)
        # Without explicit, "car_123" would extract 123
        mapping.register("car_123", 500)

        object_id = mapping.to_object_id("car_123")
        assert object_id == 500  # Explicit, not 123

    def test_to_track_id_fallback_to_auto(self):
        """Test to_track_id falls back to auto-generation for unknown object_id"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")
        # No explicit mappings

        # Unknown object_id should auto-generate
        track_id = mapping.to_track_id(99)
        assert track_id == "dggt_obj_0099"

    def test_to_track_id_format_padding(self):
        """Test that auto-generated track_id has proper zero padding"""
        mapping = TrackIDMapping(auto_prefix="obj")

        # Test various object_ids for proper formatting
        assert mapping.to_track_id(0) == "obj_0000"
        assert mapping.to_track_id(1) == "obj_0001"
        assert mapping.to_track_id(10) == "obj_0010"
        assert mapping.to_track_id(100) == "obj_0100"
        assert mapping.to_track_id(1000) == "obj_1000"
        assert mapping.to_track_id(9999) == "obj_9999"


@pytest.mark.unit
class TestTrackIDMappingEdgeCases:
    """Edge case tests for TrackIDMapping"""

    def test_empty_explicit_mapping(self):
        """Test behavior with empty explicit mapping"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")

        # Should extract from numeric string
        assert mapping.to_object_id("123") == 123

        # Should raise for unparseable
        with pytest.raises(ValueError):
            mapping.to_object_id("abc")

    def test_multiple_digits_in_track_id(self):
        """Test extraction when track_id has multiple numbers"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")

        # Should extract first number sequence
        # "vehicle_123_456" should extract 123
        object_id = mapping.to_object_id("vehicle_123_456")
        assert object_id == 123

    def test_track_id_with_leading_digits(self):
        """Test track_id starting with digits"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")

        object_id = mapping.to_object_id("123_vehicle")
        assert object_id == 123

    def test_register_updates_mapping(self):
        """Test that register updates existing mapping"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")

        # First registration
        mapping.register("test", 10)
        assert mapping.to_object_id("test") == 10

        # Update registration
        mapping.register("test", 20)
        assert mapping.to_object_id("test") == 20

    def test_from_scene_objects_empty_list(self):
        """Test from_scene_objects with empty list"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")

        # Empty list should not raise
        mapping.from_scene_objects([])

        # No mappings created, so auto-generated track_id won't have explicit mapping
        # But numeric extraction still works for track_ids like "dggt_obj_0000"
        # The regex will extract "0000" = 0 from "dggt_obj_0000"
        assert mapping.to_object_id("dggt_obj_0000") == 0  # Extracts 0

    def test_from_scene_objects_overwrites_existing(self):
        """Test that from_scene_objects adds to existing mappings"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")

        # Register some explicit mappings first
        mapping.register("special_ego", 0)

        # Auto-build from scene objects
        mapping.from_scene_objects([0, 1, 2])

        # Both explicit and auto mappings should work
        assert mapping.to_object_id("special_ego") == 0  # Explicit still works
        assert mapping.to_object_id("dggt_obj_0001") == 1  # Auto works

        # Auto-generated track_id for 0 should NOT override explicit
        # (they both map to 0, but reverse lookup prefers explicit)
        assert mapping.to_track_id(0) == "special_ego"  # Explicit wins in reverse

    def test_zero_object_id(self):
        """Test handling of object_id = 0"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")

        # Auto-generate track_id for 0
        assert mapping.to_track_id(0) == "dggt_obj_0000"

        # Extract 0 from track_id
        assert mapping.to_object_id("0") == 0
        assert mapping.to_object_id("vehicle_0") == 0

    def test_large_object_id(self):
        """Test handling of large object_id values"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")

        # Large object_id
        track_id = mapping.to_track_id(99999)
        assert track_id == "dggt_obj_99999"

        # Extract large number
        object_id = mapping.to_object_id("vehicle_99999")
        assert object_id == 99999

    def test_negative_extraction_not_allowed(self):
        """Test that negative numbers cannot be extracted"""
        mapping = TrackIDMapping(auto_prefix="dggt_obj")

        # Track_id with negative sign should still extract positive
        # "vehicle_-123" would extract 123 (the regex \d+ only matches positive digits)
        # Actually, re.search(r'(\d+)', "vehicle_-123") would find "123"
        object_id = mapping.to_object_id("vehicle_-123")
        assert object_id == 123  # Extracts the positive digits only


@pytest.mark.unit
class TestTrackIDMappingIntegration:
    """Integration-style tests for TrackIDMapping"""

    def test_full_workflow(self):
        """Test complete workflow: create, register, lookup"""
        mapping = TrackIDMapping(auto_prefix="carla_obj")

        # Step 1: Register explicit mappings for known objects
        mapping.register("ego", 0)
        mapping.register("npc_vehicle_001", 100)
        mapping.register("npc_walker_001", 200)

        # Step 2: Verify explicit lookups work
        assert mapping.to_object_id("ego") == 0
        assert mapping.to_object_id("npc_vehicle_001") == 100
        assert mapping.to_object_id("npc_walker_001") == 200

        # Step 3: Verify reverse lookups work
        assert mapping.to_track_id(0) == "ego"
        assert mapping.to_track_id(100) == "npc_vehicle_001"
        assert mapping.to_track_id(200) == "npc_walker_001"

        # Step 4: Numeric extraction for unknown track_id
        assert mapping.to_object_id("unknown_500") == 500

        # Step 5: Auto-generation for unknown object_id
        assert mapping.to_track_id(500) == "carla_obj_0500"

    def test_realistic_scene_object_mapping(self):
        """Test realistic scene with vehicle and walker objects"""
        mapping = TrackIDMapping(auto_prefix="scene_obj")

        # Typical CARLA track_ids
        carla_ids = [
            "hero",           # Ego vehicle
            "vehicle_001",    # NPC vehicle
            "vehicle_002",    # NPC vehicle
            "walker_001",     # Pedestrian
            "walker_002",     # Pedestrian
        ]

        # Corresponding DGGT object_ids
        dggt_ids = [0, 1, 2, 3, 4]

        # Register mappings
        for carla_id, dggt_id in zip(carla_ids, dggt_ids):
            mapping.register(carla_id, dggt_id)

        # Verify all conversions work
        assert mapping.to_object_id("hero") == 0
        assert mapping.to_object_id("vehicle_001") == 1
        assert mapping.to_object_id("vehicle_002") == 2
        assert mapping.to_object_id("walker_001") == 3
        assert mapping.to_object_id("walker_002") == 4

        # Reverse mappings
        assert mapping.to_track_id(0) == "hero"
        assert mapping.to_track_id(1) == "vehicle_001"
        assert mapping.to_track_id(2) == "vehicle_002"
        assert mapping.to_track_id(3) == "walker_001"
        assert mapping.to_track_id(4) == "walker_002"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])