# Test Data Directory

This directory contains test scene data for unit testing.

## scene_mini

A minimal test scene with 5 frames for unit testing.

To populate this directory, run:
```bash
./tests/scripts/prepare_test_data.sh
```

Source scene: `/home/junchuan/e2e/dggt/output/waymo/training/scene1/0328/001`

Expected contents after running the script:
- `ego_pose/` - Ego pose JSON files for frames 0-4
- `dynamic_objects/` - Dynamic object data for frames 0-4  
- `gaussians/` - Gaussian splat data (if available)
- `scene_metadata.json` - Scene metadata
- `static_scene.ply` - Static scene PLY file
- `map.xodr` - OpenDRIVE map file (if available)