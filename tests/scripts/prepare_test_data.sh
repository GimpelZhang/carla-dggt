#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Prepare Test Data Script

Copies test data from source DGGT scene to test_data/scene_mini directory.
Creates a minimal test scene with 5 frames for unit testing.

Usage:
    ./prepare_test_data.sh

Source scene: /home/junchuan/e2e/dggt/output/waymo/training/scene1/0328/001
"""

set -e  # Exit on error

# Configuration
SOURCE_SCENE="/home/junchuan/e2e/dggt/output/waymo/training/scene1/0328/001"
TEST_DATA_DIR="$(dirname "$0")/../test_data"
TARGET_SCENE="${TEST_DATA_DIR}/scene_mini"
NUM_FRAMES=5

echo "=== Preparing Test Data ==="
echo "Source: ${SOURCE_SCENE}"
echo "Target: ${TARGET_SCENE}"
echo "Frames: ${NUM_FRAMES}"

# Check source exists
if [ ! -d "${SOURCE_SCENE}" ]; then
    echo "ERROR: Source scene not found: ${SOURCE_SCENE}"
    exit 1
fi

# Create target directory
mkdir -p "${TARGET_SCENE}"
mkdir -p "${TARGET_SCENE}/ego_pose"
mkdir -p "${TARGET_SCENE}/dynamic_objects"
mkdir -p "${TARGET_SCENE}/gaussians"

# Copy ego_pose data (first N frames)
echo "Copying ego_pose data..."
for i in $(seq 0 $((NUM_FRAMES - 1))); do
    # Find ego_pose files matching pattern view_*.json or similar
    if ls "${SOURCE_SCENE}/ego_pose/"*"${i}"*.json 1>/dev/null 2>&1; then
        cp "${SOURCE_SCENE}/ego_pose/"*"${i}"*.json "${TARGET_SCENE}/ego_pose/" 2>/dev/null || true
    fi
done

# If ego_pose has numbered files like view_0.json, view_1.json, etc.
for pattern in "view_" "frame_" "pose_"; do
    for i in $(seq 0 $((NUM_FRAMES - 1))); do
        if [ -f "${SOURCE_SCENE}/ego_pose/${pattern}${i}.json" ]; then
            cp "${SOURCE_SCENE}/ego_pose/${pattern}${i}.json" "${TARGET_SCENE}/ego_pose/"
        fi
        # Also copy corresponding .npy files if they exist
        if [ -f "${SOURCE_SCENE}/${pattern}${i}.npy" ]; then
            cp "${SOURCE_SCENE}/${pattern}${i}.npy" "${TARGET_SCENE}/"
        fi
    done
done

# Copy dynamic_objects data
echo "Copying dynamic_objects data..."
if [ -d "${SOURCE_SCENE}/dynamic_objects" ]; then
    # Copy first N frames of dynamic object data
    for i in $(seq 0 $((NUM_FRAMES - 1))); do
        if ls "${SOURCE_SCENE}/dynamic_objects/"*"${i}"*.json 1>/dev/null 2>&1; then
            cp "${SOURCE_SCENE}/dynamic_objects/"*"${i}"*.json "${TARGET_SCENE}/dynamic_objects/" 2>/dev/null || true
        fi
    done
fi

# Copy static scene PLY if exists
echo "Copying static scene..."
for ply_name in "static_scene.ply" "background.ply" "scene.ply"; do
    if [ -f "${SOURCE_SCENE}/${ply_name}" ]; then
        cp "${SOURCE_SCENE}/${ply_name}" "${TARGET_SCENE}/"
        break
    fi
done

# Copy sky scene if exists
for sky_name in "sky_scene.ply" "sky.ply"; do
    if [ -f "${SOURCE_SCENE}/${sky_name}" ]; then
        cp "${SOURCE_SCENE}/${sky_name}" "${TARGET_SCENE}/"
        break
    fi
done

# Copy map file if exists
if [ -f "${SOURCE_SCENE}/map.xodr" ]; then
    cp "${SOURCE_SCENE}/map.xodr" "${TARGET_SCENE}/"
fi

# Create scene metadata JSON
echo "Creating scene metadata..."
cat > "${TARGET_SCENE}/scene_metadata.json" << EOF
{
    "scene_id": "scene_mini",
    "num_frames": ${NUM_FRAMES},
    "fps": 10.0,
    "start_timestamp_us": 0,
    "end_timestamp_us": $((NUM_FRAMES - 1))00000,
    "camera_width": 1036,
    "camera_height": 700,
    "intrinsic_matrix": [
        [1000.0, 0.0, 518.0],
        [0.0, 1000.0, 350.0],
        [0.0, 0.0, 1.0]
    ],
    "intrinsics_vary": false,
    "has_static_scene": true,
    "has_sky_scene": true,
    "dynamic_object_ids": [0, 1, 2],
    "source_path": "${SOURCE_SCENE}"
}
EOF

# List what was copied
echo ""
echo "=== Test Scene Contents ==="
ls -la "${TARGET_SCENE}/"
echo ""
echo "ego_pose files:"
ls -la "${TARGET_SCENE}/ego_pose/" 2>/dev/null || echo "  (empty)"
echo ""
echo "dynamic_objects files:"
ls -la "${TARGET_SCENE}/dynamic_objects/" 2>/dev/null || echo "  (empty)"
echo ""
echo "=== Done ==="