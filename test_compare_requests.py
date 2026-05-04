#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Compare gRPC requests from dggt_client.py and dggt_integration.py (example_dggt_replay.py).

This script intercepts the gRPC render_rgb calls from both clients and compares:
1. sensor_pose values (position and quaternion)
2. dynamic_objects count and poses
3. Any differences in request structure

Usage:
    python test_compare_requests.py --scene-dir <scene_dir> --scene-id <scene_id> --frame <frame_idx>
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add module path
sys.path.insert(0, str(Path(__file__).parent))

from nre.grpc.protos import sensorsim_pb2 as sensorsim_pb
from nre.grpc.protos import common_pb2 as common_pb

# Import from dggt_client.py
from dggt_client import matrix_to_pose, intrinsics_to_camera_spec, objects_to_dynamic, load_frame_data, build_render_request

# Import from dggt_integration.py and utils.py
from utils import se3_to_grpc_pose
from dggt_integration import DggtRenderer
from dggt_config import DggtConfigLoader

# Constants
FPS = 10.0
FRAME_DURATION_US = int(1_000_000 / FPS)


def pose_to_dict(pose: common_pb.Pose) -> dict:
    """Convert gRPC Pose to dict for comparison."""
    return {
        "position": {
            "x": pose.vec.x,
            "y": pose.vec.y,
            "z": pose.vec.z,
        },
        "quaternion": {
            "w": pose.quat.w,
            "x": pose.quat.x,
            "y": pose.quat.y,
            "z": pose.quat.z,
        },
    }


def pose_pair_to_dict(pose_pair: sensorsim_pb.PosePair) -> dict:
    """Convert gRPC PosePair to dict for comparison."""
    return {
        "start_pose": pose_to_dict(pose_pair.start_pose),
        "end_pose": pose_to_dict(pose_pair.end_pose),
    }


def dynamic_object_to_dict(obj: sensorsim_pb.DynamicObject) -> dict:
    """Convert gRPC DynamicObject to dict for comparison."""
    return {
        "track_id": obj.track_id,
        "pose_pair": pose_pair_to_dict(obj.pose_pair),
    }


def request_to_dict(request: sensorsim_pb.RGBRenderRequest) -> dict:
    """Convert gRPC RGBRenderRequest to dict for comparison."""
    return {
        "scene_id": request.scene_id,
        "resolution_w": request.resolution_w,
        "resolution_h": request.resolution_h,
        "frame_start_us": request.frame_start_us,
        "frame_end_us": request.frame_end_us,
        "sensor_pose": pose_pair_to_dict(request.sensor_pose),
        "dynamic_objects": [dynamic_object_to_dict(obj) for obj in request.dynamic_objects],
        "image_format": str(request.image_format),
        "image_quality": request.image_quality,
    }


def compare_dicts(dict1: dict, dict2: dict, path: str = "") -> list:
    """Compare two dicts and return list of differences."""
    differences = []

    for key in dict1:
        current_path = f"{path}.{key}" if path else key

        if key not in dict2:
            differences.append(f"{current_path}: missing in dict2")
            continue

        val1 = dict1[key]
        val2 = dict2[key]

        if isinstance(val1, dict) and isinstance(val2, dict):
            differences.extend(compare_dicts(val1, val2, current_path))
        elif isinstance(val1, list) and isinstance(val2, list):
            if len(val1) != len(val2):
                differences.append(f"{current_path}: list lengths differ ({len(val1)} vs {len(val2)})")
            else:
                for i, (item1, item2) in enumerate(zip(val1, val2)):
                    if isinstance(item1, dict) and isinstance(item2, dict):
                        differences.extend(compare_dicts(item1, item2, f"{current_path}[{i}]"))
                    elif item1 != item2:
                        differences.append(f"{current_path}[{i}]: {item1} vs {item2}")
        elif isinstance(val1, float) and isinstance(val2, float):
            if not np.isclose(val1, val2, rtol=1e-5, atol=1e-8):
                differences.append(f"{current_path}: {val1} vs {val2} (diff: {abs(val1 - val2)})")
        elif val1 != val2:
            differences.append(f"{current_path}: {val1} vs {val2}")

    return differences


def build_request_from_dggt_client(scene_id: str, scene_dir: str, frame_idx: int) -> sensorsim_pb.RGBRenderRequest:
    """Build request exactly as dggt_client.py does."""
    ego_data, objects_list = load_frame_data(scene_dir, frame_idx)
    return build_render_request(scene_id, ego_data, objects_list, frame_idx)


def build_request_from_dggt_integration(scene_id: str, scene_dir: str, frame_idx: int) -> sensorsim_pb.RGBRenderRequest:
    """Build request exactly as dggt_integration.py does (scene replay mode)."""

    # Load ego pose (same as DggtScenario._load_ego_pose)
    ego_path = os.path.join(scene_dir, "ego_pose", f"frame_{frame_idx:04d}_ego.json")
    with open(ego_path, "r") as f:
        ego_data = json.load(f)

    # Extract data
    c2w = np.array(ego_data["camera_extrinsics_world"])
    width = ego_data["camera"]["width"]
    height = ego_data["camera"]["height"]
    K = np.array(ego_data["camera_intrinsics"])

    # Build camera spec
    camera_spec = intrinsics_to_camera_spec(K, width, height)

    # Scene replay mode: use_raw_pose=True, no transforms
    # dggt_integration.py uses se3_to_grpc_pose (from utils.py)
    pose = se3_to_grpc_pose(c2w)

    # Load dynamic objects
    objects_path = os.path.join(scene_dir, "dynamic_objects", f"frame_{frame_idx:04d}_objects.json")
    if os.path.exists(objects_path):
        with open(objects_path, "r") as f:
            objects_list = json.load(f)
    else:
        objects_list = []

    # Convert dynamic objects using dggt_integration.py pattern
    # DggtRenderer._convert_preloaded_dynamic_objects (lines 377-412)
    dynamic_objects_pb = []
    for obj in objects_list:
        pose_world = np.array(obj["pose_world"])
        obj_pose = se3_to_grpc_pose(pose_world)
        # Note: dggt_integration uses .get('object_id', 0) with default
        track_id = f"dggt_obj_{obj.get('object_id', 0):04d}"
        dynamic_objects_pb.append(sensorsim_pb.DynamicObject(
            track_id=track_id,
            pose_pair=sensorsim_pb.PosePair(start_pose=obj_pose, end_pose=obj_pose),
        ))

    # Build timestamp
    timestamp_us = frame_idx * FRAME_DURATION_US

    # Build request (matches DggtRenderer._build_render_request pattern)
    request = sensorsim_pb.RGBRenderRequest(
        scene_id=scene_id,
        resolution_w=width,
        resolution_h=height,
        camera_intrinsics=camera_spec,
        frame_start_us=timestamp_us,
        frame_end_us=timestamp_us + 1,  # dggt_integration uses +1
        sensor_pose=sensorsim_pb.PosePair(
            start_pose=pose,
            end_pose=pose,
        ),
        dynamic_objects=dynamic_objects_pb,
        image_format=sensorsim_pb.ImageFormat.JPEG,
        image_quality=95,
    )

    return request


def compare_pose_conversion():
    """Compare matrix_to_pose vs se3_to_grpc_pose."""
    print("\n" + "=" * 60)
    print("COMPARING POSE CONVERSION FUNCTIONS")
    print("=" * 60)

    # Create a test matrix
    test_matrix = np.array([
        [0.99, -0.05, 0.01, 10.5],
        [0.05, 0.98, -0.02, -5.3],
        [-0.01, 0.02, 0.99, 2.1],
        [0, 0, 0, 1],
    ])

    # dggt_client.py approach
    pose_client = matrix_to_pose(test_matrix)

    # dggt_integration.py approach
    pose_integration = se3_to_grpc_pose(test_matrix)

    dict_client = pose_to_dict(pose_client)
    dict_integration = pose_to_dict(pose_integration)

    differences = compare_dicts(dict_client, dict_integration, "pose")

    if differences:
        print("DIFFERENCES FOUND:")
        for diff in differences:
            print(f"  - {diff}")
    else:
        print("NO DIFFERENCES - Functions are equivalent")

    # Print both for visual comparison
    print("\ndggt_client.py matrix_to_pose:")
    print(f"  Position: ({dict_client['position']['x']}, {dict_client['position']['y']}, {dict_client['position']['z']})")
    print(f"  Quaternion: w={dict_client['quaternion']['w']}, x={dict_client['quaternion']['x']}, y={dict_client['quaternion']['y']}, z={dict_client['quaternion']['z']}")

    print("\ndggt_integration.py se3_to_grpc_pose:")
    print(f"  Position: ({dict_integration['position']['x']}, {dict_integration['position']['y']}, {dict_integration['position']['z']})")
    print(f"  Quaternion: w={dict_integration['quaternion']['w']}, x={dict_integration['quaternion']['x']}, y={dict_integration['quaternion']['y']}, z={dict_integration['quaternion']['z']}")


def main():
    parser = argparse.ArgumentParser(description="Compare gRPC requests from dggt_client.py and dggt_integration.py")
    parser.add_argument("--scene-dir", required=True, help="Path to DGGT scene directory")
    parser.add_argument("--scene-id", required=True, help="Scene ID string (e.g. '0328/001')")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to compare (default: 0)")
    args = parser.parse_args()

    print("=" * 60)
    print("GRPC REQUEST COMPARISON")
    print("=" * 60)
    print(f"Scene: {args.scene_id}")
    print(f"Scene dir: {args.scene_dir}")
    print(f"Frame: {args.frame}")

    # First, compare pose conversion functions
    compare_pose_conversion()

    # Build requests from both methods
    print("\n" + "=" * 60)
    print("BUILDING REQUESTS")
    print("=" * 60)

    request_client = build_request_from_dggt_client(args.scene_id, args.scene_dir, args.frame)
    request_integration = build_request_from_dggt_integration(args.scene_id, args.scene_dir, args.frame)

    dict_client = request_to_dict(request_client)
    dict_integration = request_to_dict(request_integration)

    # Compare requests
    print("\n" + "=" * 60)
    print("REQUEST COMPARISON RESULTS")
    print("=" * 60)

    differences = compare_dicts(dict_client, dict_integration)

    if differences:
        print("DIFFERENCES FOUND:")
        for diff in differences:
            print(f"  - {diff}")
    else:
        print("NO DIFFERENCES - Requests are identical")

    # Detailed output
    print("\n" + "=" * 60)
    print("DETAILED REQUEST DATA")
    print("=" * 60)

    print("\ndggt_client.py request:")
    print(f"  scene_id: {dict_client['scene_id']}")
    print(f"  resolution: {dict_client['resolution_w']}x{dict_client['resolution_h']}")
    print(f"  frame_start_us: {dict_client['frame_start_us']}")
    print(f"  frame_end_us: {dict_client['frame_end_us']}")
    print(f"  sensor_pose.start: pos=({dict_client['sensor_pose']['start_pose']['position']['x']:.3f}, {dict_client['sensor_pose']['start_pose']['position']['y']:.3f}, {dict_client['sensor_pose']['start_pose']['position']['z']:.3f})")
    print(f"  sensor_pose.start: quat=({dict_client['sensor_pose']['start_pose']['quaternion']['w']:.3f}, {dict_client['sensor_pose']['start_pose']['quaternion']['x']:.3f}, {dict_client['sensor_pose']['start_pose']['quaternion']['y']:.3f}, {dict_client['sensor_pose']['start_pose']['quaternion']['z']:.3f})")
    print(f"  dynamic_objects: {len(dict_client['dynamic_objects'])}")

    print("\ndggt_integration.py request:")
    print(f"  scene_id: {dict_integration['scene_id']}")
    print(f"  resolution: {dict_integration['resolution_w']}x{dict_integration['resolution_h']}")
    print(f"  frame_start_us: {dict_integration['frame_start_us']}")
    print(f"  frame_end_us: {dict_integration['frame_end_us']}")
    print(f"  sensor_pose.start: pos=({dict_integration['sensor_pose']['start_pose']['position']['x']:.3f}, {dict_integration['sensor_pose']['start_pose']['position']['y']:.3f}, {dict_integration['sensor_pose']['start_pose']['position']['z']:.3f})")
    print(f"  sensor_pose.start: quat=({dict_integration['sensor_pose']['start_pose']['quaternion']['w']:.3f}, {dict_integration['sensor_pose']['start_pose']['quaternion']['x']:.3f}, {dict_integration['sensor_pose']['start_pose']['quaternion']['y']:.3f}, {dict_integration['sensor_pose']['start_pose']['quaternion']['z']:.3f})")
    print(f"  dynamic_objects: {len(dict_integration['dynamic_objects'])}")

    # Compare dynamic objects
    if len(dict_client['dynamic_objects']) > 0 or len(dict_integration['dynamic_objects']) > 0:
        print("\n" + "=" * 60)
        print("DYNAMIC OBJECTS COMPARISON")
        print("=" * 60)

        for i, (obj_client, obj_integration) in enumerate(zip(dict_client['dynamic_objects'], dict_integration['dynamic_objects'])):
            print(f"\nObject {i}:")
            print(f"  dggt_client.py track_id: {obj_client['track_id']}")
            print(f"  dggt_integration.py track_id: {obj_integration['track_id']}")

            # Compare poses
            pose_diffs = compare_dicts(obj_client['pose_pair'], obj_integration['pose_pair'], f"obj_{i}")
            if pose_diffs:
                print(f"  Pose differences:")
                for diff in pose_diffs:
                    print(f"    - {diff}")
            else:
                print(f"  Poses: IDENTICAL")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if differences:
        print(f"Total differences found: {len(differences)}")
        print("\nConclusion: Requests differ - this could indicate a bug in one of the clients")
    else:
        print("Requests are IDENTICAL")
        print("\nConclusion: If dynamic object poses are incorrect, the issue is likely SERVER-SIDE")
        print("  (request_converter.py or dggt_service.py), not in the client code")


if __name__ == "__main__":
    main()