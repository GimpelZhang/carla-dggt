#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Compare dggt_client.py vs example_dggt_replay.py gRPC request behavior.

This script simulates both implementations' request building process and
compares the resulting gRPC requests for sensor_pose and dynamic_objects.

Usage:
    python compare_dggt_requests.py \
        --scene-dir /path/to/scene \
        --scene-id "0328/001" \
        --frame 0

Key Comparisons:
    1. sensor_pose values (position and quaternion)
    2. dynamic_objects count and poses
    3. Request structure differences
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# Add module path for imports
sys.path.insert(0, str(Path(__file__).parent))

# gRPC imports
from nre.grpc.protos import sensorsim_pb2 as sensorsim_pb
from nre.grpc.protos import common_pb2 as common_pb

# Import utils.py pose conversion (used by dggt_integration.py)
from utils import se3_to_grpc_pose


# ==============================================================================
# dggt_client.py implementation (LOCAL pose conversion)
# ==============================================================================

def matrix_to_pose_client(matrix_4x4):
    """dggt_client.py's local matrix_to_pose implementation."""
    pos = matrix_4x4[:3, 3]
    rot = matrix_4x4[:3, :3]
    q_scipy = R.from_matrix(rot).as_quat()  # returns [x, y, z, w]

    return common_pb.Pose(
        vec=common_pb.Vec3(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
        quat=common_pb.Quat(
            w=float(q_scipy[3]),  # proto: w,x,y,z -- scipy: x,y,z,w
            x=float(q_scipy[0]),
            y=float(q_scipy[1]),
            z=float(q_scipy[2]),
        ),
    )


def intrinsics_to_camera_spec_client(K_3x3, width, height):
    """dggt_client.py's intrinsics conversion."""
    spec = sensorsim_pb.CameraSpec(
        resolution_w=width,
        resolution_h=height,
    )
    spec.opencv_fisheye_param.focal_length_x = float(K_3x3[0, 0])
    spec.opencv_fisheye_param.focal_length_y = float(K_3x3[1, 1])
    spec.opencv_fisheye_param.principal_point_x = float(K_3x3[0, 2])
    spec.opencv_fisheye_param.principal_point_y = float(K_3x3[1, 2])
    return spec


def objects_to_dynamic_client(objects_json):
    """dggt_client.py's objects_to_dynamic implementation."""
    dyn_objs = []
    for obj in objects_json:
        pose = matrix_to_pose_client(np.array(obj["pose_world"]))
        track_id = f"dggt_obj_{obj['object_id']:04d}"
        dyn_obj = sensorsim_pb.DynamicObject(
            track_id=track_id,
            pose_pair=sensorsim_pb.PosePair(
                start_pose=pose,
                end_pose=pose,
            ),
        )
        dyn_objs.append(dyn_obj)
    return dyn_objs


def build_request_client(scene_id, ego_data, objects_list, frame_idx):
    """Build request using dggt_client.py's approach."""
    FPS = 10.0
    FRAME_DURATION_US = int(1_000_000 / FPS)

    width = ego_data["camera"]["width"]
    height = ego_data["camera"]["height"]
    K = np.array(ego_data["camera_intrinsics"])
    c2w = np.array(ego_data["camera_extrinsics_world"])

    frame_start_us = frame_idx * FRAME_DURATION_US
    frame_end_us = (frame_idx + 1) * FRAME_DURATION_US

    sensor_pose = matrix_to_pose_client(c2w)

    request = sensorsim_pb.RGBRenderRequest(
        scene_id=scene_id,
        resolution_w=width,
        resolution_h=height,
        camera_intrinsics=intrinsics_to_camera_spec_client(K, width, height),
        frame_start_us=frame_start_us,
        frame_end_us=frame_end_us,
        sensor_pose=sensorsim_pb.PosePair(
            start_pose=sensor_pose,
            end_pose=sensor_pose,
        ),
        dynamic_objects=objects_to_dynamic_client(objects_list),
        image_format=sensorsim_pb.ImageFormat.JPEG,
        image_quality=95.0,
    )
    return request


# ==============================================================================
# dggt_integration.py implementation (uses utils.py pose conversion)
# ==============================================================================

def objects_to_dynamic_integration(dynamic_objects_list):
    """dggt_integration.py's _convert_preloaded_dynamic_objects approach."""
    dyn_objs = []
    for obj in dynamic_objects_list:
        # pose_world is already numpy array in DGGT coords
        pose = se3_to_grpc_pose(obj["pose_world"])

        # Same track_id format as dggt_client.py
        track_id = f"dggt_obj_{obj.get('object_id', 0):04d}"

        dyn_objs.append(sensorsim_pb.DynamicObject(
            track_id=track_id,
            pose_pair=sensorsim_pb.PosePair(start_pose=pose, end_pose=pose),
        ))
    return dyn_objs


def build_request_integration(scene_id, ego_data, objects_list, frame_idx):
    """Build request using dggt_integration.py's approach.

    This simulates the DggtSensor.on_tick -> DggtRenderer.render path.
    Key difference: uses se3_to_grpc_pose from utils.py instead of local matrix_to_pose.
    """
    FPS = 10.0
    FRAME_DURATION_US = int(1_000_000 / FPS)

    width = ego_data["camera"]["width"]
    height = ego_data["camera"]["height"]
    K = np.array(ego_data["camera_intrinsics"])
    c2w = np.array(ego_data["camera_extrinsics_world"])

    frame_start_us = frame_idx * FRAME_DURATION_US
    # Note: dggt_integration.py uses frame_end_us = timestamp + 1
    # dggt_client.py uses frame_end_us = (frame_idx + 1) * FRAME_DURATION_US
    frame_end_us = frame_start_us + 1  # dggt_integration pattern

    # dggt_integration.py uses se3_to_grpc_pose for camera pose
    sensor_pose = se3_to_grpc_pose(c2w)

    # Build camera spec same way
    camera_spec = sensorsim_pb.CameraSpec(
        resolution_w=width,
        resolution_h=height,
    )
    camera_spec.opencv_fisheye_param.focal_length_x = float(K[0, 0])
    camera_spec.opencv_fisheye_param.focal_length_y = float(K[1, 1])
    camera_spec.opencv_fisheye_param.principal_point_x = float(K[0, 2])
    camera_spec.opencv_fisheye_param.principal_point_y = float(K[1, 2])

    # Convert objects list to dggt_integration format (numpy arrays)
    dynamic_objects_formatted = []
    for obj in objects_list:
        dynamic_objects_formatted.append({
            "object_id": obj["object_id"],
            "pose_world": np.array(obj["pose_world"]),
        })

    request = sensorsim_pb.RGBRenderRequest(
        scene_id=scene_id,
        resolution_w=width,
        resolution_h=height,
        camera_intrinsics=camera_spec,
        frame_start_us=frame_start_us,
        frame_end_us=frame_end_us,
        sensor_pose=sensorsim_pb.PosePair(
            start_pose=sensor_pose,
            end_pose=sensor_pose,
        ),
        dynamic_objects=objects_to_dynamic_integration(dynamic_objects_formatted),
        image_format=sensorsim_pb.ImageFormat.JPEG,
        image_quality=95.0,
    )
    return request


# ==============================================================================
# Comparison Functions
# ==============================================================================

def pose_to_dict(pose):
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


def compare_poses(pose1, pose2, name1="pose1", name2="pose2", tolerance=1e-6):
    """Compare two gRPC Pose objects."""
    pos_diff = np.array([
        pose1.vec.x - pose2.vec.x,
        pose1.vec.y - pose2.vec.y,
        pose1.vec.z - pose2.vec.z,
    ])

    quat_diff = np.array([
        pose1.quat.w - pose2.quat.w,
        pose1.quat.x - pose2.quat.x,
        pose1.quat.y - pose2.quat.y,
        pose1.quat.z - pose2.quat.z,
    ])

    pos_match = np.allclose(pos_diff, 0, atol=tolerance)
    quat_match = np.allclose(quat_diff, 0, atol=tolerance)

    return {
        "position_match": pos_match,
        "position_diff": pos_diff.tolist(),
        "quaternion_match": quat_match,
        "quaternion_diff": quat_diff.tolist(),
        "overall_match": pos_match and quat_match,
    }


def compare_dynamic_objects(dyn1, dyn2):
    """Compare two lists of DynamicObject."""
    result = {
        "count_match": len(dyn1) == len(dyn2),
        "count1": len(dyn1),
        "count2": len(dyn2),
        "objects": [],
    }

    if not result["count_match"]:
        return result

    # Sort by track_id for comparison
    dyn1_sorted = sorted(dyn1, key=lambda x: x.track_id)
    dyn2_sorted = sorted(dyn2, key=lambda x: x.track_id)

    for obj1, obj2 in zip(dyn1_sorted, dyn2_sorted):
        track_match = obj1.track_id == obj2.track_id
        pose_compare = compare_poses(obj1.pose_pair.start_pose, obj2.pose_pair.start_pose)

        result["objects"].append({
            "track_id": obj1.track_id,
            "track_id_match": track_match,
            "pose_comparison": pose_compare,
        })

    return result


def compare_requests(req1, req2, name1="dggt_client", name2="dggt_integration"):
    """Compare two RGBRenderRequest objects."""
    result = {
        "scene_id_match": req1.scene_id == req2.scene_id,
        "resolution_match": (
            req1.resolution_w == req2.resolution_w and
            req1.resolution_h == req2.resolution_h
        ),
        "frame_timestamp_match": (
            req1.frame_start_us == req2.frame_start_us and
            req1.frame_end_us == req2.frame_end_us
        ),
        "frame_timestamp_diff": {
            "start_diff": req1.frame_start_us - req2.frame_start_us,
            "end_diff": req1.frame_end_us - req2.frame_end_us,
        },
        "sensor_pose_comparison": compare_poses(
            req1.sensor_pose.start_pose,
            req2.sensor_pose.start_pose,
            name1,
            name2,
        ),
        "dynamic_objects_comparison": compare_dynamic_objects(
            req1.dynamic_objects,
            req2.dynamic_objects,
        ),
        "image_format_match": req1.image_format == req2.image_format,
        "image_quality_match": req1.image_quality == req2.image_quality,
    }

    return result


def print_comparison_result(result, name1="dggt_client", name2="dggt_integration"):
    """Print comparison result in readable format."""
    print("\n" + "=" * 80)
    print("REQUEST COMPARISON RESULT")
    print("=" * 80)

    print(f"\nScene ID Match: {result['scene_id_match']}")
    print(f"Resolution Match: {result['resolution_match']}")

    print(f"\nFrame Timestamp Match: {result['frame_timestamp_match']}")
    if not result['frame_timestamp_match']:
        diff = result['frame_timestamp_diff']
        print(f"  Start diff: {diff['start_diff']} us")
        print(f"  End diff: {diff['end_diff']} us")
        print("  NOTE: dggt_client uses (frame_idx+1)*FRAME_DURATION_US")
        print("        dggt_integration uses frame_start_us + 1")

    print(f"\n--- SENSOR POSE COMPARISON ---")
    pose_comp = result['sensor_pose_comparison']
    print(f"Position Match: {pose_comp['position_match']}")
    if not pose_comp['position_match']:
        pos_diff = pose_comp['position_diff']
        print(f"  Position diff: [{pos_diff[0]:.6f}, {pos_diff[1]:.6f}, {pos_diff[2]:.6f}]")
        print(f"  {name1} position: [{pose_comp['position1_x'] if 'position1_x' in pose_comp else 'N/A'}]")

    print(f"Quaternion Match: {pose_comp['quaternion_match']}")
    if not pose_comp['quaternion_match']:
        quat_diff = pose_comp['quaternion_diff']
        print(f"  Quaternion diff: [{quat_diff[0]:.6f}, {quat_diff[1]:.6f}, {quat_diff[2]:.6f}, {quat_diff[3]:.6f}]")

    print(f"\n--- DYNAMIC OBJECTS COMPARISON ---")
    dyn_comp = result['dynamic_objects_comparison']
    print(f"Count Match: {dyn_comp['count_match']}")
    print(f"  {name1} count: {dyn_comp['count1']}")
    print(f"  {name2} count: {dyn_comp['count2']}")

    if dyn_comp['count_match'] and dyn_comp['objects']:
        for obj_comp in dyn_comp['objects']:
            pose_comp = obj_comp['pose_comparison']
            if not pose_comp['overall_match']:
                print(f"\n  Object {obj_comp['track_id']}:")
                print(f"    Position match: {pose_comp['position_match']}")
                if not pose_comp['position_match']:
                    pos_diff = pose_comp['position_diff']
                    print(f"      Position diff: [{pos_diff[0]:.6f}, {pos_diff[1]:.6f}, {pos_diff[2]:.6f}]")
                print(f"    Quaternion match: {pose_comp['quaternion_match']}")
                if not pose_comp['quaternion_match']:
                    quat_diff = pose_comp['quaternion_diff']
                    print(f"      Quaternion diff: [{quat_diff[0]:.6f}, {quat_diff[1]:.6f}, {quat_diff[2]:.6f}, {quat_diff[3]:.6f}]")

    print(f"\n--- OTHER FIELDS ---")
    print(f"Image Format Match: {result['image_format_match']}")
    print(f"Image Quality Match: {result['image_quality_match']}")

    print("\n" + "=" * 80)


# ==============================================================================
# Pose Conversion Comparison (Detailed)
# ==============================================================================

def compare_pose_conversion_methods(matrix):
    """Compare matrix_to_pose_client vs se3_to_grpc_pose directly."""
    pose1 = matrix_to_pose_client(matrix)
    pose2 = se3_to_grpc_pose(matrix)

    print("\n--- POSE CONVERSION METHOD COMPARISON ---")
    print(f"Input matrix position: [{matrix[0,3]:.6f}, {matrix[1,3]:.6f}, {matrix[2,3]:.6f}]")

    # Extract rotation as quaternion for reference
    quat_ref = R.from_matrix(matrix[:3, :3]).as_quat()  # scipy: [x,y,z,w]
    print(f"Input rotation (scipy quat): [{quat_ref[0]:.6f}, {quat_ref[1]:.6f}, {quat_ref[2]:.6f}, {quat_ref[3]:.6f}]")

    print(f"\nmatrix_to_pose_client output:")
    print(f"  Position: [{pose1.vec.x:.6f}, {pose1.vec.y:.6f}, {pose1.vec.z:.6f}]")
    print(f"  Quaternion (w,x,y,z): [{pose1.quat.w:.6f}, {pose1.quat.x:.6f}, {pose1.quat.y:.6f}, {pose1.quat.z:.6f}]")

    print(f"\nse3_to_grpc_pose output:")
    print(f"  Position: [{pose2.vec.x:.6f}, {pose2.vec.y:.6f}, {pose2.vec.z:.6f}]")
    print(f"  Quaternion (w,x,y,z): [{pose2.quat.w:.6f}, {pose2.quat.x:.6f}, {pose2.quat.y:.6f}, {pose2.quat.z:.6f}]")

    comparison = compare_poses(pose1, pose2, "matrix_to_pose", "se3_to_grpc_pose")

    print(f"\nComparison result:")
    print(f"  Position match: {comparison['position_match']}")
    print(f"  Quaternion match: {comparison['quaternion_match']}")

    # Check for canonical=False difference
    quat_canonical = R.from_matrix(matrix[:3, :3]).as_quat(canonical=True)
    quat_noncanonical = R.from_matrix(matrix[:3, :3]).as_quat(canonical=False)

    print(f"\nCanonical vs Non-Canonical check:")
    print(f"  canonical=True: [{quat_canonical[0]:.6f}, {quat_canonical[1]:.6f}, {quat_canonical[2]:.6f}, {quat_canonical[3]:.6f}]")
    print(f"  canonical=False: [{quat_noncanonical[0]:.6f}, {quat_noncanonical[1]:.6f}, {quat_noncanonical[2]:.6f}, {quat_noncanonical[3]:.6f}]")
    canonical_diff = np.allclose(quat_canonical, quat_noncanonical)
    print(f"  Are they same? {canonical_diff}")

    return comparison


# ==============================================================================
# Main
# ==============================================================================

def load_frame_data(scene_dir, frame_idx):
    """Load ego pose and dynamic objects for a frame."""
    ego_path = os.path.join(scene_dir, "ego_pose", f"frame_{frame_idx:04d}_ego.json")
    objects_path = os.path.join(scene_dir, "dynamic_objects", f"frame_{frame_idx:04d}_objects.json")

    with open(ego_path, "r") as f:
        ego_data = json.load(f)

    if os.path.exists(objects_path):
        with open(objects_path, "r") as f:
            objects_list = json.load(f)
    else:
        objects_list = []

    return ego_data, objects_list


def main():
    parser = argparse.ArgumentParser(description="Compare dggt_client vs dggt_integration request building")
    parser.add_argument("--scene-dir", required=True, help="Path to DGGT scene directory")
    parser.add_argument("--scene-id", required=True, help='Scene ID string (e.g. "0328/001")')
    parser.add_argument("--frame", type=int, default=0, help="Frame index to compare (default: 0)")
    args = parser.parse_args()

    print("=" * 80)
    print("DGGT REQUEST COMPARISON TOOL")
    print("=" * 80)
    print(f"Scene directory: {args.scene_dir}")
    print(f"Scene ID: {args.scene_id}")
    print(f"Frame index: {args.frame}")

    # Load frame data
    try:
        ego_data, objects_list = load_frame_data(args.scene_dir, args.frame)
        print(f"\nLoaded ego pose data with {len(objects_list)} dynamic objects")
    except FileNotFoundError as e:
        print(f"ERROR: Failed to load frame data: {e}")
        sys.exit(1)

    # Print raw data for reference
    print("\n--- RAW INPUT DATA ---")
    c2w = np.array(ego_data["camera_extrinsics_world"])
    print(f"Camera extrinsics (c2w) position: [{c2w[0,3]:.6f}, {c2w[1,3]:.6f}, {c2w[2,3]:.6f}]")
    print(f"Dynamic objects count: {len(objects_list)}")

    if objects_list:
        print(f"\nFirst dynamic object pose_world position:")
        obj_pose = np.array(objects_list[0]["pose_world"])
        print(f"  [{obj_pose[0,3]:.6f}, {obj_pose[1,3]:.6f}, {obj_pose[2,3]:.6f}]")

    # Compare pose conversion methods directly
    compare_pose_conversion_methods(c2w)

    # Build requests using both approaches
    print("\n--- BUILDING REQUESTS ---")
    request_client = build_request_client(args.scene_id, ego_data, objects_list, args.frame)
    request_integration = build_request_integration(args.scene_id, ego_data, objects_list, args.frame)

    print("Built request using dggt_client.py approach")
    print("Built request using dggt_integration.py approach")

    # Compare requests
    comparison_result = compare_requests(request_client, request_integration)
    print_comparison_result(comparison_result)

    # Summary
    print("\n--- SUMMARY ---")
    pose_match = comparison_result['sensor_pose_comparison']['overall_match']
    dyn_match = all(
        obj['pose_comparison']['overall_match']
        for obj in comparison_result['dynamic_objects_comparison']['objects']
    ) if comparison_result['dynamic_objects_comparison']['count_match'] else False

    print(f"Sensor pose matches: {pose_match}")
    print(f"All dynamic object poses match: {dyn_match}")

    if pose_match and dyn_match:
        print("\nCONCLUSION: Both implementations produce identical gRPC requests")
        print("  (ignoring frame_end_us difference)")
    else:
        print("\nCONCLUSION: DIFFERENCES FOUND - investigate further")
        if not pose_match:
            print("  - Sensor pose mismatch detected")
        if not dyn_match:
            print("  - Dynamic object pose mismatch detected")


if __name__ == "__main__":
    main()