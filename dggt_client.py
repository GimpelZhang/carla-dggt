# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT Client — gRPC client for testing DGGT server rendering.

Connects to a running dggt_server, sends RGBRenderRequest for frames 0–N-1,
and saves rendered images as JPEG.

Usage:
    python dggt_client.py \
        --host localhost \
        --port 50051 \
        --scene-dir /home/junchuan/e2e/dggt/output/waymo/training/scene1/0328/001 \
        --scene-id "0328/001" \
        --output-dir ./dggt_output \
        --num-frames 20
"""

import json
import os
import argparse
import logging
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
import grpc

from nre.grpc.protos import sensorsim_pb2 as sensorsim_pb
from nre.grpc.protos import common_pb2 as common_pb
from nre.grpc.protos import sensorsim_pb2_grpc as sensorsim_grpc

logger = logging.getLogger(__name__)

# 10 Hz Waymo dataset
FPS = 10.0
FRAME_DURATION_US = int(1_000_000 / FPS)  # 100,000 us


def matrix_to_pose(matrix_4x4):
    """Convert 4x4 c2w matrix to Pose protobuf message.

    Args:
        matrix_4x4: 4x4 numpy array (camera-to-world or object-to-world).

    Returns:
        common_pb.Pose with position and quaternion.
    """
    pos = matrix_4x4[:3, 3]
    rot = matrix_4x4[:3, :3]
    q_scipy = R.from_matrix(rot).as_quat()  # returns [x, y, z, w]

    return common_pb.Pose(
        vec=common_pb.Vec3(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
        quat=common_pb.Quat(
            w=float(q_scipy[3]),  # proto: w,x,y,z — scipy: x,y,z,w
            x=float(q_scipy[0]),
            y=float(q_scipy[1]),
            z=float(q_scipy[2]),
        ),
    )


def intrinsics_to_camera_spec(K_3x3, width, height):
    """Convert 3x3 intrinsic matrix to CameraSpec with opencv_fisheye_param.

    Args:
        K_3x3: 3x3 numpy intrinsic matrix.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        sensorsim_pb.CameraSpec populated with intrinsics.
    """
    spec = sensorsim_pb.CameraSpec(
        resolution_w=width,
        resolution_h=height,
    )
    spec.opencv_fisheye_param.focal_length_x = float(K_3x3[0, 0])
    spec.opencv_fisheye_param.focal_length_y = float(K_3x3[1, 1])
    spec.opencv_fisheye_param.principal_point_x = float(K_3x3[0, 2])
    spec.opencv_fisheye_param.principal_point_y = float(K_3x3[1, 2])
    return spec


def objects_to_dynamic(objects_json):
    """Convert dynamic_objects JSON array to DynamicObject protobuf list.

    Args:
        objects_json: List of dicts with 'object_id', 'pose_world', 'dimensions'.

    Returns:
        List of sensorsim_pb.DynamicObject.
    """
    dyn_objs = []
    for obj in objects_json:
        pose = matrix_to_pose(np.array(obj["pose_world"]))
        track_id = f"dggt_obj_{obj['object_id']:04d}"
        dyn_obj = sensorsim_pb.DynamicObject(
            track_id=track_id,
            pose_pair=sensorsim_pb.PosePair(
                start_pose=pose,  # static pose (no motion data)
                end_pose=pose,
            ),
        )
        dyn_objs.append(dyn_obj)
    return dyn_objs


def load_frame_data(scene_dir, frame_idx):
    """Load ego pose and dynamic objects for a frame.

    Args:
        scene_dir: Base path to the DGGT scene directory.
        frame_idx: Frame index (0-based).

    Returns:
        Tuple of (ego_data dict, objects_list list).
    """
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


def build_render_request(scene_id, ego_data, objects_list, frame_idx):
    """Build an RGBRenderRequest from loaded frame data.

    Args:
        scene_id: Scene identifier string (e.g. "0328/001").
        ego_data: Ego pose JSON dict.
        objects_list: Dynamic objects JSON list.
        frame_idx: Frame index for timestamp computation.

    Returns:
        sensorsim_pb.RGBRenderRequest.
    """
    width = ego_data["camera"]["width"]
    height = ego_data["camera"]["height"]
    K = np.array(ego_data["camera_intrinsics"])
    c2w = np.array(ego_data["camera_extrinsics_world"])

    frame_start_us = frame_idx * FRAME_DURATION_US
    frame_end_us = (frame_idx + 1) * FRAME_DURATION_US

    sensor_pose = matrix_to_pose(c2w)

    request = sensorsim_pb.RGBRenderRequest(
        scene_id=scene_id,
        resolution_w=width,
        resolution_h=height,
        camera_intrinsics=intrinsics_to_camera_spec(K, width, height),
        frame_start_us=frame_start_us,
        frame_end_us=frame_end_us,
        sensor_pose=sensorsim_pb.PosePair(
            start_pose=sensor_pose,
            end_pose=sensor_pose,
        ),
        dynamic_objects=objects_to_dynamic(objects_list),
        image_format=sensorsim_pb.ImageFormat.JPEG,
        image_quality=95.0,
    )
    return request


def main():
    parser = argparse.ArgumentParser(description="DGGT gRPC rendering client")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=50051, help="Server port (default: 50051)")
    parser.add_argument("--scene-dir", required=True, help="Path to DGGT scene directory")
    parser.add_argument("--scene-id", required=True, help='Scene ID string (e.g. "0328/001")')
    parser.add_argument("--output-dir", default="./dggt_output", help="Output directory for images")
    parser.add_argument("--num-frames", type=int, default=20, help="Number of frames to render")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Connect to server
    target = f"{args.host}:{args.port}"
    logger.info(f"Connecting to DGGT server at {target}")
    channel = grpc.insecure_channel(target)
    stub = sensorsim_grpc.SensorsimServiceStub(channel)

    # Verify server
    try:
        version = stub.get_version(common_pb.Empty())
        logger.info(f"Server version: {version.version_id} (git: {version.git_hash})")
    except grpc.RpcError as e:
        logger.error(f"Failed to connect to server: {e.details()}")
        sys.exit(1)

    try:
        scenes = stub.get_available_scenes(common_pb.Empty())
        logger.info(f"Available scenes: {list(scenes.scene_ids)}")
        if args.scene_id not in scenes.scene_ids:
            logger.warning(f"Scene '{args.scene_id}' not in available scenes list — proceeding anyway")
    except grpc.RpcError as e:
        logger.warning(f"Failed to get available scenes: {e.details()}")

    # Render frames
    success_count = 0
    for frame_idx in range(args.num_frames):
        logger.info(f"Rendering frame {frame_idx}/{args.num_frames - 1}...")

        try:
            ego_data, objects_list = load_frame_data(args.scene_dir, frame_idx)
        except FileNotFoundError as e:
            logger.error(f"Missing data file for frame {frame_idx}: {e}")
            continue

        request = build_render_request(args.scene_id, ego_data, objects_list, frame_idx)

        try:
            response = stub.render_rgb(request)
        except grpc.RpcError as e:
            logger.error(f"gRPC error rendering frame {frame_idx}: {e.code()} - {e.details()}")
            continue

        if not response.image_bytes:
            logger.warning(f"Empty image response for frame {frame_idx}")
            continue

        output_path = os.path.join(args.output_dir, f"frame_{frame_idx:04d}.jpg")
        with open(output_path, "wb") as f:
            f.write(response.image_bytes)

        logger.info(f"Saved {output_path} ({len(response.image_bytes)} bytes)")
        success_count += 1

    logger.info(f"Done: {success_count}/{args.num_frames} frames rendered to {args.output_dir}")


if __name__ == "__main__":
    main()
