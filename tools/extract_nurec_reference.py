# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
NuRec Reference Data Extraction Tool

Extracts coordinate transform reference data from NuRec rendering output
for use in validation tests.

Usage:
    python tools/extract_nurec_reference.py --input nurec_output/ --output tests/test_data/nurec_reference/
"""

import argparse
import json
import glob
import os
from pathlib import Path
import numpy as np


def extract_reference_data(nurec_output_dir: str, output_dir: str):
    """
    Extract transformation reference data from NuRec output

    Args:
        nurec_output_dir: Path to NuRec rendering output directory
        output_dir: Path to output directory for reference data
    """
    reference_data = {
        'description': 'NuRec coordinate transform reference data',
        'generated': str(Path(nurec_output_dir).stat().st_mtime if Path(nurec_output_dir).exists() else 'unknown'),
        'coordinate_systems': {
            'carla': {
                'type': 'left-handed',
                'axes': {'X': 'Forward', 'Y': 'Right', 'Z': 'Up'}
            },
            'dggt': {
                'type': 'right-handed',
                'axes': {'X': 'Right', 'Y': 'Down', 'Z': 'Forward'}
            }
        },
        'camera_poses': [],
        'dynamic_objects': []
    }

    # Search for metadata files in NuRec output
    metadata_patterns = [
        f"{nurec_output_dir}/**/metadata_*.json",
        f"{nurec_output_dir}/**/frame_*.json",
        f"{nurec_output_dir}/**/transforms.json"
    ]

    found_files = []
    for pattern in metadata_patterns:
        found_files.extend(glob.glob(pattern, recursive=True))

    if not found_files:
        print(f"Warning: No metadata files found in {nurec_output_dir}")
        print("Creating sample reference data for testing...")
        create_sample_reference_data(reference_data)
    else:
        print(f"Found {len(found_files)} metadata files")

        for meta_file in found_files:
            try:
                with open(meta_file) as f:
                    metadata = json.load(f)

                # Extract camera poses
                if 'frames' in metadata:
                    for frame in metadata['frames']:
                        reference_data['camera_poses'].append({
                            'id': frame.get('frame_id', f'frame_{len(reference_data["camera_poses"])}'),
                            'description': f"Frame from {Path(meta_file).name}",
                            'carla_pose': frame.get('carla_camera_pose', frame.get('camera_pose_carla')),
                            'nurec_pose': frame.get('nurec_camera_pose', frame.get('camera_pose_dggt'))
                        })

                # Extract dynamic object poses
                if 'dynamic_objects' in metadata:
                    for obj in metadata['dynamic_objects']:
                        reference_data['dynamic_objects'].append({
                            'track_id': obj.get('track_id', obj.get('id')),
                            'type': obj.get('type', 'unknown'),
                            'description': obj.get('description', ''),
                            'carla_pose': obj.get('carla_pose'),
                            'nurec_pose': obj.get('nurec_pose', obj.get('dggt_pose'))
                        })

                # Handle single-frame metadata
                if 'carla_pose' in metadata and 'nurec_pose' in metadata:
                    reference_data['camera_poses'].append({
                        'id': Path(meta_file).stem,
                        'description': f"Single frame from {Path(meta_file).name}",
                        'carla_pose': metadata['carla_pose'],
                        'nurec_pose': metadata['nurec_pose']
                    })

            except Exception as e:
                print(f"Warning: Failed to parse {meta_file}: {e}")

    # Validate and clean data
    validate_reference_data(reference_data)

    # Save reference data
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'coordinate_transforms.json')
    with open(output_file, 'w') as f:
        json.dump(reference_data, f, indent=2)

    print(f"\nExtracted {len(reference_data['camera_poses'])} camera poses")
    print(f"Extracted {len(reference_data['dynamic_objects'])} dynamic object poses")
    print(f"Reference data saved to: {output_file}")

    return reference_data


def create_sample_reference_data(reference_data: dict):
    """
    Create sample reference data for testing when no NuRec output is available

    Args:
        reference_data: Reference data dict to populate
    """
    # Sample camera poses with known transformations
    reference_data['camera_poses'] = [
        {
            'id': 'sample_cam_001',
            'description': 'Forward-looking camera at origin offset',
            'carla_pose': [
                [ 1.0,  0.0,  0.0,  10.0],
                [ 0.0,  1.0,  0.0,   0.0],
                [ 0.0,  0.0,  1.0,   2.0],
                [ 0.0,  0.0,  0.0,   1.0]
            ],
            'nurec_pose': [
                [ 0.0,  1.0,  0.0,   0.0],
                [ 0.0,  0.0, -1.0,  -2.0],
                [ 1.0,  0.0,  0.0,  10.0],
                [ 0.0,  0.0,  0.0,   1.0]
            ]
        },
        {
            'id': 'sample_cam_002',
            'description': 'Camera yawed 90 degrees left',
            'carla_pose': [
                [ 0.0, -1.0,  0.0,   5.0],
                [ 1.0,  0.0,  0.0,  10.0],
                [ 0.0,  0.0,  1.0,   2.0],
                [ 0.0,  0.0,  0.0,   1.0]
            ],
            'nurec_pose': [
                [ 1.0,  0.0,  0.0,  10.0],
                [ 0.0,  0.0, -1.0,  -2.0],
                [ 0.0, -1.0,  0.0,   5.0],
                [ 0.0,  0.0,  0.0,   1.0]
            ]
        }
    ]

    # Sample dynamic object poses
    reference_data['dynamic_objects'] = [
        {
            'track_id': 'sample_vehicle_001',
            'type': 'vehicle',
            'description': 'Stationary vehicle',
            'carla_pose': [
                [ 1.0,  0.0,  0.0,  50.0],
                [ 0.0,  1.0,  0.0,   3.0],
                [ 0.0,  0.0,  1.0,   0.5],
                [ 0.0,  0.0,  0.0,   1.0]
            ],
            'nurec_pose': [
                [ 0.0,  1.0,  0.0,   3.0],
                [ 0.0,  0.0, -1.0,  -0.5],
                [ 1.0,  0.0,  0.0,  50.0],
                [ 0.0,  0.0,  0.0,   1.0]
            ]
        }
    ]


def validate_reference_data(reference_data: dict):
    """
    Validate and clean reference data

    Args:
        reference_data: Reference data dict to validate
    """
    # Remove invalid camera poses
    valid_cameras = []
    for pose in reference_data['camera_poses']:
        if pose.get('carla_pose') and pose.get('nurec_pose'):
            try:
                # Verify poses are valid 4x4 matrices
                carla = np.array(pose['carla_pose'])
                nurec = np.array(pose['nurec_pose'])
                if carla.shape == (4, 4) and nurec.shape == (4, 4):
                    valid_cameras.append(pose)
            except:
                pass
    reference_data['camera_poses'] = valid_cameras

    # Remove invalid dynamic objects
    valid_objects = []
    for obj in reference_data['dynamic_objects']:
        if obj.get('carla_pose') and obj.get('nurec_pose'):
            try:
                carla = np.array(obj['carla_pose'])
                nurec = np.array(obj['nurec_pose'])
                if carla.shape == (4, 4) and nurec.shape == (4, 4):
                    valid_objects.append(obj)
            except:
                pass
    reference_data['dynamic_objects'] = valid_objects


def main():
    parser = argparse.ArgumentParser(
        description='Extract NuRec coordinate transform reference data'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to NuRec output directory'
    )
    parser.add_argument(
        '--output', '-o',
        default='tests/test_data/nurec_reference/',
        help='Path to output directory for reference data'
    )

    args = parser.parse_args()
    extract_reference_data(args.input, args.output)


if __name__ == '__main__':
    main()