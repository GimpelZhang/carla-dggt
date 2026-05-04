# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Coordinate Transform Visualization Report Generator

Generates visualizations comparing CARLA and DGGT coordinate systems,
demonstrating point and rotation transformations.

Usage: python generate_coordinate_report.py --output docs/coordinate_visualization/
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import os
import sys
import argparse

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dggt_server.coordinate_transform import CoordinateTransform


def visualize_coordinate_systems():
    """
    Visualize CARLA and DGGT coordinate systems side by side.

    Returns:
        matplotlib.figure.Figure: Figure containing coordinate system visualizations
    """
    fig = plt.figure(figsize=(14, 6))

    # CARLA coordinate system (left-handed)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('CARLA Coordinate System (Left-Handed)')

    # Coordinate axes: X=Forward, Y=Right, Z=Up
    ax1.quiver(0, 0, 0, 1, 0, 0, color='r', label='X (Forward)', arrow_length_ratio=0.15)
    ax1.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y (Right)', arrow_length_ratio=0.15)
    ax1.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z (Up)', arrow_length_ratio=0.15)

    # Add axis labels
    ax1.text(1.1, 0, 0, 'X', color='r', fontsize=12)
    ax1.text(0, 1.1, 0, 'Y', color='g', fontsize=12)
    ax1.text(0, 0, 1.1, 'Z', color='b', fontsize=12)

    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_zlim([-1.5, 1.5])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend(loc='upper left')

    # DGGT/OpenCV coordinate system (right-handed)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('DGGT/OpenCV Coordinate System (Right-Handed)')

    # Coordinate axes: X=Right, Y=Down, Z=Forward
    ax2.quiver(0, 0, 0, 1, 0, 0, color='r', label='X (Right)', arrow_length_ratio=0.15)
    ax2.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y (Down)', arrow_length_ratio=0.15)
    ax2.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z (Forward)', arrow_length_ratio=0.15)

    # Add axis labels
    ax2.text(1.1, 0, 0, 'X', color='r', fontsize=12)
    ax2.text(0, 1.1, 0, 'Y', color='g', fontsize=12)
    ax2.text(0, 0, 1.1, 'Z', color='b', fontsize=12)

    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend(loc='upper left')

    plt.tight_layout()
    return fig


def visualize_point_transform():
    """
    Visualize point transformation examples from CARLA to DGGT.

    Returns:
        matplotlib.figure.Figure: Figure containing point transformation visualizations
    """
    fig = plt.figure(figsize=(14, 6))

    # Example points in CARLA
    carla_points = np.array([
        [1, 0, 0],  # Forward (X+)
        [0, 1, 0],  # Right (Y+)
        [0, 0, 1],  # Up (Z+)
        [1, 1, 1],  # Diagonal
    ])

    labels = ['Forward', 'Right', 'Up', 'Diagonal']
    colors = ['red', 'green', 'blue', 'purple']

    # Points in CARLA frame
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Points in CARLA Frame')

    for i, (pt, label, color) in enumerate(zip(carla_points, labels, colors)):
        ax1.scatter(*pt, c=color, s=100, label=f'{label} [{pt[0]}, {pt[1]}, {pt[2]}]', marker='o')
        # Draw line from origin to point
        ax1.plot([0, pt[0]], [0, pt[1]], [0, pt[2]], c=color, alpha=0.5)

    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_zlim([-1.5, 1.5])
    ax1.legend(loc='upper left')
    ax1.set_xlabel('X (Forward)')
    ax1.set_ylabel('Y (Right)')
    ax1.set_zlabel('Z (Up)')

    # Points in DGGT frame (transformed)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Points in DGGT Frame (Transformed)')

    for i, (pt, label, color) in enumerate(zip(carla_points, labels, colors)):
        dggt_pt = CoordinateTransform.transform_point_carla_to_dggt(pt)
        dggt_pt_rounded = dggt_pt.round(2)
        ax2.scatter(*dggt_pt, c=color, s=100,
                    label=f'{label} -> [{dggt_pt_rounded[0]}, {dggt_pt_rounded[1]}, {dggt_pt_rounded[2]}]',
                    marker='s')
        # Draw line from origin to point
        ax2.plot([0, dggt_pt[0]], [0, dggt_pt[1]], [0, dggt_pt[2]], c=color, alpha=0.5)

    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    ax2.legend(loc='upper left')
    ax2.set_xlabel('X (Right)')
    ax2.set_ylabel('Y (Down)')
    ax2.set_zlabel('Z (Forward)')

    plt.tight_layout()
    return fig


def visualize_rotation_transform():
    """
    Visualize rotation transformation from CARLA to DGGT.

    Returns:
        matplotlib.figure.Figure: Figure containing rotation transformation visualizations
    """
    fig = plt.figure(figsize=(14, 6))

    # Test rotations: Z-axis rotations at different angles
    angles = [0, 45, 90, 180]
    carla_rotations = [R.from_euler('z', a, degrees=True).as_matrix() for a in angles]

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Rotations in CARLA Frame (Z-axis)')

    for i, (rot, angle) in enumerate(zip(carla_rotations, angles)):
        # Show X-axis direction after rotation
        x_axis = rot @ np.array([1, 0, 0])
        ax1.quiver(0, 0, 0, *x_axis, color=f'C{i}', label=f'Z-rot {angle} deg',
                   arrow_length_ratio=0.15, linewidth=2)

    # Add reference axes
    ax1.quiver(0, 0, 0, 1, 0, 0, color='r', alpha=0.3, arrow_length_ratio=0.15)
    ax1.quiver(0, 0, 0, 0, 1, 0, color='g', alpha=0.3, arrow_length_ratio=0.15)
    ax1.quiver(0, 0, 0, 0, 0, 1, color='b', alpha=0.3, arrow_length_ratio=0.15)

    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_zlim([-1.5, 1.5])
    ax1.legend(loc='upper left')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Transformed Rotations in DGGT Frame')

    for i, (rot, angle) in enumerate(zip(carla_rotations, angles)):
        dggt_rot = CoordinateTransform.transform_rotation_carla_to_dggt(rot)
        # Show X-axis direction after transformation in DGGT frame
        x_axis_dggt = dggt_rot @ np.array([1, 0, 0])
        ax2.quiver(0, 0, 0, *x_axis_dggt, color=f'C{i}', label=f'Z-rot {angle} deg',
                   arrow_length_ratio=0.15, linewidth=2)

    # Add reference axes
    ax2.quiver(0, 0, 0, 1, 0, 0, color='r', alpha=0.3, arrow_length_ratio=0.15)
    ax2.quiver(0, 0, 0, 0, 1, 0, color='g', alpha=0.3, arrow_length_ratio=0.15)
    ax2.quiver(0, 0, 0, 0, 0, 1, color='b', alpha=0.3, arrow_length_ratio=0.15)

    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    ax2.legend(loc='upper left')

    plt.tight_layout()
    return fig


def generate_report(output_dir: str):
    """
    Generate complete visualization report.

    Creates PNG and PDF visualizations plus a markdown report.

    Args:
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating coordinate visualization report in: {output_dir}")

    # Generate coordinate systems visualization
    fig1 = visualize_coordinate_systems()
    fig1.savefig(os.path.join(output_dir, "coordinate_systems.png"), dpi=150, bbox_inches='tight')
    fig1.savefig(os.path.join(output_dir, "coordinate_systems.pdf"), bbox_inches='tight')
    plt.close(fig1)
    print("  - coordinate_systems.png/pdf")

    # Generate point transform visualization
    fig2 = visualize_point_transform()
    fig2.savefig(os.path.join(output_dir, "point_transform.png"), dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, "point_transform.pdf"), bbox_inches='tight')
    plt.close(fig2)
    print("  - point_transform.png/pdf")

    # Generate rotation transform visualization
    fig3 = visualize_rotation_transform()
    fig3.savefig(os.path.join(output_dir, "rotation_transform.png"), dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, "rotation_transform.pdf"), bbox_inches='tight')
    plt.close(fig3)
    print("  - rotation_transform.png/pdf")

    # Generate Markdown report
    report_content = """# Coordinate Transform Visualization Report

## 1. Coordinate Systems

![Coordinate Systems](coordinate_systems.png)

**CARLA (Left-Handed):**
- X: Forward
- Y: Right
- Z: Up

**DGGT/OpenCV (Right-Handed):**
- X: Right
- Y: Down
- Z: Forward

## 2. Point Transform

![Point Transform](point_transform.png)

| CARLA Point | DGGT Point | Description |
|-------------|------------|-------------|
| [1, 0, 0] | [0, 0, 1] | Forward -> Forward in DGGT |
| [0, 1, 0] | [1, 0, 0] | Right -> Right in DGGT |
| [0, 0, 1] | [0, -1, 0] | Up -> Down in DGGT (= -Y) |
| [1, 1, 1] | [1, -1, 1] | Diagonal transformation |

## 3. Rotation Transform

![Rotation Transform](rotation_transform.png)

Shows Z-axis rotations at 0, 45, 90, and 180 degrees transformed from CARLA to DGGT.

## 4. Transform Matrix

The complete transform includes:
1. `undo_carla_coordinate_transform()`: Undo CARLA's special Euler angle convention
2. Axis rotation: Apply the CARLA->DGGT rotation matrix

```python
# CARLA -> DGGT Rotation Matrix
# X_dggt = Y_carla (Right), Y_dggt = -Z_carla (Down), Z_dggt = X_carla (Forward)
CARLA_TO_DGGT_ROTATION = [
    [ 0,  1,  0],
    [ 0,  0, -1],
    [ 1,  0,  0]
]

# DGGT -> CARLA Rotation Matrix
DGGT_TO_CARLA_ROTATION = [
    [ 0,  0,  1],
    [ 1,  0,  0],
    [ 0, -1,  0]
]
```

## 5. Transform Chain

Following NuRec pattern:

```python
dggt_pose = t_carla_dggt @ undo_carla_coordinate_transform(carla_pose)
```

Where `t_carla_dggt` is computed from OpenDRIVE georeference using `get_t_rig_enu_from_ecef()`.

## 6. Notes

- Rotation matrices have determinant -1 (reflection, not pure rotation)
- This is expected for left-handed -> right-handed coordinate system conversion
- Matrices are orthogonal: R @ R.T = I
- Matrices are inverse of each other: CARLA_TO_DGGT @ DGGT_TO_CARLA = I

---

*Generated by generate_coordinate_report.py*
"""

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, 'w') as f:
        f.write(report_content)
    print("  - report.md")

    print(f"\nReport generation complete. Files saved to: {output_dir}")


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate coordinate transform visualization report"
    )
    parser.add_argument(
        "--output",
        default="docs/coordinate_visualization",
        help="Output directory for visualization files"
    )
    args = parser.parse_args()

    generate_report(args.output)


if __name__ == "__main__":
    main()