#!/usr/bin/env python
"""Camera+LiDAR fusion view: project the south LiDAR and its 3D boxes onto
the synchronised camera image (TUMTraf Intersection).

Demonstrates the sensor-fusion geometry the posting calls for: the LiDAR
detector runs in 3D, and its output is registered into the camera frame
via the dataset's calibration (io/tumtraf.camera_projection_matrix). Points
are coloured by range; boxes are drawn as projected 3D wireframes.

Boxes come from the ground-truth labels by default (--boxes gt) or from a
detector-output pickle (--boxes <pkl>) holding a list of
[x, y, z_centre, l, w, h, yaw, label] per frame keyed by stem.

Usage:
  python scripts/project_lidar_to_camera.py \
      --image sample_img.jpg --pcd sample.pcd --label sample_label.json \
      --camera s110_camera_basler_south1_8mm --out figures/fusion.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from lidar_pilot.io.tumtraf import (camera_projection_matrix,
                                    project_lidar_points, read_openlabel_boxes,
                                    read_pcd)
from lidar_pilot.viz import CLASS_COLORS

WIDTH, HEIGHT = 1920, 1200
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),          # bottom
         (4, 5), (5, 6), (6, 7), (7, 4),          # top
         (0, 4), (1, 5), (2, 6), (3, 7)]          # verticals


def box_corners(xyz, lwh, yaw):
    """8 corners of a 3D box in the LiDAR frame (centre xyz, dims l,w,h)."""
    l, w, h = lwh
    # corner order matches EDGES: bottom ring (0-3), top ring (4-7)
    c = np.array([[ l / 2,  w / 2, -h / 2], [ l / 2, -w / 2, -h / 2],
                  [-l / 2, -w / 2, -h / 2], [-l / 2,  w / 2, -h / 2],
                  [ l / 2,  w / 2,  h / 2], [ l / 2, -w / 2,  h / 2],
                  [-l / 2, -w / 2,  h / 2], [-l / 2,  w / 2,  h / 2]])
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw),  np.cos(yaw), 0],
                    [0, 0, 1]])
    return (rot @ c.T).T + xyz


def draw_box(ax, P, xyz, lwh, yaw, color):
    corners = box_corners(np.asarray(xyz), np.asarray(lwh), float(yaw))
    homog = np.column_stack([corners, np.ones(8)])
    cam = (P @ homog.T).T
    if np.any(cam[:, 2] <= 0.5):           # any corner behind camera -> skip
        return False
    uv = cam[:, :2] / cam[:, 2:3]
    if uv[:, 0].max() < 0 or uv[:, 0].min() > WIDTH or \
            uv[:, 1].max() < 0 or uv[:, 1].min() > HEIGHT:
        return False
    for a, b in EDGES:
        ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]],
                color=color, lw=1.6, alpha=0.9)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--pcd', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--camera', default='s110_camera_basler_south1_8mm')
    ap.add_argument('--boxes', default='gt',
                    help="'gt' or a detector-output .pkl keyed by frame stem")
    ap.add_argument('--stem', default=None, help='frame stem for the pkl')
    ap.add_argument('--out', default='figures/fusion_tumtraf.png')
    args = ap.parse_args()

    P = camera_projection_matrix(args.label, args.camera)
    img = plt.imread(args.image)

    pts = read_pcd(args.pcd)[:, :3]
    uv, depth, _ = project_lidar_points(P, pts)
    vis = (uv[:, 0] >= 0) & (uv[:, 0] < WIDTH) & (uv[:, 1] >= 0) & (uv[:, 1] < HEIGHT)
    uv, depth = uv[vis], depth[vis]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(img)
    sc = ax.scatter(uv[:, 0], uv[:, 1], c=np.clip(depth, 0, 80), s=2.2,
                    cmap='turbo_r', alpha=0.55)

    if args.boxes == 'gt':
        boxes = [(b['xyz'], b['lwh'], b['yaw'], b['label'])
                 for b in read_openlabel_boxes(args.label)]
        src = 'ground-truth 3D boxes'
    else:
        stem = args.stem or 'frame'
        det = json.load(open(args.boxes)) if args.boxes.endswith('.json') \
            else __import__('pickle').load(open(args.boxes, 'rb'))
        boxes = [((r[0], r[1], r[2]), (r[3], r[4], r[5]), r[6], r[7])
                 for r in det[stem]]
        src = 'LiDAR detector 3D boxes'

    drawn = 0
    for xyz, lwh, yaw, label in boxes:
        if draw_box(ax, P, xyz, lwh, yaw, CLASS_COLORS.get(label, '#ffffff')):
            drawn += 1

    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis('off')
    ax.set_title(f'Camera+LiDAR fusion (TUMTraf Intersection): south LiDAR points '
                 f'(range-coloured) + {src} ({drawn} shown) projected into '
                 f'{args.camera}', fontsize=11)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label('LiDAR range (m)')
    fig.savefig(args.out, dpi=170, bbox_inches='tight')
    print(f'wrote {args.out} ({drawn} boxes, {vis.sum()} points in view)')


if __name__ == '__main__':
    main()
