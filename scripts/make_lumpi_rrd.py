#!/usr/bin/env python
"""Build a rerun recording from LUMPI labels (+ point clouds if available).

Logs ground-truth boxes, class colors and trajectory trails for a time
window of a measurement; when --lidar-dir is given, frames NNNNNN.ply
(10 Hz) inside the window are logged as point clouds.

Run with rerun-sdk 0.18.x. One recording per process (the SDK deadlocks
on repeated in-process recordings).

Usage:
  python make_lumpi_rrd.py Label.csv --out teaser.rrd \
      --start 0 --end 3.1 --lidar-dir test_data/Measurement5/lidar
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from lidar_pilot.io.lumpi import LUMPI_CLASSES

FPS = 10
CLASS_COLORS = {
    'pedestrian': (245, 66, 66), 'car': (66, 135, 245),
    'bicycle': (66, 245, 96), 'motorcycle': (32, 196, 160),
    'bus': (245, 66, 221), 'truck': (245, 167, 66),
    'scooter': (66, 245, 230),
}


def load_rows(csv_path, t0, t1):
    cols = np.loadtxt(csv_path, delimiter=',', skiprows=1,
                      usecols=range(16), ndmin=2)
    m = (cols[:, 0] >= t0) & (cols[:, 0] <= t1)
    cols = cols[m]
    frames = defaultdict(list)
    for r in cols:
        frames[int(round(r[0] * FPS))].append(r)
    return frames


def read_ply_xyz(path):
    from plyfile import PlyData
    v = PlyData.read(str(path))['vertex']
    return np.column_stack([v['x'], v['y'], v['z']])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('label_csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, required=True)
    ap.add_argument('--lidar-dir', default=None)
    ap.add_argument('--point-stride', type=int, default=3)
    args = ap.parse_args()

    import rerun as rr
    rr.init('lumpi', spawn=False)
    rec = rr.new_recording('lumpi')
    rr.save(args.out, recording=rec)

    frames = load_rows(args.label_csv, args.start, args.end)
    lidar_dir = Path(args.lidar_dir) if args.lidar_dir else None
    trails: dict[int, list] = {}

    for fidx in sorted(frames):
        t = fidx / FPS
        rr.set_time_seconds('t', t, recording=rec)

        if lidar_dir is not None:
            ply = lidar_dir / f'{fidx:06d}.ply'
            if ply.exists():
                pts = read_ply_xyz(ply)[::args.point_stride]
                keep = np.linalg.norm(pts[:, :2], axis=1) < 120  # trim far noise
                rr.log('world/points',
                       rr.Points3D(pts[keep], radii=0.04, colors=(165, 165, 175)),
                       recording=rec)

        centers, half_sizes, quats, colors, labels = [], [], [], [], []
        for r in frames[fidx]:
            oid, cls = int(r[1]), LUMPI_CLASSES.get(int(r[7]), 'unknown')
            x, y, z = r[9], r[10], r[11]
            l, w, h, yaw = r[12], r[13], r[14], r[15]
            centers.append([x, y, z])
            half_sizes.append([l / 2, w / 2, h / 2])
            quats.append(rr.Quaternion(xyzw=[0, 0, np.sin(yaw / 2), np.cos(yaw / 2)]))
            colors.append(CLASS_COLORS.get(cls, (200, 200, 200)))
            labels.append(f'#{oid} {cls}')
            trails.setdefault(oid, []).append([x, y, z])
        if centers:
            rr.log('world/objects',
                   rr.Boxes3D(centers=centers, half_sizes=half_sizes,
                              rotations=quats, colors=colors, labels=labels),
                   recording=rec)
        strips = [np.asarray(v) for v in trails.values() if len(v) >= 2]
        if strips:
            rr.log('world/trails',
                   rr.LineStrips3D(strips, radii=0.05, colors=(255, 230, 60)),
                   recording=rec)
    print(f'wrote {args.out} ({len(frames)} frames, window {args.start}-{args.end}s)')


if __name__ == '__main__':
    main()
