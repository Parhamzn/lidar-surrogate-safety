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
import json
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


def crop_object_points(pts, rows, margin=0.25):
    """Points inside the frame's labeled boxes, with per-point class color.

    Used over the static reference scan: the survey backdrop replaces the
    static scene, so only the moving objects' own returns are drawn.
    """
    out_pts, out_col = [], []
    for r in rows:
        cx, cy, cz = r[9], r[10], r[11]
        l, w, h, yaw = r[12], r[13], r[14], r[15]
        rad = np.hypot(l, w) / 2 + margin
        m0 = ((np.abs(pts[:, 0] - cx) < rad)
              & (np.abs(pts[:, 1] - cy) < rad))
        sub = pts[m0]
        if not sub.size:
            continue
        dx, dy = sub[:, 0] - cx, sub[:, 1] - cy
        c, s = np.cos(yaw), np.sin(yaw)
        bx, by = c * dx + s * dy, -s * dx + c * dy
        m = ((np.abs(bx) < l / 2 + margin) & (np.abs(by) < w / 2 + margin)
             & (np.abs(sub[:, 2] - cz) < h / 2 + margin))
        if m.any():
            cls = LUMPI_CLASSES.get(int(r[7]), 'unknown')
            out_pts.append(sub[m])
            out_col.append(np.tile(CLASS_COLORS.get(cls, (200, 200, 200)),
                                   (int(m.sum()), 1)))
    if not out_pts:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    return np.vstack(out_pts), np.vstack(out_col).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('label_csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, required=True)
    ap.add_argument('--lidar-dir', default=None)
    ap.add_argument('--point-stride', type=int, default=3)
    ap.add_argument('--ref-cloud', default=None,
                    help='UTM reference scan (ply) as a static backdrop')
    ap.add_argument('--georef', default=None,
                    help='georef.json (required with --ref-cloud)')
    ap.add_argument('--ref-stride', type=int, default=2)
    args = ap.parse_args()

    import rerun as rr
    rr.init('lumpi', spawn=False)
    rec = rr.new_recording('lumpi')
    rr.save(args.out, recording=rec)

    if args.ref_cloud:
        # Static survey-scan backdrop, shifted into the label frame: raw
        # UTM magnitudes (~5.8e6 m) would destroy float32 view precision.
        from plyfile import PlyData
        g = json.load(open(args.georef))
        off = np.array([g['t'][0] + g['utm_offset'][0],
                        g['t'][1] + g['utm_offset'][1],
                        g['z_offset']])
        v = PlyData.read(args.ref_cloud)['vertex']
        pts = (np.column_stack([v['x'], v['y'], v['z']]).astype(np.float64)
               - off)[::args.ref_stride].astype(np.float32)
        refl = np.asarray(v['reflectance'], float)[::args.ref_stride]
        lo, hi = np.percentile(refl, [2, 98])
        grey = (55 + 175 * np.clip((refl - lo) / (hi - lo), 0, 1)).astype(np.uint8)
        colors = np.stack([grey, grey, grey], axis=1)
        try:
            rr.log('world/refmap',
                   rr.Points3D(pts, radii=0.02, colors=colors),
                   static=True, recording=rec)
        except TypeError:
            rr.log('world/refmap',
                   rr.Points3D(pts, radii=0.02, colors=colors),
                   timeless=True, recording=rec)
        print(f'logged reference scan: {len(pts)} points')

    frames = load_rows(args.label_csv, args.start, args.end)
    lidar_dir = Path(args.lidar_dir) if args.lidar_dir else None
    trails: dict[int, list] = {}
    trail_colors: dict[int, tuple] = {}

    for fidx in sorted(frames):
        t = fidx / FPS
        rr.set_time_seconds('t', t, recording=rec)

        if lidar_dir is not None:
            ply = lidar_dir / f'{fidx:06d}.ply'
            if ply.exists():
                if args.ref_cloud:
                    # static backdrop present: draw only the moving
                    # objects' own returns, class-colored
                    pts = read_ply_xyz(ply)
                    obj_pts, obj_col = crop_object_points(pts, frames[fidx])
                    rr.log('world/object_points',
                           rr.Points3D(obj_pts, radii=0.05, colors=obj_col),
                           recording=rec)
                else:
                    pts = read_ply_xyz(ply)[::args.point_stride]
                    keep = np.linalg.norm(pts[:, :2], axis=1) < 120
                    rr.log('world/points',
                           rr.Points3D(pts[keep], radii=0.04,
                                       colors=(165, 165, 175)),
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
            trail_colors[oid] = CLASS_COLORS.get(cls, (200, 200, 200))
        if centers:
            rr.log('world/objects',
                   rr.Boxes3D(centers=centers, half_sizes=half_sizes,
                              rotations=quats, colors=colors, labels=labels),
                   recording=rec)
        strip_ids = [oid for oid, v in trails.items() if len(v) >= 2]
        if strip_ids:
            rr.log('world/trails',
                   rr.LineStrips3D([np.asarray(trails[i]) for i in strip_ids],
                                   radii=0.05,
                                   colors=[trail_colors[i] for i in strip_ids]),
                   recording=rec)
    print(f'wrote {args.out} ({len(frames)} frames, window {args.start}-{args.end}s)')


if __name__ == '__main__':
    main()
