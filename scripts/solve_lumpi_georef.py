#!/usr/bin/env python
"""Derive the LUMPI label-frame -> UTM transform.

The labels/point clouds live in a local frame; meta.json's sensor
extrinsics live in an offset-UTM frame (UTM32 minus a round offset,
verified against the absolute-UTM reference map point cloud). Bridge:

  1. every point in a fused frame carries its sensor id and raw range,
     so each sensor's origin in the LABEL frame is a least-squares
     sphere-fit:  || p - o || = distance
  2. a rigid 2D Procrustes fit of those origins onto the extrinsic
     translations gives label -> offset-UTM (residuals validate it)
  3. + the round offset -> absolute UTM32 (EPSG:25832)

Writes georef.json with the rotation, translation, UTM offset and fit
residuals.

Usage (on the GPU box):
  python solve_lumpi_georef.py --ply data/lumpi/Measurement5/lidar/000100.ply \
      --meta data/lumpi/meta.json --sessions 49,50,51,52,53 \
      --out data/lumpi/georef.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from plyfile import PlyData
from scipy.optimize import least_squares


def sensor_origin(points, dists):
    """Least-squares sensor origin and range unit from || p - o || = s * d.

    The PLY 'distance' field is in raw sensor ticks whose size differs per
    LiDAR model (2 mm Velodyne, 4 mm Hesai, ...), so the scale s is fitted
    jointly with the origin instead of assumed.
    """
    def residual(params):
        o, s = params[:3], params[3]
        return np.linalg.norm(points - o, axis=1) - s * dists
    s0 = np.median(np.linalg.norm(points, axis=1)) / np.median(dists)
    p0 = np.array([points[:, 0].mean(), points[:, 1].mean(), 3.0, s0])
    sol = least_squares(residual, p0, method='lm')
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    return sol.x[:3], float(sol.x[3]), rms


def rigid_fit_2d(src, dst):
    """Procrustes: rotation R (2x2) + translation t with dst = R @ src + t."""
    src_c, dst_c = src - src.mean(0), dst - dst.mean(0)
    u, _, vt = np.linalg.svd(src_c.T @ dst_c)
    d = np.sign(np.linalg.det(u @ vt))
    R = (u @ np.diag([1, d]) @ vt).T
    t = dst.mean(0) - R @ src.mean(0)
    res = dst - (src @ R.T + t)
    return R, t, np.linalg.norm(res, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', required=True)
    ap.add_argument('--meta', required=True)
    ap.add_argument('--sessions', default='49,50,51,52,53')
    ap.add_argument('--utm-offset', default='548800,5803300')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-points', type=int, default=30000)
    args = ap.parse_args()

    v = PlyData.read(args.ply)['vertex']
    xyz = np.column_stack([v['x'], v['y'], v['z']]).astype(float)
    ids = np.asarray(v['id'], int)
    dist = np.asarray(v['distance'], float)

    meta = json.load(open(args.meta))
    sessions = [int(s) for s in args.sessions.split(',')]

    local, target = [], []
    print(f'{"session":>8}{"origin (label frame)":>34}'
          f'{"unit [mm]":>11}{"rms [m]":>9}')
    for sid in sessions:
        m = ids == sid
        idx = np.where(m & (dist > 100))[0]
        if idx.size < 100:
            print(f'{sid:>8}  skipped (too few points)')
            continue
        sel = np.random.default_rng(0).choice(
            idx, min(args.max_points, idx.size), replace=False)
        origin, scale, rms = sensor_origin(xyz[sel], dist[sel])
        ex = np.asarray(meta['session'][str(sid)]['extrinsic'], float)
        local.append(origin)
        target.append(ex[:3, 3])
        print(f'{sid:>8}  [{origin[0]:8.2f} {origin[1]:8.2f} {origin[2]:6.2f}]'
              f'{1000 * scale:>11.2f}{rms:>9.3f}')

    local, target = np.asarray(local), np.asarray(target)
    R, t, res = rigid_fit_2d(local[:, :2], target[:, :2])
    z_off = float(np.mean(target[:, 2] - local[:, 2]))
    ang = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    print(f'\nrigid fit: rotation {ang:+.3f} deg, translation {t.round(3).tolist()}')
    print(f'per-sensor 2D residuals [m]: {res.round(3).tolist()}')
    print(f'z offset: {z_off:+.2f} m')

    ux, uy = (float(s) for s in args.utm_offset.split(','))
    out = dict(R=R.tolist(), t=t.tolist(), utm_offset=[ux, uy],
               z_offset=z_off, rotation_deg=ang,
               residuals_m=res.tolist(), epsg=25832,
               note='UTM = R @ xy_label + t + utm_offset')
    json.dump(out, open(args.out, 'w'), indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
