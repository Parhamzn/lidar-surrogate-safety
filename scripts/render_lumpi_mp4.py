#!/usr/bin/env python
"""Render an MP4 clip (bird's-eye view) from LUMPI labels + point clouds.

Produces presentation-ready video: grey point cloud, oriented boxes and
fading trails colored by class, optional highlighted track IDs. Uses
matplotlib (Agg) + imageio-ffmpeg; no display needed.

Usage:
  python render_lumpi_mp4.py Label.csv --out clip.mp4 --start 270 --end 300 \
      --lidar-dir .../Measurement5/lidar --highlight 977,1387
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from make_lumpi_rrd import (CLASS_COLORS, FPS, crop_object_points, load_rows,
                            read_ply_xyz)
from lidar_pilot.io.lumpi import LUMPI_CLASSES

TRAIL_SECONDS = 8.0


def rasterize_ref_cloud(ply_path, georef_path, mid, side, res=0.15,
                        z_max=0.5):
    """Greyscale base map from the static survey scan (reflectance mean
    per cell). Cropped to near-ground height so canopy and roofs don't
    smear over the roads in top-down view."""
    import json
    from plyfile import PlyData
    g = json.load(open(georef_path))
    off = np.array([g['t'][0] + g['utm_offset'][0],
                    g['t'][1] + g['utm_offset'][1], g['z_offset']])
    v = PlyData.read(str(ply_path))['vertex']
    pts = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float64) - off
    refl = np.asarray(v['reflectance'], float)
    half = side / 2
    m = ((np.abs(pts[:, 0] - mid[0]) < half)
         & (np.abs(pts[:, 1] - mid[1]) < half) & (pts[:, 2] < z_max))
    pts, refl = pts[m], refl[m]
    n = max(int(side / res), 64)
    rng = [[mid[0] - half, mid[0] + half], [mid[1] - half, mid[1] + half]]
    sums, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=n, range=rng,
                                  weights=refl)
    counts, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=n, range=rng)
    mean = np.divide(sums, counts, out=np.zeros_like(sums),
                     where=counts > 0)
    nz = mean[counts > 0]
    lo, hi = np.percentile(nz, [2, 98]) if nz.size else (0, 1)
    grey = np.clip((mean - lo) / max(hi - lo, 1e-9), 0, 1)
    img = np.full((n, n, 3), (16 / 255, 16 / 255, 24 / 255))
    # deliberately dim: the base map is context, the moving objects and
    # overlays must stay the brightest things in frame
    g_vals = 0.10 + 0.33 * grey[counts > 0]
    img[counts > 0] = np.stack([g_vals] * 3, axis=1)
    data_bounds = (pts[:, 0].min(), pts[:, 0].max(),
                   pts[:, 1].min(), pts[:, 1].max()) if len(pts) else None
    # histogram2d is [xbin, ybin]; imshow(origin='lower') wants [row=y, col=x]
    return (np.transpose(img, (1, 0, 2)),
            (rng[0][0], rng[0][1], rng[1][0], rng[1][1]), data_bounds)


def box_corners(cx, cy, l, w, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]) / 2
    return local @ np.array([[c, s], [-s, c]]) + [cx, cy]


def mpl_color(cls):
    r, g, b = CLASS_COLORS.get(cls, (200, 200, 200))
    return (r / 255, g / 255, b / 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('label_csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, required=True)
    ap.add_argument('--lidar-dir', default=None)
    ap.add_argument('--point-stride', type=int, default=5)
    ap.add_argument('--ref-cloud', default=None,
                    help='static survey scan: rasterized base map; live '
                         'points reduce to class-colored object returns')
    ap.add_argument('--georef', default=None)
    ap.add_argument('--highlight', default='',
                    help='comma-separated object ids to emphasize')
    ap.add_argument('--pad', type=float, default=12.0)
    args = ap.parse_args()

    import imageio.v2 as imageio

    frames = load_rows(args.label_csv, args.start, args.end)
    lidar_dir = Path(args.lidar_dir) if args.lidar_dir else None
    highlight = {int(x) for x in args.highlight.split(',') if x}

    # fixed view: bounds of every labeled object in the window
    all_xy = np.array([[r[9], r[10]] for rows in frames.values() for r in rows])
    lo, hi = all_xy.min(0) - args.pad, all_xy.max(0) + args.pad
    side = float(max(hi - lo))  # square view
    mid = (lo + hi) / 2

    ref_img, ref_ext = None, None
    if args.ref_cloud:
        ref_img, ref_ext, data_b = rasterize_ref_cloud(
            args.ref_cloud, args.georef, mid, side)
        if data_b is not None:
            # clamp the view to where the base map has data
            lo = np.maximum(lo, [data_b[0], data_b[2]])
            hi = np.minimum(hi, [data_b[1], data_b[3]])
            side = float(max(hi - lo))
            mid = (lo + hi) / 2
        print(f'rasterized reference scan: {ref_img.shape[0]}px base map, '
              f'view clamped to {side:.0f} m')

    trails: dict[int, deque] = defaultdict(
        lambda: deque(maxlen=int(TRAIL_SECONDS * FPS)))
    trail_cls: dict[int, str] = {}
    trail_last: dict[int, int] = {}

    def append_trail(oid, fidx, point):
        """Track-gap aware: a time gap or an impossible jump starts a new
        trail instead of drawing a long phantom chord."""
        d = trails[oid]
        if d and (fidx - trail_last.get(oid, fidx) > 5
                  or np.hypot(point[0] - d[-1][0], point[1] - d[-1][1]) > 3.0):
            d.clear()
        d.append(point)
        trail_last[oid] = fidx

    writer = imageio.get_writer(args.out, fps=FPS, codec='libx264',
                                quality=8, macro_block_size=1)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

    for n, fidx in enumerate(sorted(frames)):
        t = fidx / FPS
        ax.clear()
        ax.set_facecolor('#101018')

        if ref_img is not None:
            ax.imshow(ref_img, extent=ref_ext, origin='lower', zorder=0,
                      interpolation='bilinear')

        if lidar_dir is not None:
            ply = lidar_dir / f'{fidx:06d}.ply'
            if ply.exists():
                if ref_img is not None:
                    # frozen base map present: live cloud reduces to the
                    # moving objects' own returns, class-colored
                    obj_pts, obj_col = crop_object_points(
                        read_ply_xyz(ply), frames[fidx])
                    if len(obj_pts):
                        ax.scatter(obj_pts[:, 0], obj_pts[:, 1], s=1.6,
                                   c=obj_col / 255, linewidths=0,
                                   rasterized=True, zorder=2)
                else:
                    pts = read_ply_xyz(ply)[::args.point_stride]
                    m = ((np.abs(pts[:, 0] - mid[0]) < side / 2 + 5)
                         & (np.abs(pts[:, 1] - mid[1]) < side / 2 + 5))
                    ax.scatter(pts[m, 0], pts[m, 1], s=0.25, c='#5a5a66',
                               linewidths=0, rasterized=True)

        for r in frames[fidx]:
            oid, cls = int(r[1]), LUMPI_CLASSES.get(int(r[7]), 'unknown')
            append_trail(oid, fidx, (r[9], r[10]))
            trail_cls[oid] = cls

        for oid, pts_hist in trails.items():
            # only objects still present keep a visible trail
            if len(pts_hist) >= 2 and fidx - trail_last.get(oid, -99) <= 3:
                arr = np.asarray(pts_hist)
                ax.plot(arr[:, 0], arr[:, 1], '-', lw=1.4,
                        color=mpl_color(trail_cls[oid]), alpha=0.65)

        for r in frames[fidx]:
            oid, cls = int(r[1]), LUMPI_CLASSES.get(int(r[7]), 'unknown')
            corners = box_corners(r[9], r[10], r[12], r[13], r[15])
            emph = oid in highlight
            ax.add_patch(Polygon(corners, closed=True, fill=emph,
                                 facecolor=(*mpl_color(cls), 0.35) if emph else 'none',
                                 edgecolor=mpl_color(cls),
                                 lw=2.6 if emph else 1.2, zorder=5))
            if emph:
                ax.annotate(f'#{oid} {cls}', (r[9], r[10]),
                            xytext=(0, 14), textcoords='offset points',
                            color='white', fontsize=11, ha='center',
                            fontweight='bold', zorder=6)

        ax.set_xlim(mid[0] - side / 2, mid[0] + side / 2)
        ax.set_ylim(mid[1] - side / 2, mid[1] + side / 2)
        ax.set_aspect('equal')
        ax.set_xticks([]), ax.set_yticks([])
        ax.text(0.015, 0.975, f't = {t:6.1f} s', transform=ax.transAxes,
                color='white', fontsize=13, va='top', family='monospace',
                bbox=dict(facecolor='#101018', alpha=0.75, pad=4,
                          edgecolor='none'))
        handles = [plt.Line2D([], [], color=mpl_color(c), lw=3, label=c)
                   for c in ('car', 'truck', 'bus', 'pedestrian', 'bicycle',
                             'motorcycle', 'scooter')]
        ax.legend(handles=handles, loc='lower right', fontsize=8,
                  framealpha=0.85, facecolor='#101018', edgecolor='none',
                  labelcolor='white')

        fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(frame)
        if n % 50 == 0:
            print(f'frame {n}/{len(frames)}')

    writer.close()
    plt.close(fig)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
