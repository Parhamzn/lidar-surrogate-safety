#!/usr/bin/env python
"""Render a 3D-perspective MP4 clip from LUMPI labels + point clouds.

Cinematic variant of render_lumpi_mp4.py: an elevated oblique camera
framed on the conflict area, perspective projection, wireframe 3D boxes,
class-colored trails on the ground and the point cloud for context.

Usage:
  python render_lumpi_mp4_3d.py Label.csv --out clip3d.mp4 --start 45 --end 75 \
      --lidar-dir .../lidar --highlight 7,8 --center 12,13 --span 60 \
      --elev 32 --azim -50 [--still 558]   # --still renders one PNG frame
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from make_lumpi_rrd import CLASS_COLORS, FPS, load_rows, read_ply_xyz
from lidar_pilot.io.lumpi import LUMPI_CLASSES

TRAIL_SECONDS = 8.0
BG = '#101018'


def mpl_color(cls):
    r, g, b = CLASS_COLORS.get(cls, (200, 200, 200))
    return (r / 255, g / 255, b / 255)


def box_edges(cx, cy, zc, l, w, h, yaw):
    """12 wireframe segments of an oriented 3D box (center z convention)."""
    c, s = np.cos(yaw), np.sin(yaw)
    base = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]) / 2
    base = base @ np.array([[c, s], [-s, c]]) + [cx, cy]
    lo, hi = zc - h / 2, zc + h / 2
    bot = [(x, y, lo) for x, y in base]
    top = [(x, y, hi) for x, y in base]
    segs = []
    for k in range(4):
        segs.append([bot[k], bot[(k + 1) % 4]])
        segs.append([top[k], top[(k + 1) % 4]])
        segs.append([bot[k], top[k]])
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('label_csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, required=True)
    ap.add_argument('--lidar-dir', default=None)
    ap.add_argument('--point-stride', type=int, default=4)
    ap.add_argument('--highlight', default='')
    ap.add_argument('--center', required=True, help='cx,cy of the view box')
    ap.add_argument('--span', type=float, default=60.0, help='view width [m]')
    ap.add_argument('--elev', type=float, default=32.0)
    ap.add_argument('--azim', type=float, default=-50.0)
    ap.add_argument('--zmin', type=float, default=-3.0)
    ap.add_argument('--zmax', type=float, default=12.0)
    ap.add_argument('--zoom', type=float, default=1.25)
    ap.add_argument('--still', type=int, default=None,
                    help='render only this frame index as PNG (camera tuning)')
    args = ap.parse_args()

    import imageio.v2 as imageio

    cx, cy = (float(v) for v in args.center.split(','))
    half = args.span / 2
    frames = load_rows(args.label_csv, args.start, args.end)
    lidar_dir = Path(args.lidar_dir) if args.lidar_dir else None
    highlight = {int(v) for v in args.highlight.split(',') if v}

    trails: dict[int, deque] = defaultdict(
        lambda: deque(maxlen=int(TRAIL_SECONDS * FPS)))
    trail_cls: dict[int, str] = {}

    frame_ids = sorted(frames)
    writer = None
    if args.still is None:
        writer = imageio.get_writer(args.out, fps=FPS, codec='libx264',
                                    quality=8, macro_block_size=1)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(projection='3d')

    for n, fidx in enumerate(frame_ids):
        t = fidx / FPS
        # trails accumulate over all frames, drawing happens per frame
        for r in frames[fidx]:
            oid, cls = int(r[1]), LUMPI_CLASSES.get(int(r[7]), 'unknown')
            trails[oid].append((r[9], r[10], r[11] - r[14] / 2 + 0.15))
            trail_cls[oid] = cls
        if args.still is not None and fidx != args.still:
            continue

        ax.clear()
        ax.set_facecolor(BG)
        try:
            ax.set_proj_type('persp', focal_length=0.22)
        except TypeError:
            ax.set_proj_type('persp')

        if lidar_dir is not None:
            ply = lidar_dir / f'{fidx:06d}.ply'
            if ply.exists():
                pts = read_ply_xyz(ply)[::args.point_stride]
                m = ((np.abs(pts[:, 0] - cx) < half)
                     & (np.abs(pts[:, 1] - cy) < half)
                     & (pts[:, 2] > args.zmin) & (pts[:, 2] < args.zmax))
                ax.scatter(pts[m, 0], pts[m, 1], pts[m, 2], s=0.25,
                           c='#62626e', linewidths=0, depthshade=False)

        for oid, hist in trails.items():
            if len(hist) >= 2:
                arr = np.asarray(hist)
                ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], '-', lw=1.6,
                        color=mpl_color(trail_cls[oid]), alpha=0.75)

        plain_segs, plain_colors = [], []
        hl_rank = {oid: k for k, oid in enumerate(sorted(highlight))}
        for r in frames[fidx]:
            oid, cls = int(r[1]), LUMPI_CLASSES.get(int(r[7]), 'unknown')
            segs = box_edges(r[9], r[10], r[11], r[12], r[13], r[14], r[15])
            if oid in highlight:
                ax.add_collection3d(Line3DCollection(
                    segs, colors=[mpl_color(cls)], linewidths=2.8))
                # stagger label heights so co-located protagonists at the
                # conflict moment don't overprint each other
                dz = 1.2 + 1.6 * hl_rank[oid]
                ax.text(r[9], r[10], r[11] + r[14] / 2 + dz,
                        f'#{oid} {cls}', color='white', fontsize=11,
                        ha='center', fontweight='bold')
            else:
                plain_segs.extend(segs)
                plain_colors.extend([(*mpl_color(cls), 0.85)] * len(segs))
        if plain_segs:
            ax.add_collection3d(Line3DCollection(plain_segs,
                                                 colors=plain_colors,
                                                 linewidths=1.0))

        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(args.zmin, args.zmax)
        ax.set_box_aspect((1, 1, (args.zmax - args.zmin) / args.span),
                          zoom=args.zoom)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_axis_off()
        fig.text(0.04, 0.94, f't = {t:6.1f} s', color='white',
                 fontsize=14, family='monospace')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        if args.still is not None:
            out = args.out.replace('.mp4', f'_still{fidx}.png')
            fig.savefig(out, dpi=100, facecolor=BG)
            print(f'wrote {out}')
            return
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(frame)
        if n % 50 == 0:
            print(f'frame {n}/{len(frame_ids)}')

    writer.close()
    plt.close(fig)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
