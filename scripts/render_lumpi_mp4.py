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

from make_lumpi_rrd import CLASS_COLORS, FPS, load_rows, read_ply_xyz
from lidar_pilot.io.lumpi import LUMPI_CLASSES

TRAIL_SECONDS = 8.0


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

    trails: dict[int, deque] = defaultdict(
        lambda: deque(maxlen=int(TRAIL_SECONDS * FPS)))
    trail_cls: dict[int, str] = {}

    writer = imageio.get_writer(args.out, fps=FPS, codec='libx264',
                                quality=8, macro_block_size=1)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

    for n, fidx in enumerate(sorted(frames)):
        t = fidx / FPS
        ax.clear()
        ax.set_facecolor('#101018')

        if lidar_dir is not None:
            ply = lidar_dir / f'{fidx:06d}.ply'
            if ply.exists():
                pts = read_ply_xyz(ply)[::args.point_stride]
                m = ((np.abs(pts[:, 0] - mid[0]) < side / 2 + 5)
                     & (np.abs(pts[:, 1] - mid[1]) < side / 2 + 5))
                ax.scatter(pts[m, 0], pts[m, 1], s=0.25, c='#5a5a66',
                           linewidths=0, rasterized=True)

        for r in frames[fidx]:
            oid, cls = int(r[1]), LUMPI_CLASSES.get(int(r[7]), 'unknown')
            trails[oid].append((r[9], r[10]))
            trail_cls[oid] = cls

        for oid, pts_hist in trails.items():
            if len(pts_hist) >= 2:
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
                color='white', fontsize=13, va='top', family='monospace')
        handles = [plt.Line2D([], [], color=mpl_color(c), lw=3, label=c)
                   for c in ('car', 'truck', 'bus', 'pedestrian', 'bicycle',
                             'motorcycle', 'scooter')]
        ax.legend(handles=handles, loc='lower right', fontsize=8,
                  framealpha=0.25, labelcolor='white')

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
