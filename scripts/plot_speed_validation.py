#!/usr/bin/env python
"""Speed-validation figure: pipeline speeds vs nuScenes GT velocities.

Reads speeds.csv produced by run_nuscenes_pipeline.py and writes
figures/speed_validation.png plus a per-class stats table to stdout.

Usage: python scripts/plot_speed_validation.py outputs/nuscenes_mini/speeds.csv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lidar_pilot.viz import CLASS_COLORS


def main(csv_path: str):
    rows = list(csv.DictReader(open(csv_path)))
    gt = np.array([float(r['speed_gt']) for r in rows])
    kf = np.array([float(r['speed_kf']) for r in rows])
    head = np.array([float(r['speed_head']) if r['speed_head'] else np.nan
                     for r in rows])
    cls = np.array([r['class'] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    lim = max(gt.max(), kf.max(), np.nanmax(head)) * 1.05
    for ax, est, name in ((axes[0], kf, 'Kalman-smoothed track speed'),
                          (axes[1], head, 'CenterPoint velocity head')):
        ok = ~np.isnan(est)
        for c in sorted(set(cls)):
            m = ok & (cls == c)
            if m.any():
                ax.scatter(gt[m], est[m], s=10, alpha=0.45,
                           color=CLASS_COLORS.get(c, '#999999'), label=c)
        ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.6)
        err = est[ok] - gt[ok]
        moving = ok & (gt > 1.0)
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae_mov = float(np.mean(np.abs(est[moving] - gt[moving]))) if moving.any() else np.nan
        ax.set_title(f'{name}\nRMSE {rmse:.2f} m/s | MAE (moving) {mae_mov:.2f} m/s')
        ax.set_xlabel('ground-truth speed [m/s]')
        ax.set_xlim(0, lim), ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('estimated speed [m/s]')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle(f'Speed validation vs nuScenes GT  (n={len(rows)} matched '
                 f'track-annotation pairs, {len(set(r["scene"] for r in rows))} scenes)')
    fig.tight_layout()

    out = Path('figures'); out.mkdir(exist_ok=True)
    fig.savefig(out / 'speed_validation.png', dpi=200)
    print(f'wrote {out / "speed_validation.png"}')

    # per-class table
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r['class']].append((float(r['speed_kf']), float(r['speed_gt'])))
    print(f'\n{"class":<22}{"n":>6}{"KF RMSE":>10}{"KF MAE>1m/s":>14}')
    for c, pairs in sorted(by_cls.items()):
        e = np.array([k - g for k, g in pairs])
        mov = np.array([abs(k - g) for k, g in pairs if g > 1.0])
        print(f'{c:<22}{len(pairs):>6}{np.sqrt(np.mean(e**2)):>10.2f}'
              f'{(np.mean(mov) if mov.size else np.nan):>14.2f}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'outputs/nuscenes_mini/speeds.csv')
