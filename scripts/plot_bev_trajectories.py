#!/usr/bin/env python
"""Bird's-eye-view trajectory maps from saved tracks.

Reads tracks_<scene>.pkl files (list[Trajectory], global frame) and writes
one BEV figure per scene to figures/.

Usage: python scripts/plot_bev_trajectories.py outputs/nuscenes_mini
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from lidar_pilot.viz import CLASS_COLORS


def plot_scene(pkl: Path, out_dir: Path):
    trajs = pickle.load(open(pkl, 'rb'))
    scene = pkl.stem.replace('tracks_', '')
    fig, ax = plt.subplots(figsize=(8, 8))
    seen = set()
    for tr in trajs:
        if len(tr) < 3:
            continue
        c = CLASS_COLORS.get(tr.label, '#aaaaaa')
        ax.plot(tr.xy[:, 0], tr.xy[:, 1], '-', color=c, lw=1.3, alpha=0.8,
                label=tr.label if tr.label not in seen else None)
        ax.plot(tr.xy[0, 0], tr.xy[0, 1], 'o', color=c, ms=3.5)
        seen.add(tr.label)
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.set_xlabel('global x [m]'), ax.set_ylabel('global y [m]')
    n = sum(1 for tr in trajs if len(tr) >= 3)
    ax.set_title(f'{scene}: {n} confirmed tracks (dots = track birth)')
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(out_dir / f'bev_{scene}.png', dpi=200)
    plt.close(fig)
    print(f'wrote figures/bev_{scene}.png ({n} tracks)')


def main(outputs_dir: str):
    out = Path('figures'); out.mkdir(exist_ok=True)
    for pkl in sorted(Path(outputs_dir).glob('tracks_*.pkl')):
        plot_scene(pkl, out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'outputs/nuscenes_mini')
