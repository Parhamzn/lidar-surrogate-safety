#!/usr/bin/env python
"""Conflict analysis figures: intersection conflict map + severity histograms.

Reads conflicts.csv and tracks_*.pkl from an outputs directory and writes
figures/conflict_map_<name>.png and figures/conflict_severity_<name>.png.

Usage: python scripts/plot_conflict_analysis.py outputs/lumpi
"""

from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VRU = {'pedestrian', 'bicycle', 'motorcycle', 'scooter'}


def pair_type(ca: str, cb: str) -> str:
    a_vru, b_vru = ca in VRU, cb in VRU
    if a_vru and b_vru:
        return 'VRU-VRU'
    if a_vru or b_vru:
        return 'vehicle-VRU'
    return 'vehicle-vehicle'


def main(outputs_dir: str):
    out_dir = Path(outputs_dir)
    rows = list(csv.DictReader(open(out_dir / 'conflicts.csv')))
    # Pedestrian-pedestrian proximity is crowd dynamics, not a traffic
    # conflict: the conflict literature requires at least one vehicle or
    # rider in the pair. (Cyclist/scooter vs pedestrian stays in.)
    rows = [r for r in rows
            if not (r['class_a'] == 'pedestrian' and r['class_b'] == 'pedestrian')]
    fig_dir = Path('figures'); fig_dir.mkdir(exist_ok=True)

    for pkl in sorted(out_dir.glob('tracks_*.pkl')):
        name = pkl.stem.replace('tracks_', '')
        scene_rows = [r for r in rows if r['scene'] == name]
        trajs = pickle.load(open(pkl, 'rb'))

        # ---- conflict map: serious conflicts only ----
        # TTC below 1.5 s is the conventional serious-conflict cutoff; the
        # same threshold is used for PET to keep the map readable. The full
        # 0-3 s range stays in the histograms.
        SERIOUS = 1.5
        fig, ax = plt.subplots(figsize=(9, 9))
        for tr in trajs:
            if len(tr) >= 5:
                ax.plot(tr.xy[:, 0], tr.xy[:, 1], '-', color='0.82', lw=0.5,
                        alpha=0.5, zorder=1)
        markers = {'TTC': ('o', 'Reds_r'), 'PET': ('s', 'Purples_r'),
                   'HBE': ('^', 'Blues_r')}
        for metric, (marker, cmap) in markers.items():
            ev = [r for r in scene_rows if r['metric'] == metric and r['x']
                  and (metric == 'HBE' or float(r['value']) <= SERIOUS)]
            if not ev:
                continue
            x = np.array([float(r['x']) for r in ev])
            y = np.array([float(r['y']) for r in ev])
            v = np.array([abs(float(r['value'])) for r in ev])
            sc = ax.scatter(x, y, c=v, cmap=cmap, marker=marker, s=46,
                            edgecolors='k', linewidths=0.4, zorder=3,
                            label=f'{metric} (n={len(ev)})')
            plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.01,
                         label=f'{metric} severity '
                               f'[{"m/s²" if metric == "HBE" else "s"}]')
        ax.set_aspect('equal'); ax.grid(alpha=0.2)
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_title(f'{name}: conflict map (grey = all trajectories)')
        ax.legend(loc='upper left', fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / f'conflict_map_{name}.png', dpi=200)
        plt.close(fig)

        # ---- severity histograms by interaction type ----
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, metric, title in ((axes[0], 'TTC', 'min TTC'),
                                  (axes[1], 'PET', 'PET')):
            ev = [r for r in scene_rows if r['metric'] == metric]
            if not ev:
                ax.set_axis_off()
                continue
            groups = {}
            for r in ev:
                groups.setdefault(pair_type(r['class_a'], r['class_b']),
                                  []).append(float(r['value']))
            bins = np.arange(0, 3.25, 0.25)
            bottom = np.zeros(len(bins) - 1)
            for g in ('vehicle-vehicle', 'vehicle-VRU', 'VRU-VRU'):
                if g not in groups:
                    continue
                h, _ = np.histogram(groups[g], bins=bins)
                ax.bar(bins[:-1], h, width=0.23, bottom=bottom, align='edge',
                       label=f'{g} (n={len(groups[g])})')
                bottom += h
            ax.set_xlabel(f'{title} [s]'); ax.set_ylabel('conflicts')
            ax.set_title(f'{title} distribution'); ax.legend(fontsize=8)
            ax.grid(alpha=0.25, axis='y')
        fig.suptitle(f'{name}: surrogate conflict severity by interaction type')
        fig.tight_layout()
        fig.savefig(fig_dir / f'conflict_severity_{name}.png', dpi=200)
        plt.close(fig)
        print(f'wrote conflict_map_{name}.png and conflict_severity_{name}.png')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'outputs/lumpi')
