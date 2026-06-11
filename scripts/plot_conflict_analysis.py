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

from lidar_pilot.viz import load_orthophoto

VRU = {'pedestrian', 'bicycle', 'motorcycle', 'scooter'}


def pair_type(ca: str, cb: str) -> str:
    a_vru, b_vru = ca in VRU, cb in VRU
    if a_vru and b_vru:
        return 'VRU-VRU'
    if a_vru or b_vru:
        return 'vehicle-VRU'
    return 'vehicle-vehicle'


def load_road_polylines(csv_path):
    """LUMPI Map/lumpi_polylines.csv: one polyline per row as alternating
    x,y values, already in the label coordinate frame."""
    lines = []
    for line in open(csv_path):
        vals = np.fromstring(line, sep=',')
        if vals.size >= 4:
            lines.append(vals.reshape(-1, 2))
    return lines


def main(outputs_dir: str, map_csv: str | None = None):
    out_dir = Path(outputs_dir)
    road = (load_road_polylines(map_csv)
            if map_csv and Path(map_csv).exists() else [])
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

        # ---- conflict map: one density panel per metric ----
        # Hotspot density communicates "where is risky" better than piles
        # of overlapping markers; only the extreme tail is drawn as
        # individual events. TTC/PET capped at the 1.5 s serious-conflict
        # threshold for inclusion; the full range stays in the histograms.
        SERIOUS = 1.5
        N_EXTREME = 25      # circle only the N most severe events per panel
        CMAPS = {'TTC': 'Reds', 'PET': 'Purples', 'HBE': 'Blues'}

        def alpha_cmap(name):
            """Sequential cmap with an alpha ramp: sparse cells stay
            see-through over the orthophoto, dense cells read solid."""
            from matplotlib.colors import ListedColormap
            colors = plt.get_cmap(name)(np.linspace(0, 1, 256))
            colors[:, 3] = np.linspace(0.25, 0.95, 256)
            return ListedColormap(colors)

        from matplotlib.collections import LineCollection
        traj_lines = [tr.xy for tr in trajs if len(tr) >= 5]
        all_pts = np.vstack(traj_lines)
        x_lim = (all_pts[:, 0].min() - 5, all_pts[:, 0].max() + 5)
        y_lim = (all_pts[:, 1].min() - 5, all_pts[:, 1].max() + 5)

        fig, axes_map = plt.subplots(1, 3, figsize=(16.5, 6.2),
                                     sharex=True, sharey=True)
        ortho, ortho_ext = load_orthophoto()
        for ax, metric in zip(axes_map, ('TTC', 'PET', 'HBE')):
            if ortho is not None:
                ax.imshow(ortho, extent=ortho_ext, alpha=0.75, zorder=0.5)
            else:
                ax.add_collection(LineCollection(traj_lines, colors='0.85',
                                                 linewidths=0.4, alpha=0.5,
                                                 zorder=1))
            if road:
                ax.add_collection(LineCollection(road, colors='0.3',
                                                 linewidths=0.7, alpha=0.7,
                                                 zorder=1.5))
            ev = [r for r in scene_rows if r['metric'] == metric and r['x']
                  and (metric == 'HBE' or float(r['value']) <= SERIOUS)]
            if ev:
                x = np.array([float(r['x']) for r in ev])
                y = np.array([float(r['y']) for r in ev])
                v = np.array([float(r['value']) for r in ev])
                hb = ax.hexbin(x, y, gridsize=34,
                               cmap=(alpha_cmap(CMAPS[metric])
                                     if ortho is not None else CMAPS[metric]),
                               mincnt=1, zorder=2,
                               extent=(*x_lim, *y_lim))
                plt.colorbar(hb, ax=ax, shrink=0.8, pad=0.015,
                             label='events per cell')
                # severity ranks: small is severe for TTC/PET, large
                # (negative) deceleration is severe for HBE
                order = np.argsort(v if metric != 'HBE' else -np.abs(v))
                top = order[:N_EXTREME]
                ax.scatter(x[top], y[top], s=48, marker='o', facecolor='none',
                           edgecolors='black', linewidths=1.2, zorder=3,
                           label=f'{len(top)} most severe')
                ax.legend(loc='upper left', fontsize=9)
            ax.set_title(f'{metric} (n={len(ev)})')
            ax.set_xlim(*x_lim), ax.set_ylim(*y_lim)
            ax.set_aspect('equal')
            ax.set_xlabel('x [m]')
        axes_map[0].set_ylabel('y [m]')
        fig.suptitle(f'{name}: conflict hotspots'
                     f'{" (orthophoto: LGLN DOP20, CC-BY)" if ortho is not None else ""}'
                     f'{", dark grey = road network" if road else ""}',
                     y=0.98)
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
    main(sys.argv[1] if len(sys.argv) > 1 else 'outputs/lumpi',
         sys.argv[2] if len(sys.argv) > 2 else 'data/lumpi/lumpi_polylines.csv')
