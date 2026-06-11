#!/usr/bin/env python
"""Closing-the-loop dashboard: label-free pipeline vs ground truth.

One figure, three metric columns (TTC / PET / HBE). Per column:
  row 1  hotspot density from the human-labeled ground truth
  row 2  hotspot density from the label-free machine pipeline,
         on an identical color scale (honest visual comparison)
  row 3  per-cell event-count agreement scatter with count ratio and
         spatial correlation

Both sources are restricted to the detector's sensing envelope so the
comparison is like-for-like.

Usage: python scripts/plot_e2e_validation.py
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from lidar_pilot.viz import load_orthophoto

R_MAX = 45.0
CELL = 8.0
CMAPS = {'TTC': 'Reds', 'PET': 'Purples', 'HBE': 'Blues'}
ACCENT = {'TTC': '#c23a2f', 'PET': '#6a3fa8', 'HBE': '#2c6fb5'}
UNITS = {'TTC': 's', 'PET': 's', 'HBE': 'm/s²'}


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r['x']]
    return [r for r in rows
            if np.hypot(float(r['x']), float(r['y'])) <= R_MAX]


def alpha_cmap(name):
    colors = plt.get_cmap(name)(np.linspace(0, 1, 256))
    colors[:, 3] = np.linspace(0.30, 0.95, 256)
    return ListedColormap(colors)


def main():
    gt = load('outputs/lumpi/conflicts.csv')
    e2e = load('outputs/lumpi_e2e/conflicts.csv')
    ortho, ortho_ext = load_orthophoto()

    fig, axes = plt.subplots(3, 3, figsize=(15, 14.5),
                             gridspec_kw=dict(height_ratios=[1, 1, 0.78]))
    lim = R_MAX + 3
    headline = []

    for col, metric in enumerate(('TTC', 'PET', 'HBE')):
        evs = {}
        for tag, rows in (('gt', gt), ('e2e', e2e)):
            evs[tag] = np.array([(float(r['x']), float(r['y'])) for r in rows
                                 if r['metric'] == metric]).reshape(-1, 2)

        # shared color scale: first pass computes both maximima
        vmax = 1
        for tag in ('gt', 'e2e'):
            h, _, _ = np.histogram2d(evs[tag][:, 0], evs[tag][:, 1],
                                     bins=24, range=[[-lim, lim], [-lim, lim]])
            vmax = max(vmax, h.max())

        for row, tag, label in ((0, 'gt', 'human-labeled ground truth'),
                                (1, 'e2e', 'label-free pipeline')):
            ax = axes[row, col]
            if ortho is not None:
                ax.imshow(ortho, extent=ortho_ext, alpha=0.85, zorder=0)
            hb = ax.hexbin(evs[tag][:, 0], evs[tag][:, 1], gridsize=22,
                           cmap=alpha_cmap(CMAPS[metric]), mincnt=1,
                           vmin=0, vmax=vmax, zorder=2,
                           extent=(-lim, lim, -lim, lim))
            ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f'{metric} — {label} (n={len(evs[tag])})',
                         fontsize=11,
                         color=ACCENT[metric] if row == 0 else 'black')
            if col == 2:
                plt.colorbar(hb, ax=ax, shrink=0.85, pad=0.02,
                             label='events per cell')

        # per-cell agreement
        bins = np.arange(-lim, lim + CELL, CELL)
        maps = {}
        for tag in ('gt', 'e2e'):
            h, _, _ = np.histogram2d(evs[tag][:, 0], evs[tag][:, 1],
                                     bins=[bins, bins])
            maps[tag] = h.ravel()
        m = (maps['gt'] > 0) | (maps['e2e'] > 0)
        r = np.corrcoef(maps['gt'][m], maps['e2e'][m])[0, 1]
        ratio = len(evs['e2e']) / max(len(evs['gt']), 1)
        headline.append(f'{metric}: ratio {ratio:.2f}, r={r:.2f}')

        ax = axes[2, col]
        top = max(maps['gt'][m].max(), maps['e2e'][m].max()) * 1.12
        ax.plot([0, top], [0, top], '--', color='0.55', lw=1, zorder=1)
        ax.scatter(maps['gt'][m], maps['e2e'][m], s=34, alpha=0.55,
                   color=ACCENT[metric], edgecolors='white', linewidths=0.4,
                   zorder=2)
        ax.set_xlim(0, top), ax.set_ylim(0, top)
        ax.set_aspect('equal')
        ax.grid(alpha=0.25)
        ax.set_xlabel('events per cell — ground truth', fontsize=9)
        if col == 0:
            ax.set_ylabel('events per cell — pipeline', fontsize=9)
        ax.text(0.04, 0.96,
                f'count ratio {ratio:.2f}\nspatial r = {r:.2f}',
                transform=ax.transAxes, va='top', fontsize=11,
                fontweight='bold', color=ACCENT[metric],
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

    fig.suptitle('Closing the loop: label-free LiDAR pipeline vs '
                 'human-labeled ground truth\n'
                 f'Measurement5, within {R_MAX:.0f} m sensing envelope — '
                 + '   ·   '.join(headline),
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = 'figures/closing_the_loop_Measurement5.png'
    fig.savefig(out, dpi=200)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
