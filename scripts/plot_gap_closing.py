#!/usr/bin/env python
"""Gap-closing figure: in-domain fine-tuning on TUMTraf Intersection.

AP@0.1 (the TUMTraf benchmark metric) on the held-out subset s04 for three
checkpoints — nuScenes-pretrained, LUMPI-fine-tuned (zero-shot from
Hanover), and LUMPI+TUMTraf-fine-tuned (trained on s01-s03). s04 contains
only vehicle classes (car/truck/bus; no VRUs), so this measures how much a
little in-domain data closes the cross-site gap on vehicles. The dataset's
published PointPillars baselines are drawn as reference lines (note: those
are 6-class mAP on the official split; ours is 3-class vehicle mAP on s04,
so the comparison is indicative, not apples-to-apples).

Usage: python scripts/plot_gap_closing.py
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

CLASSES = ['car', 'truck', 'bus', 'mAP']
TAGS = [('pretrained', 'nuScenes-pretrained', '#c2c2c2'),
        ('lumpi', 'LUMPI zero-shot (Hanover only)', '#7badd6'),
        ('tumtraf', 'LUMPI+TUMTraf-fine-tuned (s01-s03)', '#1f5c99')]
# their published PointPillars (mAP@0.1, 6 classes, official split)
THEIR_SINGLE, THEIR_FUSION = 46.93, 55.21


def ap_by_tag(path):
    out = {}
    for r in csv.DictReader(open(path)):
        try:
            out.setdefault(r['tag'], {})[r['class']] = float(r['ap@0.1'])
        except ValueError:
            out.setdefault(r['tag'], {})[r['class']] = np.nan
    return out


def main():
    ap = ap_by_tag('outputs/tumtraf_s04/ap_eval.csv')
    x = np.arange(len(CLASSES))
    w = 0.26
    fig, axx = plt.subplots(figsize=(10, 5.5), layout='constrained')

    for k, (tag, label, color) in enumerate(TAGS):
        vals = [ap.get(tag, {}).get(c, np.nan) for c in CLASSES]
        bars = axx.bar(x + (k - 1) * w, vals, w, label=label, color=color,
                       edgecolor='white')
        for xi, v in zip(x + (k - 1) * w, vals):
            if not np.isnan(v):
                axx.text(xi, v + 0.8, f'{v:.0f}', ha='center', fontsize=7.5,
                         color=color if color != '#c2c2c2' else '#666')

    axx.axhline(THEIR_SINGLE, ls='--', color='#c23a2f', lw=1.3,
                label=f'their PointPillars, single-LiDAR ({THEIR_SINGLE:.0f}, 6-cls)')
    axx.axhline(THEIR_FUSION, ls=':', color='#c23a2f', lw=1.3,
                label=f'their PointPillars, early-fusion ({THEIR_FUSION:.0f}, 6-cls)')

    axx.set_xticks(x)
    axx.set_xticklabels(['Car', 'Truck', 'Bus', 'mAP'])
    axx.set_ylabel('AP@0.1')
    axx.set_ylim(0, 95)
    axx.set_title('Closing the cross-site gap with in-domain data — TUMTraf Intersection, held-out s04\n'
                  'AP@0.1 (vehicle classes; s04 has no VRUs). Reference lines are their 6-class benchmark.',
                  fontsize=10.5)
    axx.legend(loc='upper left', fontsize=8.3, framealpha=0.95)
    axx.grid(axis='y', alpha=0.25)

    out = 'figures/gap_closing_tumtraf.png'
    fig.savefig(out, dpi=200)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
