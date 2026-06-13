#!/usr/bin/env python
"""VRU-inclusive gap-closing figure on the time-split held-out val.

The first gap-closing figure used the s04 subset, which has no vulnerable
road users. This one uses the temporal-split val (last 25% of each subset,
VRUs in both halves), so PEDESTRIAN detection is measurable on held-out
data. AP@0.1 per class for nuScenes-pretrained / LUMPI zero-shot /
LUMPI+TUMTraf-fine-tuned. Only classes with held-out GT are shown
(bus/bicycle clustered into the train portion).

Usage: python scripts/plot_gap_closing_vru.py
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

TAGS = [('pretrained', 'nuScenes-pretrained', '#c2c2c2'),
        ('lumpi', 'LUMPI zero-shot (Hanover only)', '#7badd6'),
        ('tumtraf', 'LUMPI+TUMTraf-fine-tuned', '#1f5c99')]


def load(path):
    out = {}
    gt = {}
    for r in csv.DictReader(open(path)):
        try:
            v = float(r['ap@0.1'])
        except ValueError:
            v = np.nan
        out.setdefault(r['tag'], {})[r['class']] = v
        gt[r['class']] = int(r['gt'])
    return out, gt


def main():
    ap, gt = load('outputs/tumtraf_ts_val/ap_eval.csv')
    # classes present in held-out val (GT>0), plus mAP; keep a sensible order
    order = ['car', 'truck', 'bus', 'pedestrian', 'bicycle', 'motorcycle']
    classes = [c for c in order if gt.get(c, 0) > 0] + ['mAP']

    x = np.arange(len(classes))
    w = 0.26
    fig, axx = plt.subplots(figsize=(10, 5.5), layout='constrained')
    for k, (tag, label, color) in enumerate(TAGS):
        vals = [ap.get(tag, {}).get(c, np.nan) for c in classes]
        axx.bar(x + (k - 1) * w, vals, w, label=label, color=color, edgecolor='white')
        for xi, v in zip(x + (k - 1) * w, vals):
            if not np.isnan(v):
                axx.text(xi, v + 0.8, f'{v:.0f}', ha='center', fontsize=7.5,
                         color=color if color != '#c2c2c2' else '#666')

    axx.set_xticks(x)
    axx.set_xticklabels([c.capitalize() for c in classes[:-1]] + ['mAP'])
    axx.set_ylabel('AP@0.1')
    axx.set_ylim(0, max(95, np.nanmax([ap.get('tumtraf', {}).get(c, 0)
                                       for c in classes]) + 8))
    axx.set_title('Closing the cross-site gap incl. VRUs — TUMTraf, temporal-split held-out val\n'
                  'pedestrians now in both splits; AP@0.1, fine-tuned on s01-s03 (first 70%)',
                  fontsize=10.5)
    axx.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
    axx.grid(axis='y', alpha=0.25)

    out = 'figures/gap_closing_vru_tumtraf.png'
    fig.savefig(out, dpi=200)
    print(f'wrote {out}  (classes: {classes})')


if __name__ == '__main__':
    main()
