#!/usr/bin/env python
"""Cross-site generalization figure: detector on TUMTraf Intersection.

Bars: per-class F1 of the nuScenes-pretrained head vs the LUMPI-fine-tuned
head, both run unchanged on TUMTraf (Munich). Markers: the same
fine-tuned head's in-site F1 on LUMPI (Hanover) — the "same-site ceiling".
The gap between the fine-tuned bar and its marker is the site-transfer
cost; the gap between the two bars is what fine-tuning on a *different*
roadside site buys you.

Usage: python scripts/plot_cross_site.py
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

CLASSES = ['car', 'pedestrian', 'bicycle', 'bus', 'truck']


def f1s(path, tag):
    rows = {r['class']: r for r in csv.DictReader(open(path)) if r['tag'] == tag}

    def val(c):
        v = rows.get(c, {}).get('f1', 'nan')
        try:
            return float(v)
        except ValueError:
            return np.nan
    return {c: val(c) for c in CLASSES}


def main():
    pre = f1s('outputs/tumtraf_eval/detection_eval.csv', 'pretrained_tumtraf')
    fin = f1s('outputs/tumtraf_eval/detection_eval.csv', 'finetuned_tumtraf')
    insite = f1s('outputs/lumpi_m6_eval/detection_eval.csv', 'finetuned_m6')

    x = np.arange(len(CLASSES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')

    ax.bar(x - w / 2, [pre[c] for c in CLASSES], w, label='nuScenes-pretrained → TUMTraf',
           color='#c2c2c2', edgecolor='white')
    ax.bar(x + w / 2, [fin[c] for c in CLASSES], w, label='LUMPI-fine-tuned → TUMTraf',
           color='#2c6fb5', edgecolor='white')
    # in-site ceiling markers (same fine-tuned head, evaluated on LUMPI)
    for i, c in enumerate(CLASSES):
        if not np.isnan(insite[c]):
            ax.plot([i - w, i + w], [insite[c], insite[c]], '--', color='#c23a2f', lw=1.6,
                    label='same head, in-site (LUMPI/Hanover)' if i == 0 else None)
            ax.plot(i + w / 2, insite[c], 'v', color='#c23a2f', ms=7)

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASSES])
    ax.set_ylabel('Detection F1')
    ax.set_ylim(0, 0.95)
    ax.set_title('Cross-site generalization: roadside detector tested on TUMTraf Intersection (Munich)\n'
                 'fine-tuned only on LUMPI (Hanover) — zero TUMTraf training', fontsize=11)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.grid(axis='y', alpha=0.25)
    for i, c in enumerate(CLASSES):
        ax.text(i + w / 2, fin[c] + 0.015, f'{fin[c]:.2f}', ha='center', fontsize=8,
                color='#2c6fb5', fontweight='bold')

    out = 'figures/cross_site_tumtraf.png'
    fig.savefig(out, dpi=200)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
