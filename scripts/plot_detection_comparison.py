#!/usr/bin/env python
"""Domain-shift comparison figure: pretrained vs fine-tuned, same protocol.

Reads detection_eval.csv rows for two tags and plots per-class F1 plus
recall side by side.

Usage: python scripts/plot_detection_comparison.py outputs/lumpi_eval/detection_eval.csv \
    --baseline pretrained_valwin --finetuned finetuned_ep20
"""

from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv_path')
    ap.add_argument('--baseline', default='pretrained_valwin')
    ap.add_argument('--finetuned', default='finetuned_ep20')
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv_path))
            if r['tag'] in (args.baseline, args.finetuned) and int(r['gt']) > 0]
    classes = sorted({r['class'] for r in rows},
                     key=lambda c: -max(int(r['gt']) for r in rows if r['class'] == c))

    def get(tag, cls, field):
        for r in rows:
            if r['tag'] == tag and r['class'] == cls:
                v = float(r[field])
                return 0.0 if np.isnan(v) else v
        return 0.0

    x = np.arange(len(classes))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, field in ((axes[0], 'f1'), (axes[1], 'recall')):
        ax.bar(x - 0.18, [get(args.baseline, c, field) for c in classes],
               width=0.36, label='pretrained (nuScenes, ego-vehicle)',
               color='#b0b6c4')
        ax.bar(x + 0.18, [get(args.finetuned, c, field) for c in classes],
               width=0.36, label='fine-tuned on roadside data',
               color='#2c7fb8')
        ax.set_xticks(x, classes, rotation=20)
        ax.set_title(field.upper())
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, axis='y')
    axes[0].set_ylabel('score (center-distance matching, 2 m)')
    axes[0].legend(fontsize=9)
    n_gt = {c: max(int(r['gt']) for r in rows if r['class'] == c) for c in classes}
    fig.suptitle('Ego-vehicle → roadside domain shift: detection before/after '
                 f'fine-tuning (held-out slice, GT n: '
                 + ', '.join(f'{c} {n_gt[c]}' for c in classes) + ')',
                 fontsize=10)
    fig.tight_layout()
    out = 'figures/detection_domain_shift.png'
    fig.savefig(out, dpi=200)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
