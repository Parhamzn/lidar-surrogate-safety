#!/usr/bin/env python
"""Convert LUMPI ground-truth labels to trajectories for conflict mining.

Loads a measurement's Label.csv, reports the traffic composition, and
saves tracks_<name>.pkl in the same format the nuScenes pipeline emits,
so extract_conflicts.py and the plotting scripts run unchanged on
roadside intersection data.

Usage:
  python scripts/run_lumpi_conflicts.py data/lumpi/Measurement5_Label.csv \
      --name Measurement5 --out-dir outputs/lumpi
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from lidar_pilot.io import load_lumpi_trajectories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('label_csv')
    ap.add_argument('--name', default='Measurement5')
    ap.add_argument('--out-dir', default='outputs/lumpi')
    args = ap.parse_args()

    trajs = load_lumpi_trajectories(args.label_csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'tracks_{args.name}.pkl', 'wb') as f:
        pickle.dump(trajs, f)

    t_max = max(tr.t[-1] for tr in trajs)
    print(f'{args.name}: {len(trajs)} tracked objects over {t_max/60:.1f} min')
    comp = Counter(tr.label for tr in trajs)
    for cls, n in comp.most_common():
        durs = [tr.duration for tr in trajs if tr.label == cls]
        print(f'  {cls:<12} x{n:<5} median presence {np.median(durs):6.1f} s')
    print(f'wrote {out / f"tracks_{args.name}.pkl"}')


if __name__ == '__main__':
    main()
