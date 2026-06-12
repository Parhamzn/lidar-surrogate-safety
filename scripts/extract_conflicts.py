#!/usr/bin/env python
"""Surrogate-safety extraction over saved tracks: TTC, PET, hard braking.

Reads tracks_<scene>.pkl files and mines every track pair for conflicts,
writing conflicts.csv and printing a summary. This is the final stage of
the pipeline: point clouds -> detections -> tracks -> kinematics ->
surrogate safety metrics. The mining logic itself (artifact filters,
severity thresholds) lives in lidar_pilot.conflicts.

Usage: python scripts/extract_conflicts.py outputs/nuscenes_mini

Ground-truth trajectories use the defaults (their human-fitted boxes get
one smoothing stage, the MA-3 window). RTS-smoothed tracker output is
already smoothed, so it skips the extra window — everything else is
identical (operating point from scripts/sweep_hbe_recovery.py):

  python scripts/extract_conflicts.py outputs/lumpi_e2e --hbe-smooth-window 1
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

from lidar_pilot.conflicts import SEVERE_PET, SEVERE_TTC, mine_conflicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outputs_dir', nargs='?', default='outputs/nuscenes_mini')
    ap.add_argument('--hbe-smooth-window', type=int, default=3,
                    help='moving-average window for the braking estimator')
    ap.add_argument('--hbe-min-duration', type=float, default=0.2)
    ap.add_argument('--hbe-speed-source', default='positions',
                    choices=['positions', 'kf_velocity'])
    ap.add_argument('--hbe-despike', action='store_true',
                    help='median-filter the speed series before the '
                         'braking estimator (for glitchy tracker output)')
    args = ap.parse_args()

    out_dir = Path(args.outputs_dir)
    rows = []
    for pkl in sorted(out_dir.glob('tracks_*.pkl')):
        scene = pkl.stem.replace('tracks_', '')
        all_trajs = pickle.load(open(pkl, 'rb'))
        scene_rows, st = mine_conflicts(
            all_trajs, scene,
            hbe_smooth_window=args.hbe_smooth_window,
            hbe_min_duration=args.hbe_min_duration,
            hbe_speed_source=args.hbe_speed_source,
            hbe_despike=args.hbe_despike)
        rows += scene_rows
        print(f'{scene}: {st.n_moving}/{st.n_tracks} moving tracks '
              f'({st.n_implausible} class-implausible excluded), '
              f'{st.n_pairs} pairs ({st.n_same_object} same-object skipped) -> '
              f'{st.n_ttc} TTC<{SEVERE_TTC}s, {st.n_pet} PET<{SEVERE_PET}s, '
              f'{st.n_hbe} hard-braking')

    with open(out_dir / 'conflicts.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scene', 'metric', 'value', 't', 'id_a', 'class_a',
                    'id_b', 'class_b', 'x', 'y'])
        w.writerows(rows)
    print(f'\nwrote {out_dir / "conflicts.csv"} ({len(rows)} events)')

    # most severe interactions involving a VRU
    vru = [r for r in rows if r[1] in ('TTC', 'PET')
           and ('pedestrian' in (r[5], r[7]) or 'bicycle' in (r[5], r[7])
                or 'motorcycle' in (r[5], r[7]))]
    vru.sort(key=lambda r: float(r[2]))
    if vru:
        print('\nMost severe VRU conflicts:')
        for r in vru[:8]:
            print(f'  {r[0]}  {r[1]}={r[2]}s at t={r[3]}s   '
                  f'#{r[4]} {r[5]}  vs  #{r[6]} {r[7]}')


if __name__ == '__main__':
    main()
