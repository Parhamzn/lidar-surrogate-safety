#!/usr/bin/env python
"""Surrogate-safety extraction over saved tracks: TTC, PET, hard braking.

Reads tracks_<scene>.pkl files and mines every track pair for conflicts,
writing conflicts.csv and printing a summary. This is the final stage of
the pipeline: point clouds -> detections -> tracks -> kinematics ->
surrogate safety metrics.

Usage: python scripts/extract_conflicts.py outputs/nuscenes_mini
"""

from __future__ import annotations

import csv
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


from lidar_pilot.metrics import hard_braking_events, min_pet, min_ttc

MIN_TRACK_LEN = 5          # samples
MIN_DISPLACEMENT = 3.0     # m net displacement: separates genuinely moving
                           # users from association jitter on parked objects
                           # (peak instantaneous speed is fooled by jitter)
MIN_BRAKE_SPEED = 3.0      # m/s: braking from walking pace is not an HBE
SEVERE_TTC = 3.0           # s, report TTC conflicts below this
SEVERE_PET = 3.0           # s, report PET events below this


def bboxes_overlap(a, b, margin=3.0) -> bool:
    """Cheap spatial prefilter: inflated BEV bounding boxes must intersect."""
    a_lo, a_hi = a.xy.min(0) - margin, a.xy.max(0) + margin
    b_lo, b_hi = b.xy.min(0) - margin, b.xy.max(0) + margin
    return bool(np.all(a_hi >= b_lo) and np.all(b_hi >= a_lo))


def main(outputs_dir: str):
    out_dir = Path(outputs_dir)
    rows = []
    for pkl in sorted(out_dir.glob('tracks_*.pkl')):
        scene = pkl.stem.replace('tracks_', '')
        all_trajs = [tr for tr in pickle.load(open(pkl, 'rb')) if len(tr) >= MIN_TRACK_LEN]
        # Surrogate conflicts are interactions between moving road users;
        # parked vehicles and bike racks are scenery, not conflict parties.
        trajs = [tr for tr in all_trajs
                 if np.linalg.norm(tr.xy - tr.xy[0], axis=1).max() >= MIN_DISPLACEMENT]

        n_pairs = n_ttc = n_pet = 0
        for a, b in combinations(trajs, 2):
            if not bboxes_overlap(a, b):
                continue
            n_pairs += 1
            if a.overlap_window(b) is not None:
                res = min_ttc(a, b)
                if res is not None and res.min_ttc < SEVERE_TTC:
                    n_ttc += 1
                    rows.append([scene, 'TTC', f'{res.min_ttc:.2f}', f'{res.t_at_min:.1f}',
                                 a.track_id, a.label, b.track_id, b.label])
            pet = min_pet(a, b)
            if pet is not None and pet.pet < SEVERE_PET:
                n_pet += 1
                rows.append([scene, 'PET', f'{pet.pet:.2f}', f'{pet.t_second:.1f}',
                             pet.first_id, a.label if a.track_id == pet.first_id else b.label,
                             pet.second_id, b.label if b.track_id == pet.second_id else a.label])

        n_hbe = 0
        for tr in trajs:
            for ev in hard_braking_events(tr):
                if ev.speed_before < MIN_BRAKE_SPEED:
                    continue
                n_hbe += 1
                rows.append([scene, 'HBE', f'{ev.peak_decel:.2f}', f'{ev.t_start:.1f}',
                             tr.track_id, tr.label, '', ''])
        print(f'{scene}: {len(trajs)}/{len(all_trajs)} moving tracks, '
              f'{n_pairs} interacting pairs -> {n_ttc} TTC<{SEVERE_TTC}s, '
              f'{n_pet} PET<{SEVERE_PET}s, {n_hbe} hard-braking')

    with open(out_dir / 'conflicts.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scene', 'metric', 'value', 't', 'id_a', 'class_a', 'id_b', 'class_b'])
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
    main(sys.argv[1] if len(sys.argv) > 1 else 'outputs/nuscenes_mini')
