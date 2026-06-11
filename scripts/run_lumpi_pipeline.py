#!/usr/bin/env python
"""End-to-end machine pipeline on raw LUMPI frames: detect -> track.

No ground-truth labels are used anywhere: the fine-tuned CenterPoint
produces detections per 10 Hz frame, the Kalman tracker links them into
trajectories, and the result is saved in the same tracks_*.pkl format the
conflict miner consumes. Comparing its conflicts against the GT-derived
ones is the pipeline's capstone validation.

Usage (on the GPU box):
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python run_lumpi_pipeline.py \
      --lidar-dir data/lumpi/Measurement5/lidar \
      --out-dir outputs/lumpi_e2e --name Measurement5_e2e
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from lidar_pilot.tracking import Tracker3D

LUMPI_TRAIN_CLASSES = ['car', 'truck', 'bus', 'pedestrian', 'bicycle',
                       'motorcycle', 'scooter']
# 10 Hz association gates: per-frame travel is small, so gates are tight
# (loose gates let pedestrian tracks hop between people in groups)
MATCH_GATES = {'car': 2.5, 'truck': 3.0, 'bus': 3.5, 'pedestrian': 1.0,
               'bicycle': 1.4, 'motorcycle': 1.6, 'scooter': 1.4,
               'default': 2.5}
FPS = 10


def read_points(ply_path, z_shift):
    from plyfile import PlyData
    v = PlyData.read(str(ply_path))['vertex']
    return np.column_stack([
        np.asarray(v['x'], np.float32),
        np.asarray(v['y'], np.float32),
        np.asarray(v['z'], np.float32) + z_shift,
        np.asarray(v['intensity'], np.float32),
        np.zeros(v.count, np.float32),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lidar-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--name', default='Measurement5_e2e')
    ap.add_argument('--config', default='/mnt/T9/parham/lidar-pilot/repo/configs/'
                    'centerpoint_pillar02_lumpi_finetune.py')
    ap.add_argument('--checkpoint', default='/mnt/T9/parham/lidar-pilot/'
                    'work_dirs/lumpi_finetune/epoch_20.pth')
    ap.add_argument('--score-thr', type=float, default=0.35)
    ap.add_argument('--max-range', type=float, default=50.0)
    ap.add_argument('--z-shift', type=float, default=0.26)
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()

    from mmdet3d.apis import LidarDet3DInferencer
    inf = LidarDet3DInferencer(model=args.config, weights=args.checkpoint,
                               device='cuda:0')
    inf.show_progress = False

    plys = sorted(Path(args.lidar_dir).glob('*.ply'))[::args.stride]
    tracker = Tracker3D(max_match_distance=MATCH_GATES, max_age=5,
                        min_hits=3, min_score=args.score_thr)

    wall0 = time.perf_counter()
    for k, ply in enumerate(plys):
        fidx = int(ply.stem)
        t = fidx / FPS
        pts = read_points(ply, args.z_shift)
        keep = np.linalg.norm(pts[:, :2], axis=1) < args.max_range + 10
        res = inf(dict(points=pts[keep]), no_save_vis=True)
        p = res['predictions'][0]
        boxes9 = np.asarray(p['bboxes_3d'], float).reshape(-1, 9)
        scores = np.asarray(p['scores_3d'], float)
        labels = [LUMPI_TRAIN_CLASSES[c] for c in p['labels_3d']]

        rng_ok = np.linalg.norm(boxes9[:, :2], axis=1) <= args.max_range
        boxes9, scores = boxes9[rng_ok], scores[rng_ok]
        labels = [l for l, ok in zip(labels, rng_ok) if ok]

        tboxes = np.column_stack([boxes9[:, :3], boxes9[:, 6],
                                  boxes9[:, 3], boxes9[:, 4], boxes9[:, 5]])
        tracker.step(tboxes, scores, labels, t, velocities=boxes9[:, 7:9])

        if k % 500 == 0:
            rate = (k + 1) / (time.perf_counter() - wall0)
            print(f'{k}/{len(plys)} frames ({rate:.1f} fps, '
                  f'{len(tracker.trajectories)} confirmed tracks so far)',
                  flush=True)

    trajs = tracker.trajectories
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'tracks_{args.name}.pkl', 'wb') as f:
        pickle.dump(trajs, f)
    dt = time.perf_counter() - wall0
    print(f'done: {len(plys)} frames in {dt/60:.1f} min '
          f'({len(plys)/dt:.1f} fps) -> {len(trajs)} confirmed tracks')
    print(f'wrote {out / f"tracks_{args.name}.pkl"}')


if __name__ == '__main__':
    main()
