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

Detection inference dominates the runtime, so the raw detections can be
cached once (--save-detections) and the tracking stage re-run on any
machine with --from-detections; tracker parameters then sweep in seconds
instead of GPU-minutes. The cache keeps every detection the model emits
(its internal score floor, full BEV grid), and --score-thr / --max-range
are applied at tracking time.
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
# Feed the model every point its BEV grid can voxelize (the +-51.2 m grid
# reaches ~72 m radially in the corners); the voxelizer drops the rest.
POINT_CROP_RANGE = 80.0


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


def detect_frames(args):
    """Run the detector over the ply sequence, yielding per-frame results."""
    from mmdet3d.apis import LidarDet3DInferencer
    inf = LidarDet3DInferencer(model=args.config, weights=args.checkpoint,
                               device='cuda:0')
    inf.show_progress = False

    plys = sorted(Path(args.lidar_dir).glob('*.ply'))[::args.stride]
    for ply in plys:
        fidx = int(ply.stem)
        pts = read_points(ply, args.z_shift)
        keep = np.linalg.norm(pts[:, :2], axis=1) < POINT_CROP_RANGE
        res = inf(dict(points=pts[keep]), no_save_vis=True)
        p = res['predictions'][0]
        boxes9 = np.asarray(p['bboxes_3d'], float).reshape(-1, 9)
        scores = np.asarray(p['scores_3d'], float)
        labels = np.asarray(p['labels_3d'], int)
        yield fidx, boxes9, scores, labels


def cached_frames(path):
    """Yield frames from a detection cache written by --save-detections."""
    with open(path, 'rb') as f:
        cache = pickle.load(f)
    assert cache['meta']['classes'] == LUMPI_TRAIN_CLASSES
    for fr in cache['frames']:
        yield fr['fidx'], fr['boxes9'].astype(float), \
            fr['scores'].astype(float), fr['labels'].astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lidar-dir')
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
    ap.add_argument('--save-detections', default=None, metavar='PKL',
                    help='also write every raw detection to this cache file')
    ap.add_argument('--from-detections', default=None, metavar='PKL',
                    help='skip the GPU and re-track from a detection cache')
    ap.add_argument('--max-age', type=int, default=5)
    ap.add_argument('--min-hits', type=int, default=3)
    ap.add_argument('--gate-growth', type=float, default=0.0)
    # 4.5 m/s^2 process noise for the online association filter: calibrated
    # against the GT braking reference (the stock 3.0 lags hard maneuvers;
    # see scripts/sweep_hbe_recovery.py)
    ap.add_argument('--kf-accel-std', type=float, default=4.5)
    # Validated operating point: record raw matched detections and smooth
    # them offline with an RTS pass — a causal filter must lag maneuvers,
    # a forward-backward smoother does not. The rts noise model
    # (meas 0.15 m, accel allowance 15 m/s^2) is a calibrated q/r
    # bandwidth, chosen by count parity + event-level F1 against the
    # ground-truth conflict reference.
    ap.add_argument('--record-source', default='detection',
                    choices=['posterior', 'detection'])
    ap.add_argument('--rts', default=True,
                    action=argparse.BooleanOptionalAction,
                    help='RTS-smooth finished tracks (--no-rts to disable)')
    ap.add_argument('--rts-accel-std', type=float, default=15.0)
    ap.add_argument('--rts-meas-std', type=float, default=0.15)
    args = ap.parse_args()

    if args.from_detections:
        frames = cached_frames(args.from_detections)
    else:
        if not args.lidar_dir:
            raise SystemExit('--lidar-dir is required unless --from-detections')
        frames = detect_frames(args)

    tracker = Tracker3D(max_match_distance=MATCH_GATES, max_age=args.max_age,
                        min_hits=args.min_hits, min_score=args.score_thr,
                        gate_growth=args.gate_growth,
                        kf_params=dict(accel_std=args.kf_accel_std),
                        record_source=args.record_source)
    cache = dict(meta=dict(classes=LUMPI_TRAIN_CLASSES, config=args.config,
                           checkpoint=args.checkpoint, fps=FPS,
                           point_crop_range=POINT_CROP_RANGE,
                           stride=args.stride),
                 frames=[])

    wall0 = time.perf_counter()
    n_frames = 0
    for k, (fidx, boxes9, scores, labels) in enumerate(frames):
        n_frames += 1
        t = fidx / FPS
        if args.save_detections:
            cache['frames'].append(dict(
                fidx=fidx, boxes9=boxes9.astype(np.float32),
                scores=scores.astype(np.float32),
                labels=labels.astype(np.int8)))

        rng_ok = np.linalg.norm(boxes9[:, :2], axis=1) <= args.max_range
        boxes9, scores, labels = boxes9[rng_ok], scores[rng_ok], labels[rng_ok]
        names = [LUMPI_TRAIN_CLASSES[c] for c in labels]

        tboxes = np.column_stack([boxes9[:, :3], boxes9[:, 6],
                                  boxes9[:, 3], boxes9[:, 4], boxes9[:, 5]])
        tracker.step(tboxes, scores, names, t, velocities=boxes9[:, 7:9])

        if k % 500 == 0:
            rate = (k + 1) / (time.perf_counter() - wall0)
            print(f'{k} frames ({rate:.1f} fps, '
                  f'{len(tracker.trajectories)} confirmed tracks so far)',
                  flush=True)

    trajs = tracker.trajectories
    if args.rts:
        from lidar_pilot.tracking import rts_smooth
        trajs = [rts_smooth(tr, accel_std=args.rts_accel_std,
                            meas_std=args.rts_meas_std) for tr in trajs]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'tracks_{args.name}.pkl', 'wb') as f:
        pickle.dump(trajs, f)
    if args.save_detections:
        with open(args.save_detections, 'wb') as f:
            pickle.dump(cache, f, protocol=4)
        print(f'wrote detection cache {args.save_detections} '
              f'({len(cache["frames"])} frames)')
    dt = time.perf_counter() - wall0
    print(f'done: {n_frames} frames in {dt/60:.1f} min '
          f'({n_frames/dt:.1f} fps) -> {len(trajs)} confirmed tracks')
    print(f'wrote {out / f"tracks_{args.name}.pkl"}')


if __name__ == '__main__':
    main()
