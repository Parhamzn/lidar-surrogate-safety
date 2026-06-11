#!/usr/bin/env python
"""Phase B: pretrained CenterPoint on LUMPI roadside frames vs GT labels.

Quantifies the ego-vehicle -> infrastructure domain shift: runs a
nuScenes-pretrained detector on fused roadside point clouds and scores
per-class precision/recall against the LUMPI ground truth.

Protocol notes (kept deliberately simple and stated in the output):
  * matching: per class, greedy by detection score, BEV center distance
    <= 2.0 m (nuScenes-style), within --max-range of the origin only
    (the model's BEV grid ends at 51.2 m).
  * z handling: nuScenes models expect the ground near z = -1.84 in the
    sensor frame; --z-shift auto estimates the ground as the modal point
    height and shifts the cloud to match. Matching itself is BEV-only.
  * classes: the 6 shared classes are scored. LUMPI's scooter has no
    nuScenes counterpart: scooter GT is an ignore region (detections
    matched to it are dropped, misses are not counted).
  * detections of non-shared classes (trailer, construction_vehicle,
    traffic_cone, barrier) are reported but not scored.

Usage (on the GPU box):
  python eval_lumpi_detection.py --lidar-dir data/lumpi/test_data/Measurement5/lidar \
      --label-csv data/lumpi/Label/Measurement5/Label.csv --out-dir outputs/lumpi_eval
"""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

FPS = 10
NUS_CLASSES = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
LUMPI_CLASSES = {0: 'pedestrian', 1: 'car', 2: 'bicycle', 3: 'motorcycle',
                 4: 'bus', 5: 'truck', 6: 'scooter'}
SHARED = ['car', 'truck', 'bus', 'pedestrian', 'bicycle', 'motorcycle']
NUS_GROUND_Z = -1.84


def load_labels_by_frame(csv_path):
    cols = np.loadtxt(csv_path, delimiter=',', skiprows=1,
                      usecols=range(16), ndmin=2)
    frames = defaultdict(list)
    for r in cols:
        frames[int(round(r[0] * FPS))].append(
            (LUMPI_CLASSES.get(int(r[7]), 'unknown'), r[9], r[10]))
    return frames


def read_points(ply_path, z_shift):
    from plyfile import PlyData
    v = PlyData.read(str(ply_path))['vertex']
    pts = np.column_stack([
        np.asarray(v['x'], np.float32),
        np.asarray(v['y'], np.float32),
        np.asarray(v['z'], np.float32) + z_shift,
        np.asarray(v['intensity'], np.float32),
        np.zeros(v.count, np.float32),          # sweep dt: single static frame
    ])
    return pts


def estimate_z_shift(ply_path):
    from plyfile import PlyData
    v = PlyData.read(str(ply_path))['vertex']
    pts = np.column_stack([v['x'], v['y'], v['z']])
    near = pts[np.linalg.norm(pts[:, :2], axis=1) < 50]
    hist, edges = np.histogram(near[:, 2], bins=np.arange(-10, 5, 0.2))
    ground = edges[np.argmax(hist)] + 0.1
    return float(NUS_GROUND_Z - ground)


def match_frame(dets, gts, max_dist=2.0):
    """Greedy per-class matching. dets: (cls, x, y, score) sorted globally
    by score desc; gts: (cls, x, y). Returns tp, fp, fn, ignored per class."""
    stats = {c: dict(tp=0, fp=0, fn=0) for c in SHARED}
    n_other = 0
    gt_used = [False] * len(gts)
    scooters = [(i, g) for i, g in enumerate(gts) if g[0] == 'scooter']

    for cls, x, y, _ in sorted(dets, key=lambda d: -d[3]):
        # scooter ignore region: any detection on top of a scooter GT is
        # dropped from scoring regardless of its class
        near_scooter = any(not gt_used[i] and np.hypot(x - g[1], y - g[2]) <= max_dist
                           for i, g in scooters)
        if near_scooter:
            continue
        if cls not in SHARED:
            n_other += 1
            continue
        best_i, best_d = None, max_dist
        for i, (gcls, gx, gy) in enumerate(gts):
            if gt_used[i] or gcls != cls:
                continue
            d = np.hypot(x - gx, y - gy)
            if d <= best_d:
                best_i, best_d = i, d
        if best_i is None:
            stats[cls]['fp'] += 1
        else:
            gt_used[best_i] = True
            stats[cls]['tp'] += 1

    for i, (gcls, _, _) in enumerate(gts):
        if not gt_used[i] and gcls in SHARED:
            stats[gcls]['fn'] += 1
    return stats, n_other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lidar-dir', required=True)
    ap.add_argument('--label-csv', required=True)
    ap.add_argument('--out-dir', default='outputs/lumpi_eval')
    ap.add_argument('--config', default='/mnt/T9/parham/lidar-pilot/mmdetection3d/configs/'
                    'centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py')
    ap.add_argument('--checkpoint', default='/mnt/T9/parham/lidar-pilot/checkpoints/'
                    'centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth')
    ap.add_argument('--stride', type=int, default=1, help='use every Nth frame')
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--max-range', type=float, default=50.0)
    ap.add_argument('--z-shift', default='auto')
    ap.add_argument('--tag', default='pretrained')
    args = ap.parse_args()

    lidar_dir = Path(args.lidar_dir)
    plys = sorted(lidar_dir.glob('*.ply'))[::args.stride]
    if not plys:
        raise SystemExit(f'no .ply files in {lidar_dir}')

    z_shift = (estimate_z_shift(plys[0]) if args.z_shift == 'auto'
               else float(args.z_shift))
    print(f'{len(plys)} frames | z-shift {z_shift:+.2f} m | '
          f'range <= {args.max_range} m | score >= {args.score_thr}')

    from mmdet3d.apis import LidarDet3DInferencer
    inf = LidarDet3DInferencer(model=args.config, weights=args.checkpoint,
                               device='cuda:0')
    inf.show_progress = False

    labels = load_labels_by_frame(args.label_csv)
    totals = {c: dict(tp=0, fp=0, fn=0) for c in SHARED}
    n_other_total = 0
    per_frame_dets = {}

    for ply in plys:
        fidx = int(ply.stem)
        pts = read_points(ply, z_shift)
        keep = np.linalg.norm(pts[:, :2], axis=1) < args.max_range + 10
        res = inf(dict(points=pts[keep]), no_save_vis=True)
        p = res['predictions'][0]
        boxes = np.asarray(p['bboxes_3d'], float).reshape(-1, 9)
        scores = np.asarray(p['scores_3d'], float)
        cls_ids = np.asarray(p['labels_3d'], int)

        dets = [(NUS_CLASSES[c], b[0], b[1], s)
                for b, s, c in zip(boxes, scores, cls_ids)
                if s >= args.score_thr and np.hypot(b[0], b[1]) <= args.max_range]
        gts = [(c, x, y) for c, x, y in labels.get(fidx, [])
               if np.hypot(x, y) <= args.max_range]
        per_frame_dets[fidx] = dets

        stats, n_other = match_frame(dets, gts)
        n_other_total += n_other
        for c in SHARED:
            for k in ('tp', 'fp', 'fn'):
                totals[c][k] += stats[c][k]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'detections_{args.tag}.pkl', 'wb') as f:
        pickle.dump(dict(dets=per_frame_dets, z_shift=z_shift), f)

    print(f'\n{"class":<12}{"GT":>7}{"TP":>7}{"FP":>7}{"FN":>7}'
          f'{"precision":>11}{"recall":>9}{"F1":>7}')
    rows = []
    for c in SHARED:
        tp, fp, fn = totals[c]['tp'], totals[c]['fp'], totals[c]['fn']
        gt_n = tp + fn
        prec = tp / (tp + fp) if tp + fp else float('nan')
        rec = tp / gt_n if gt_n else float('nan')
        f1 = (2 * prec * rec / (prec + rec)
              if prec + rec and not (np.isnan(prec) or np.isnan(rec)) else float('nan'))
        print(f'{c:<12}{gt_n:>7}{tp:>7}{fp:>7}{fn:>7}{prec:>11.3f}{rec:>9.3f}{f1:>7.3f}')
        rows.append([args.tag, c, gt_n, tp, fp, fn,
                     f'{prec:.4f}', f'{rec:.4f}', f'{f1:.4f}'])
    print(f'(unscored detections of non-shared classes: {n_other_total})')

    summary = out / 'detection_eval.csv'
    new = not summary.exists()
    with open(summary, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['tag', 'class', 'gt', 'tp', 'fp', 'fn',
                        'precision', 'recall', 'f1'])
        w.writerows(rows)
    print(f'appended to {summary}')


if __name__ == '__main__':
    main()
