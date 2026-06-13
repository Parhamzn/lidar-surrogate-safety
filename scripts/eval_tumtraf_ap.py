#!/usr/bin/env python
"""mAP@0.1 on TUMTraf Intersection — comparable to the dataset's benchmark.

The TUMTraf paper reports detection as mAP at IoU 0.1 over 6 classes
(Car, Truck, Bus, Motorcycle, Pedestrian, Bicycle). This script computes
the same metric for our checkpoints so the cross-site / fine-tuned numbers
sit on the published scale (their in-domain PointPillars: ~47 single-LiDAR,
55.2 early-fusion). Matching uses BEV rotated-box IoU (shapely); AP is the
area under the score-ranked precision-recall curve (all-points / VOC2010).

Caveat vs their protocol: same metric and threshold, but (a) BEV IoU not
3D IoU (marginal at 0.1), (b) no Easy/Mod/Hard difficulty stratification,
(c) our train regime differs (zero-shot LUMPI, or fine-tuned on s01-s03).
Evaluate on the held-out subset (default s04) for the fine-tuned model.

Usage (GPU box), the three checkpoints in turn:
  python eval_tumtraf_ap.py --tag pretrained                 # nuScenes head
  python eval_tumtraf_ap.py --config .../lumpi_finetune.py \
      --checkpoint .../lumpi_finetune/epoch_20.pth --det-classes lumpi --tag lumpi
  python eval_tumtraf_ap.py --config .../tumtraf_finetune.py \
      --checkpoint .../tumtraf_finetune/epoch_20.pth --det-classes lumpi --tag tumtraf
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).parent))
from eval_lumpi_detection import LUMPI_TRAIN_CLASSES, NUS_CLASSES, NUS_GROUND_Z  # noqa: E402
from eval_tumtraf_detection import frame_pairs, label_ground_z  # noqa: E402

from lidar_pilot.io.tumtraf import read_openlabel_boxes, read_pcd  # noqa: E402

BENCH_CLASSES = ['car', 'truck', 'bus', 'pedestrian', 'bicycle', 'motorcycle']
IOU_THR = 0.1


def bev_poly(x, y, l, w, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array([[ l / 2,  w / 2], [ l / 2, -w / 2],
                        [-l / 2, -w / 2], [-l / 2,  w / 2]])
    r = corners @ np.array([[c, s], [-s, c]]) + [x, y]
    return Polygon(r)


def iou_bev(a, b):
    pa, pb = bev_poly(*a), bev_poly(*b)
    if not pa.is_valid or not pb.is_valid:
        return 0.0
    inter = pa.intersection(pb).area
    return inter / (pa.area + pb.area - inter + 1e-9)


def average_precision(scores, tps, n_gt):
    """All-points AP from score-ranked TP flags."""
    if n_gt == 0:
        return float('nan')
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps, float)[order]
    fp = 1 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    # monotone envelope, integrate
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/mnt/T9/parham/lidar-pilot/data/tumtraf')
    ap.add_argument('--subsets', default='s04')
    ap.add_argument('--sensor', default='s110_lidar_ouster_south')
    ap.add_argument('--out-dir', default='outputs/tumtraf_eval')
    ap.add_argument('--config', default='/mnt/T9/parham/lidar-pilot/mmdetection3d/configs/'
                    'centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py')
    ap.add_argument('--checkpoint', default='/mnt/T9/parham/lidar-pilot/checkpoints/'
                    'centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth')
    ap.add_argument('--det-classes', choices=['nuscenes', 'lumpi'], default='nuscenes')
    ap.add_argument('--score-thr', type=float, default=0.1)
    ap.add_argument('--max-range', type=float, default=50.0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--tag', default='pretrained')
    args = ap.parse_args()

    det_class_list = (NUS_CLASSES if args.det_classes == 'nuscenes'
                      else LUMPI_TRAIN_CLASSES)
    pairs = frame_pairs(args.data_root, args.subsets.split(','), args.sensor, args.stride)
    z_shift = NUS_GROUND_Z - label_ground_z([jp for _, jp in pairs])
    print(f'{len(pairs)} frames | z-shift {z_shift:+.2f} | head={args.det_classes} | AP@{IOU_THR}')

    from mmdet3d.apis import LidarDet3DInferencer
    inf = LidarDet3DInferencer(model=args.config, weights=args.checkpoint, device='cuda:0')
    inf.show_progress = False

    # accumulate per class: list of (score, is_tp); and GT counts
    acc = {c: [] for c in BENCH_CLASSES}
    n_gt = {c: 0 for c in BENCH_CLASSES}

    for k, (pcd_path, json_path) in enumerate(pairs):
        pts = read_pcd(pcd_path, z_shift)
        pts = pts[np.linalg.norm(pts[:, :2], axis=1) < args.max_range + 10]
        res = inf(dict(points=pts), no_save_vis=True)
        p = res['predictions'][0]
        b9 = np.asarray(p['bboxes_3d'], float).reshape(-1, 9)
        scr = np.asarray(p['scores_3d'], float)
        cid = np.asarray(p['labels_3d'], int)

        gts = {c: [] for c in BENCH_CLASSES}
        for g in read_openlabel_boxes(json_path):
            if g['label'] in BENCH_CLASSES and np.hypot(*g['xyz'][:2]) <= args.max_range:
                gts[g['label']].append((g['xyz'][0], g['xyz'][1],
                                        g['lwh'][0], g['lwh'][1], g['yaw']))
                n_gt[g['label']] += 1

        for c in BENCH_CLASSES:
            # detections of this class, sorted by score desc
            dets = [(b9[i, 0], b9[i, 1], b9[i, 3], b9[i, 4], b9[i, 6], scr[i])
                    for i in range(len(b9))
                    if det_class_list[cid[i]] == c and scr[i] >= args.score_thr
                    and np.hypot(b9[i, 0], b9[i, 1]) <= args.max_range]
            dets.sort(key=lambda d: -d[5])
            used = [False] * len(gts[c])
            for *box, s in dets:
                best_j, best_iou = -1, IOU_THR
                for j, gt in enumerate(gts[c]):
                    if used[j]:
                        continue
                    i = iou_bev(box, gt)
                    if i >= best_iou:
                        best_j, best_iou = j, i
                if best_j >= 0:
                    used[best_j] = True
                    acc[c].append((s, 1))
                else:
                    acc[c].append((s, 0))
        if k % 200 == 0:
            print(f'  {k}/{len(pairs)}', flush=True)

    print(f'\n{"class":<12}{"GT":>7}{"AP@0.1":>9}')
    aps, rows = [], []
    for c in BENCH_CLASSES:
        scores = [a[0] for a in acc[c]]
        tps = [a[1] for a in acc[c]]
        ap_c = average_precision(scores, tps, n_gt[c])
        aps.append(ap_c)
        print(f'{c:<12}{n_gt[c]:>7}{ap_c * 100:>9.2f}')
        rows.append([args.tag, c, n_gt[c], f'{ap_c * 100:.2f}'])
    valid = [a for a in aps if not np.isnan(a)]
    mAP = float(np.mean(valid)) * 100 if valid else float('nan')
    print(f'{"mAP":<12}{"":>7}{mAP:>9.2f}  (over {len(valid)} classes with GT)')
    rows.append([args.tag, 'mAP', sum(n_gt.values()), f'{mAP:.2f}'])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = out / 'ap_eval.csv'
    write_header = not summary.exists()
    with open(summary, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['tag', 'class', 'gt', 'ap@0.1'])
        w.writerows(rows)
    print(f'appended to {summary}')


if __name__ == '__main__':
    main()
