#!/usr/bin/env python
"""Cross-site detection eval: CenterPoint on TUMTraf Intersection vs GT.

The external-validity test for the pilot: a detector pretrained on
nuScenes (ego-vehicle) and fine-tuned on LUMPI (Hanover roadside) is run,
unchanged, on TUMTraf Intersection (Munich roadside) and scored against
its OpenLABEL ground truth. "Trained on Hanover, tested on Munich."

Same matching protocol as eval_lumpi_detection.py (greedy per-class, BEV
centre distance <= 2 m, within --max-range; GT of classes the head cannot
predict form ignore regions). Reused from that script via import.

Two TUMTraf specifics vs LUMPI:
  * GT and points come from the OpenLABEL/.pcd adapter (io/tumtraf.py).
  * z-shift is derived from the LABELS (median box-bottom of vehicles),
    not the modal-ground heuristic: the gantry-mounted sensor sees large
    elevated surfaces, so the modal z is not the road. nuScenes models
    expect the ground near z = -1.84.

Usage (GPU box), both checkpoints:
  python eval_tumtraf_detection.py --data-root data/tumtraf \
      --tag pretrained_tumtraf                       # nuScenes head
  python eval_tumtraf_detection.py --data-root data/tumtraf \
      --config configs/centerpoint_pillar02_lumpi_finetune.py \
      --checkpoint work_dirs/lumpi_finetune/epoch_20.pth \
      --det-classes lumpi --tag finetuned_tumtraf
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_lumpi_detection import (LUMPI_TRAIN_CLASSES, NUS_CLASSES,  # noqa: E402
                                  NUS_GROUND_Z, SHARED, match_frame)

from lidar_pilot.io.tumtraf import read_openlabel_boxes, read_pcd  # noqa: E402

SENSOR = 's110_lidar_ouster_south'   # the reference LiDAR; labels are in its frame


def frame_pairs(data_root, subsets, sensor, stride):
    """(pcd_path, json_path) pairs across the requested subsets, in order."""
    pairs = []
    for sub in subsets:
        base = Path(data_root) / f'a9_dataset_r02_{sub}'
        pcd_dir = base / 'point_clouds' / sensor
        lbl_dir = base / 'labels_point_clouds' / sensor
        for pcd in sorted(pcd_dir.glob('*.pcd')):
            jp = lbl_dir / (pcd.stem + '.json')
            if jp.exists():
                pairs.append((pcd, jp))
    return pairs[::stride]


def label_ground_z(json_paths, sample=60):
    """Ground height = median box-bottom (centre_z - h/2) over vehicles in a
    sample of frames; robust to the gantry geometry that defeats modal-z."""
    bottoms = []
    for jp in json_paths[:sample]:
        for b in read_openlabel_boxes(jp):
            if b['label'] in ('car', 'truck', 'bus'):
                bottoms.append(b['xyz'][2] - b['lwh'][2] / 2)
    return float(np.median(bottoms)) if bottoms else -1.84


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--subsets', default='s01,s02,s03,s04')
    ap.add_argument('--sensor', default=SENSOR)
    ap.add_argument('--out-dir', default='outputs/tumtraf_eval')
    ap.add_argument('--config', default='/mnt/T9/parham/lidar-pilot/mmdetection3d/configs/'
                    'centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py')
    ap.add_argument('--checkpoint', default='/mnt/T9/parham/lidar-pilot/checkpoints/'
                    'centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth')
    ap.add_argument('--det-classes', choices=['nuscenes', 'lumpi'], default='nuscenes')
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--max-range', type=float, default=50.0)
    ap.add_argument('--z-shift', default='auto')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--tag', default='pretrained_tumtraf')
    args = ap.parse_args()

    det_class_list = (NUS_CLASSES if args.det_classes == 'nuscenes'
                      else LUMPI_TRAIN_CLASSES)
    scored = SHARED + (['scooter'] if args.det_classes == 'lumpi' else [])

    pairs = frame_pairs(args.data_root, args.subsets.split(','),
                        args.sensor, args.stride)
    if not pairs:
        raise SystemExit('no (pcd, json) frame pairs found')

    if args.z_shift == 'auto':
        ground = label_ground_z([jp for _, jp in pairs])
        z_shift = NUS_GROUND_Z - ground
    else:
        z_shift = float(args.z_shift)
    print(f'{len(pairs)} frames | {args.sensor} | z-shift {z_shift:+.2f} m | '
          f'range <= {args.max_range} m | score >= {args.score_thr} | '
          f'head={args.det_classes}')

    from mmdet3d.apis import LidarDet3DInferencer
    inf = LidarDet3DInferencer(model=args.config, weights=args.checkpoint,
                               device='cuda:0')
    inf.show_progress = False

    totals = {c: dict(tp=0, fp=0, fn=0) for c in scored}
    n_other_total = 0
    per_frame_dets = {}

    for k, (pcd_path, json_path) in enumerate(pairs):
        pts = read_pcd(pcd_path, z_shift)
        keep = np.linalg.norm(pts[:, :2], axis=1) < args.max_range + 10
        res = inf(dict(points=pts[keep]), no_save_vis=True)
        p = res['predictions'][0]
        boxes = np.asarray(p['bboxes_3d'], float).reshape(-1, 9)
        scores = np.asarray(p['scores_3d'], float)
        cls_ids = np.asarray(p['labels_3d'], int)

        dets = [(det_class_list[c], b[0], b[1], s)
                for b, s, c in zip(boxes, scores, cls_ids)
                if s >= args.score_thr and np.hypot(b[0], b[1]) <= args.max_range]
        gts = [(b['label'], b['xyz'][0], b['xyz'][1])
               for b in read_openlabel_boxes(json_path)
               if b['label'] != 'unknown' and np.hypot(*b['xyz'][:2]) <= args.max_range]
        per_frame_dets[k] = dets

        stats, n_other = match_frame(dets, gts, scored)
        n_other_total += n_other
        for c in scored:
            for kk in ('tp', 'fp', 'fn'):
                totals[c][kk] += stats[c][kk]
        if k % 300 == 0:
            print(f'  {k}/{len(pairs)} frames', flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'detections_{args.tag}.pkl', 'wb') as f:
        pickle.dump(dict(dets=per_frame_dets, z_shift=z_shift), f)

    print(f'\n{"class":<12}{"GT":>7}{"TP":>7}{"FP":>7}{"FN":>7}'
          f'{"precision":>11}{"recall":>9}{"F1":>7}')
    rows = []
    for c in scored:
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
    write_header = not summary.exists()
    with open(summary, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['tag', 'class', 'gt', 'tp', 'fp', 'fn',
                        'precision', 'recall', 'f1'])
        w.writerows(rows)
    print(f'appended to {summary}')


if __name__ == '__main__':
    main()
