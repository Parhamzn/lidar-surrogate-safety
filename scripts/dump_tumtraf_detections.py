#!/usr/bin/env python
"""Run the LiDAR detector on TUMTraf frames and dump full 3D boxes.

Writes {frame_stem: [[x, y, z_centre, l, w, h, yaw, label], ...]} as JSON,
the format scripts/project_lidar_to_camera.py reads with --boxes, so the
detector's own output (not ground truth) can be drawn in the camera view.

Usage (GPU box):
  python dump_tumtraf_detections.py --pcds f1.pcd f2.pcd \
      --label-for-zshift some_label.json \
      --config .../tumtraf_finetune.py --checkpoint .../epoch_20.pth \
      --det-classes lumpi --out outputs/tumtraf_eval/fusion_dets.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_lumpi_detection import LUMPI_TRAIN_CLASSES, NUS_CLASSES, NUS_GROUND_Z  # noqa: E402
from eval_tumtraf_detection import label_ground_z  # noqa: E402

from lidar_pilot.io.tumtraf import read_pcd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pcds', nargs='+', required=True)
    ap.add_argument('--label-for-zshift', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--det-classes', choices=['nuscenes', 'lumpi'], default='lumpi')
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--max-range', type=float, default=60.0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    classes = NUS_CLASSES if args.det_classes == 'nuscenes' else LUMPI_TRAIN_CLASSES
    z_shift = NUS_GROUND_Z - label_ground_z([args.label_for_zshift])

    from mmdet3d.apis import LidarDet3DInferencer
    inf = LidarDet3DInferencer(model=args.config, weights=args.checkpoint, device='cuda:0')
    inf.show_progress = False

    out = {}
    for pcd_path in args.pcds:
        pts = read_pcd(pcd_path, z_shift)
        pts = pts[np.linalg.norm(pts[:, :2], axis=1) < args.max_range + 10]
        res = inf(dict(points=pts), no_save_vis=True)
        p = res['predictions'][0]
        b9 = np.asarray(p['bboxes_3d'], float).reshape(-1, 9)
        scr = np.asarray(p['scores_3d'], float)
        cid = np.asarray(p['labels_3d'], int)
        rows = []
        for i in range(len(b9)):
            if scr[i] < args.score_thr or np.hypot(b9[i, 0], b9[i, 1]) > args.max_range:
                continue
            # detector z is bottom-centre (z-shifted); undo shift, lift to
            # box centre so the projected wireframe matches the label frame
            zc = b9[i, 2] - z_shift + b9[i, 5] / 2
            rows.append([float(b9[i, 0]), float(b9[i, 1]), float(zc),
                         float(b9[i, 3]), float(b9[i, 4]), float(b9[i, 5]),
                         float(b9[i, 6]), classes[cid[i]]])
        out[Path(pcd_path).stem] = rows
        print(f'{Path(pcd_path).stem}: {len(rows)} boxes')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, 'w'))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
