#!/usr/bin/env python
"""Convert LUMPI frames + labels into MMDetection3D training format.

Writes (N, 5) float32 .bin point files [x, y, z+shift, intensity, 0] and
mmdet3d-v1.4 info pickles with a TIME-based train/val split: temporally
adjacent frames are near-duplicates, so a random split would leak the
val set into training. A gap of --split-gap seconds separates the two.

Boxes are stored as [x, y, z_bottom, l, w, h, yaw] (LiDARInstance3DBoxes
bottom-center convention), with the same z-shift applied as the points.

Usage (on the GPU box):
  python prepare_lumpi_training.py --lidar-dir data/lumpi/Measurement5/lidar \
      --label-csv data/lumpi/Label/Measurement5/Label.csv \
      --out-dir data/lumpi/m5_kit
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

FPS = 10
CLASSES = ['car', 'truck', 'bus', 'pedestrian', 'bicycle', 'motorcycle', 'scooter']
LUMPI_ID_TO_NAME = {0: 'pedestrian', 1: 'car', 2: 'bicycle', 3: 'motorcycle',
                    4: 'bus', 5: 'truck', 6: 'scooter'}
NUS_GROUND_Z = -1.84


def estimate_z_shift(ply_path):
    from plyfile import PlyData
    v = PlyData.read(str(ply_path))['vertex']
    pts = np.column_stack([v['x'], v['y'], v['z']])
    near = pts[np.linalg.norm(pts[:, :2], axis=1) < 50]
    hist, edges = np.histogram(near[:, 2], bins=np.arange(-10, 5, 0.2))
    return float(NUS_GROUND_Z - (edges[np.argmax(hist)] + 0.1))


def load_labels_by_frame(csv_path, max_range):
    cols = np.loadtxt(csv_path, delimiter=',', skiprows=1,
                      usecols=range(16), ndmin=2)
    frames = defaultdict(list)
    for r in cols:
        if np.hypot(r[9], r[10]) > max_range:
            continue
        frames[int(round(r[0] * FPS))].append(r)
    return frames


def convert(ply_path, bin_path, z_shift):
    from plyfile import PlyData
    v = PlyData.read(str(ply_path))['vertex']
    pts = np.column_stack([
        np.asarray(v['x'], np.float32),
        np.asarray(v['y'], np.float32),
        np.asarray(v['z'], np.float32) + z_shift,
        np.asarray(v['intensity'], np.float32),
        np.zeros(v.count, np.float32),
    ])
    pts.astype(np.float32).tofile(bin_path)


def build_split(frame_ids, lidar_dir, bins_dir, labels, z_shift):
    data_list = []
    for fidx in frame_ids:
        ply = lidar_dir / f'{fidx:06d}.ply'
        if not ply.exists() or fidx not in labels:
            continue
        bin_path = bins_dir / f'{fidx:06d}.bin'
        if not bin_path.exists():
            convert(ply, bin_path, z_shift)
        instances = []
        for r in labels[fidx]:
            name = LUMPI_ID_TO_NAME.get(int(r[7]))
            if name is None:
                continue
            x, y, zc, l, w, h, yaw = r[9], r[10], r[11], r[12], r[13], r[14], r[15]
            instances.append(dict(
                bbox_3d=[float(x), float(y), float(zc - h / 2 + z_shift),
                         float(l), float(w), float(h), float(yaw)],
                bbox_label_3d=CLASSES.index(name),
            ))
        data_list.append(dict(
            sample_idx=fidx,
            lidar_points=dict(lidar_path=f'{fidx:06d}.bin', num_pts_feats=5),
            instances=instances,
        ))
    return data_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lidar-dir', required=True)
    ap.add_argument('--label-csv', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--train-end', type=float, default=600.0, help='seconds')
    ap.add_argument('--split-gap', type=float, default=30.0, help='seconds')
    ap.add_argument('--train-stride', type=int, default=2)
    ap.add_argument('--val-stride', type=int, default=5)
    ap.add_argument('--max-range', type=float, default=54.0)
    args = ap.parse_args()

    lidar_dir = Path(args.lidar_dir)
    out = Path(args.out_dir)
    bins_dir = out / 'bins'
    bins_dir.mkdir(parents=True, exist_ok=True)

    first_ply = sorted(lidar_dir.glob('*.ply'))[0]
    last_ply = sorted(lidar_dir.glob('*.ply'))[-1]
    z_shift = estimate_z_shift(first_ply)
    n_frames = int(last_ply.stem) + 1
    print(f'{n_frames} frames available | z-shift {z_shift:+.2f} m')

    labels = load_labels_by_frame(args.label_csv, args.max_range)

    train_ids = range(0, int(args.train_end * FPS), args.train_stride)
    val_ids = range(int((args.train_end + args.split_gap) * FPS),
                    n_frames, args.val_stride)

    metainfo = dict(classes=CLASSES, z_shift=z_shift)
    for name, ids in (('train', train_ids), ('val', val_ids)):
        data_list = build_split(ids, lidar_dir, bins_dir, labels, z_shift)
        n_inst = sum(len(d['instances']) for d in data_list)
        with open(out / f'lumpi_infos_{name}.pkl', 'wb') as f:
            pickle.dump(dict(metainfo=metainfo, data_list=data_list), f)
        print(f'{name}: {len(data_list)} frames, {n_inst} instances '
              f'-> lumpi_infos_{name}.pkl')


if __name__ == '__main__':
    main()
