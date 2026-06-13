#!/usr/bin/env python
"""Convert TUMTraf Intersection into MMDetection3D training format.

Mirrors prepare_lumpi_training.py but for TUMTraf's OpenLABEL labels and
.pcd clouds (read via io/tumtraf.py), writing the SAME info-pickle format
the LumpiDataset plugin consumes, so the existing dataset/config machinery
is reused unchanged. Boxes are 9-DoF [x, y, z_bottom, l, w, h, yaw, vx, vy]
in the z-shifted south-LiDAR frame; per-object velocity is differentiated
from the OpenLABEL track (uuid) positions over time.

Train/val split is by SUBSET (s01-s03 train, s04 val by default): the
subsets are separate recordings, so this is a clean split with no temporal
leakage. The 7-class head is kept (TUMTraf has no scooter; that slot just
gets no examples, as motorcycle did on LUMPI).

Usage (GPU box):
  python prepare_tumtraf_training.py --data-root data/tumtraf \
      --out-dir data/tumtraf/kit
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from lidar_pilot.io.tumtraf import read_openlabel_boxes, read_pcd

CLASSES = ['car', 'truck', 'bus', 'pedestrian', 'bicycle', 'motorcycle', 'scooter']
NUS_GROUND_Z = -1.84
SENSOR = 's110_lidar_ouster_south'


def label_ground_z(json_paths, sample=80):
    bottoms = []
    for jp in json_paths[:sample]:
        for b in read_openlabel_boxes(jp):
            if b['label'] in ('car', 'truck', 'bus'):
                bottoms.append(b['xyz'][2] - b['lwh'][2] / 2)
    return float(np.median(bottoms)) if bottoms else NUS_GROUND_Z


def subset_pairs(data_root, sub, sensor):
    base = Path(data_root) / f'a9_dataset_r02_{sub}'
    pcd_dir = base / 'point_clouds' / sensor
    lbl_dir = base / 'labels_point_clouds' / sensor
    pairs = []
    for pcd in sorted(pcd_dir.glob('*.pcd')):
        jp = lbl_dir / (pcd.stem + '.json')
        if jp.exists():
            pairs.append((pcd, jp))
    return pairs


MAX_SPEED = 40.0   # m/s (~144 km/h); clamp jitter-induced spikes


def frame_velocities(json_paths):
    """Per-(frame, uuid) ground-plane velocity from track positions.

    Returns {frame_index: {uuid: (vx, vy)}}. Differentiates each uuid's
    centre over the OpenLABEL timestamps within this subset. Duplicate
    timestamps (dt=0) would make np.gradient blow up to inf/nan and poison
    the regression loss, so they are dropped before differentiating, and
    the result is clamped to a physical speed bound.
    """
    import json
    tracks: dict[str, list] = {}
    times = []
    for k, jp in enumerate(json_paths):
        doc = json.load(open(jp))
        fr = next(iter(doc['openlabel']['frames'].values()))
        times.append(float(fr.get('frame_properties', {}).get('timestamp', k * 0.1)))
        for uuid, obj in (fr.get('objects', {}) or {}).items():
            cub = obj.get('object_data', {}).get('cuboid')
            if cub:
                tracks.setdefault(uuid, []).append((k, cub['val'][0], cub['val'][1]))
    times = np.asarray(times)
    vel: dict[int, dict] = {k: {} for k in range(len(json_paths))}
    for uuid, rows in tracks.items():
        ks = np.array([r[0] for r in rows])
        t = times[ks]
        order = np.argsort(t, kind='stable')
        ks, t = ks[order], t[order]
        xy = np.array([(rows[order[i]][1], rows[order[i]][2])
                       for i in range(len(order))])
        keep = np.concatenate([[True], np.diff(t) > 1e-6])  # drop dup timestamps
        ks, t, xy = ks[keep], t[keep], xy[keep]
        if len(t) < 2:
            continue
        vx = np.clip(np.gradient(xy[:, 0], t), -MAX_SPEED, MAX_SPEED)
        vy = np.clip(np.gradient(xy[:, 1], t), -MAX_SPEED, MAX_SPEED)
        for i, k in enumerate(ks):
            vel[int(k)][uuid] = (float(vx[i]), float(vy[i]))
    return vel


def convert_frame(pcd_path, jp, vel_k, bins_dir, z_shift, max_range, sidx):
    """Convert one (pcd, label) frame to an mmdet3d data dict + bin file."""
    import json
    from lidar_pilot.io.tumtraf import quat_to_yaw, TUMTRAF_CLASS_MAP
    bin_path = bins_dir / f'{sidx:06d}.bin'
    if not bin_path.exists():
        pts = read_pcd(pcd_path, z_shift)
        pts[np.linalg.norm(pts[:, :2], axis=1) <= max_range + 10].astype(
            np.float32).tofile(bin_path)
    fr = next(iter(json.load(open(jp))['openlabel']['frames'].values()))
    instances = []
    for uuid, obj in (fr.get('objects', {}) or {}).items():
        cub = obj.get('object_data', {}).get('cuboid')
        if not cub:
            continue
        raw = (obj['object_data'].get('type') or 'OTHER').upper()
        name = TUMTRAF_CLASS_MAP.get(raw, 'unknown')
        if name not in CLASSES:
            continue
        v = cub['val']
        x, y, zc = v[0], v[1], v[2]
        if np.hypot(x, y) > max_range:
            continue
        l, w, h = v[7], v[8], v[9]
        vx, vy = vel_k.get(uuid, (0.0, 0.0))
        if not (np.isfinite(vx) and np.isfinite(vy)):
            vx, vy = 0.0, 0.0
        instances.append(dict(
            bbox_3d=[float(x), float(y), float(zc - h / 2 + z_shift),
                     float(l), float(w), float(h),
                     float(quat_to_yaw(*v[3:7])), vx, vy],
            bbox_label_3d=CLASSES.index(name)))
    return dict(sample_idx=sidx,
                lidar_points=dict(lidar_path=f'{sidx:06d}.bin', num_pts_feats=5),
                instances=instances)


def build_subset_mode(data_root, subs, sensor, bins_dir, z_shift, max_range, start_idx):
    """Whole-subset assignment (each subset entirely train or val)."""
    data_list, sidx = [], start_idx
    for sub in subs:
        pairs = subset_pairs(data_root, sub, sensor)
        vel = frame_velocities([jp for _, jp in pairs])
        for k, (pcd_path, jp) in enumerate(pairs):
            data_list.append(convert_frame(pcd_path, jp, vel.get(k, {}),
                                           bins_dir, z_shift, max_range, sidx))
            sidx += 1
    return data_list, sidx


def build_time_mode(data_root, subs, sensor, bins_dir, z_shift, max_range,
                    train_frac, gap_frac, start_idx):
    """Per-subset temporal split: the first train_frac of EACH subset's
    frames go to train, the last (1 - train_frac - gap_frac) to val, with a
    gap between to limit leakage between near-duplicate adjacent frames.
    Velocities use the full subset. Puts each subset's VRUs in BOTH splits
    — necessary here because s03 holds nearly all motorcycles/buses/bikes."""
    train, val, val_paths, sidx = [], [], [], start_idx
    for sub in subs:
        pairs = subset_pairs(data_root, sub, sensor)
        vel = frame_velocities([jp for _, jp in pairs])
        n = len(pairs)
        train_end = int(n * train_frac)
        val_start = int(n * (train_frac + gap_frac))
        for k, (pcd_path, jp) in enumerate(pairs):
            data = convert_frame(pcd_path, jp, vel.get(k, {}),
                                 bins_dir, z_shift, max_range, sidx)
            sidx += 1
            if k < train_end:
                train.append(data)
            elif k >= val_start:
                val.append(data)
                val_paths.append(str(pcd_path))
            # else: gap frame, dropped
    return train, val, val_paths, sidx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--sensor', default=SENSOR)
    ap.add_argument('--split-mode', choices=['subset', 'time'], default='time',
                    help="'subset' = whole subsets per split; 'time' = "
                         'temporal split within each subset (VRUs in both)')
    ap.add_argument('--subsets', default='s01,s02,s03,s04',
                    help='subsets to use (time mode splits each one)')
    ap.add_argument('--train-subsets', default='s01,s02,s03', help='subset mode')
    ap.add_argument('--val-subsets', default='s04', help='subset mode')
    ap.add_argument('--train-frac', type=float, default=0.7, help='time mode')
    ap.add_argument('--gap-frac', type=float, default=0.05, help='time mode')
    ap.add_argument('--max-range', type=float, default=54.0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    bins_dir = out / 'bins'
    bins_dir.mkdir(parents=True, exist_ok=True)

    if args.split_mode == 'time':
        subs = args.subsets.split(',')
        all_json = [jp for s in subs for _, jp in subset_pairs(args.data_root, s, args.sensor)]
    else:
        subs = args.train_subsets.split(',') + args.val_subsets.split(',')
        all_json = [jp for s in subs for _, jp in subset_pairs(args.data_root, s, args.sensor)]
    z_shift = NUS_GROUND_Z - label_ground_z(all_json)
    print(f'z-shift {z_shift:+.2f} m | split-mode {args.split_mode}')

    metainfo = dict(classes=CLASSES, z_shift=z_shift)
    if args.split_mode == 'time':
        train, val, val_paths, _ = build_time_mode(
            args.data_root, args.subsets.split(','), args.sensor, bins_dir,
            z_shift, args.max_range, args.train_frac, args.gap_frac, 0)
        splits = [('train', train), ('val', val)]
        with open(out / 'val_frames.txt', 'w') as f:
            f.write('\n'.join(val_paths) + '\n')
        print(f'wrote val_frames.txt ({len(val_paths)} held-out frame paths)')
    else:
        idx = 0
        train, idx = build_subset_mode(args.data_root, args.train_subsets.split(','),
                                       args.sensor, bins_dir, z_shift, args.max_range, idx)
        val, idx = build_subset_mode(args.data_root, args.val_subsets.split(','),
                                     args.sensor, bins_dir, z_shift, args.max_range, idx)
        splits = [('train', train), ('val', val)]

    from collections import Counter
    for name, data_list in splits:
        cc = Counter()
        for d in data_list:
            for inst in d['instances']:
                cc[CLASSES[inst['bbox_label_3d']]] += 1
        with open(out / f'tumtraf_infos_{name}.pkl', 'wb') as f:
            pickle.dump(dict(metainfo=metainfo, data_list=data_list), f)
        print(f'{name}: {len(data_list)} frames | ' +
              ' '.join(f'{k}={cc[k]}' for k in CLASSES if cc[k]))


if __name__ == '__main__':
    main()
