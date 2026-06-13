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


def build(data_root, subs, sensor, bins_dir, z_shift, max_range, start_idx):
    """Convert a list of subsets; returns (data_list, next sample index)."""
    import json
    data_list = []
    sidx = start_idx
    for sub in subs:
        pairs = subset_pairs(data_root, sub, sensor)
        vel = frame_velocities([jp for _, jp in pairs])
        for k, (pcd_path, jp) in enumerate(pairs):
            bin_path = bins_dir / f'{sidx:06d}.bin'
            if not bin_path.exists():
                pts = read_pcd(pcd_path, z_shift)
                pts[np.linalg.norm(pts[:, :2], axis=1) <= max_range + 10].astype(
                    np.float32).tofile(bin_path)
            doc = json.load(open(jp))
            fr = next(iter(doc['openlabel']['frames'].values()))
            instances = []
            for uuid, obj in (fr.get('objects', {}) or {}).items():
                cub = obj.get('object_data', {}).get('cuboid')
                if not cub:
                    continue
                from lidar_pilot.io.tumtraf import quat_to_yaw, TUMTRAF_CLASS_MAP
                raw = (obj['object_data'].get('type') or 'OTHER').upper()
                name = TUMTRAF_CLASS_MAP.get(raw, 'unknown')
                if name not in CLASSES:
                    continue
                v = cub['val']
                x, y, zc = v[0], v[1], v[2]
                if np.hypot(x, y) > max_range:
                    continue
                l, w, h = v[7], v[8], v[9]
                vx, vy = vel.get(k, {}).get(uuid, (0.0, 0.0))
                if not (np.isfinite(vx) and np.isfinite(vy)):
                    vx, vy = 0.0, 0.0
                instances.append(dict(
                    bbox_3d=[float(x), float(y), float(zc - h / 2 + z_shift),
                             float(l), float(w), float(h),
                             float(quat_to_yaw(*v[3:7])), vx, vy],
                    bbox_label_3d=CLASSES.index(name),
                ))
            data_list.append(dict(
                sample_idx=sidx,
                lidar_points=dict(lidar_path=f'{sidx:06d}.bin', num_pts_feats=5),
                instances=instances))
            sidx += 1
    return data_list, sidx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--sensor', default=SENSOR)
    ap.add_argument('--train-subsets', default='s01,s02,s03')
    ap.add_argument('--val-subsets', default='s04')
    ap.add_argument('--max-range', type=float, default=54.0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    bins_dir = out / 'bins'
    bins_dir.mkdir(parents=True, exist_ok=True)

    all_json = []
    for sub in args.train_subsets.split(',') + args.val_subsets.split(','):
        all_json += [jp for _, jp in subset_pairs(args.data_root, sub, args.sensor)]
    ground = label_ground_z(all_json)
    z_shift = NUS_GROUND_Z - ground
    print(f'z-shift {z_shift:+.2f} m (label ground {ground:.2f})')

    metainfo = dict(classes=CLASSES, z_shift=z_shift)
    idx = 0
    for name, subs in (('train', args.train_subsets.split(',')),
                       ('val', args.val_subsets.split(','))):
        data_list, idx = build(args.data_root, subs, args.sensor, bins_dir,
                               z_shift, args.max_range, idx)
        n_inst = sum(len(d['instances']) for d in data_list)
        with open(out / f'tumtraf_infos_{name}.pkl', 'wb') as f:
            pickle.dump(dict(metainfo=metainfo, data_list=data_list), f)
        print(f'{name}: {len(data_list)} frames, {n_inst} instances '
              f'({",".join(subs)}) -> tumtraf_infos_{name}.pkl')


if __name__ == '__main__':
    main()
