#!/usr/bin/env python
"""CenterPoint detection + tracking + speed validation on nuScenes-mini.

Runs on the GPU box. For every keyframe of the selected scenes:
  1. accumulate the standard 10-sweep point cloud (devkit, lidar frame),
  2. run CenterPoint-pillar (boxes + velocity head),
  3. transform detections to the global (map) frame,
  4. update the Kalman tracker (velocity-head seeding),
  5. match confirmed tracks to GT annotations and record speed triplets
     (Kalman speed, velocity-head speed, devkit GT speed).

Outputs under --out-dir:
  speeds.csv                per-frame matched speed comparisons
  tracks_<scene>.pkl        list[Trajectory] per scene (global frame)
  <scene>.rrd               rerun recording (when --rrd)

Usage:
  python run_nuscenes_pipeline.py --scenes scene-0103 --rrd
  python run_nuscenes_pipeline.py --all
"""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import mini_train, mini_val
from pyquaternion import Quaternion

from lidar_pilot.tracking import Tracker3D

DET_CLASSES = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
# Static street furniture is not a road user; do not track it.
TRACKED = {'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
           'bicycle', 'motorcycle', 'pedestrian'}

GT_CATEGORY_MAP = {
    'vehicle.car': 'car', 'vehicle.truck': 'truck', 'vehicle.trailer': 'trailer',
    'vehicle.bus.bendy': 'bus', 'vehicle.bus.rigid': 'bus',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.bicycle': 'bicycle', 'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian', 'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
}

CLASS_COLORS = {
    'car': (66, 135, 245), 'truck': (245, 167, 66), 'trailer': (181, 101, 29),
    'bus': (245, 66, 221), 'construction_vehicle': (130, 130, 130),
    'bicycle': (66, 245, 96), 'motorcycle': (32, 196, 160),
    'pedestrian': (245, 66, 66),
}


def quaternion_yaw(q: Quaternion) -> float:
    v = q.rotation_matrix @ np.array([1.0, 0.0, 0.0])
    return float(np.arctan2(v[1], v[0]))


def lidar_to_global(nusc, sd_token):
    """Rotation matrix, translation and yaw of the lidar->global transform."""
    sd = nusc.get('sample_data', sd_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])
    q = Quaternion(ep['rotation']) * Quaternion(cs['rotation'])
    t = (np.array(ep['translation'])
         + Quaternion(ep['rotation']).rotation_matrix @ np.array(cs['translation']))
    return q.rotation_matrix, t, quaternion_yaw(q)


def accumulate_sweeps(nusc, sample, nsweeps=10):
    """(N, 5) [x, y, z, intensity, dt] in the keyframe lidar frame."""
    pc, times = LidarPointCloud.from_file_multisweep(
        nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)
    return np.vstack([pc.points[:4], times]).T.astype(np.float32)


def gt_objects(nusc, sample):
    """GT road users of this keyframe: (class, center_xy global, speed)."""
    out = []
    for tok in sample['anns']:
        ann = nusc.get('sample_annotation', tok)
        cls = GT_CATEGORY_MAP.get(ann['category_name'])
        if cls is None:
            continue
        vel = nusc.box_velocity(tok)
        if np.any(np.isnan(vel)):
            continue
        out.append((cls, np.array(ann['translation'][:2]), float(np.linalg.norm(vel[:2]))))
    return out


class RerunLogger:
    """Thin wrapper so the pipeline never depends on rerun API details."""

    def __init__(self, path: Path):
        import rerun as rr
        self.rr = rr
        rr.init('lidar_pilot', spawn=False)
        rr.save(str(path))
        self.trails: dict[int, list] = {}

    def set_time(self, t: float) -> None:
        rr = self.rr
        if hasattr(rr, 'set_time_seconds'):
            rr.set_time_seconds('t', t)
        else:  # rerun >= 0.23
            rr.set_time('t', duration=t)

    def log_frame(self, points_global, tracks):
        rr = self.rr
        rr.log('world/points', rr.Points3D(points_global[::4, :3],
                                           radii=0.03, colors=(160, 160, 170)))
        centers, half_sizes, quats, colors, labels = [], [], [], [], []
        for tr in tracks:
            x, y, z, yaw, l, w, h = tr.kf.box
            centers.append([x, y, z + h / 2])
            half_sizes.append([l / 2, w / 2, h / 2])
            quats.append(rr.Quaternion(xyzw=[0, 0, np.sin(yaw / 2), np.cos(yaw / 2)]))
            colors.append(CLASS_COLORS.get(tr.label, (200, 200, 200)))
            labels.append(f'#{tr.track_id} {tr.label} '
                          f'{np.linalg.norm(tr.kf.velocity[:2]) * 3.6:.0f} km/h')
            self.trails.setdefault(tr.track_id, []).append([x, y, z + h / 2])
        if centers:
            rr.log('world/tracks', rr.Boxes3D(centers=centers, half_sizes=half_sizes,
                                              quaternions=quats, colors=colors,
                                              labels=labels))
        strips = [np.asarray(v) for v in self.trails.values() if len(v) >= 2]
        if strips:
            rr.log('world/trails', rr.LineStrips3D(strips, radii=0.06,
                                                   colors=(255, 230, 60)))


# Per-class association gates (metres) for 2 Hz keyframes: roughly the
# distance the class can plausibly travel in 0.5 s plus localization slack.
# Tight VRU gates stop tracks hopping between distinct parked bikes/people.
MATCH_GATES = {'car': 4.0, 'truck': 4.5, 'bus': 5.5, 'trailer': 5.5,
               'construction_vehicle': 4.0, 'pedestrian': 2.0,
               'bicycle': 2.5, 'motorcycle': 3.0, 'default': 4.0}


def run_scene(nusc, inferencer, scene, out_dir: Path, rrd: bool,
              score_thr: float, csv_writer, split_name: str):
    sample = nusc.get('sample', scene['first_sample_token'])
    tracker = Tracker3D(max_match_distance=MATCH_GATES, max_age=2, min_hits=2,
                        min_score=score_thr)
    logger = RerunLogger(out_dir / f"{scene['name']}.rrd") if rrd else None
    t0 = sample['timestamp'] / 1e6
    n_frames = 0
    wall0 = time.perf_counter()

    while True:
        t = sample['timestamp'] / 1e6 - t0
        pts = accumulate_sweeps(nusc, sample)
        res = inferencer(dict(points=pts), no_save_vis=True)
        p = res['predictions'][0]
        boxes9 = np.asarray(p['bboxes_3d'], dtype=float).reshape(-1, 9)
        scores = np.asarray(p['scores_3d'], dtype=float)
        labels = [DET_CLASSES[i] for i in p['labels_3d']]

        keep = np.array([l in TRACKED for l in labels], dtype=bool)
        boxes9, scores = boxes9[keep], scores[keep]
        labels = [l for l, k in zip(labels, keep) if k]

        # lidar frame -> global frame
        R, T, yaw_lg = lidar_to_global(nusc, sample['data']['LIDAR_TOP'])
        centers_g = (R @ boxes9[:, :3].T).T + T
        yaws_g = boxes9[:, 6] + yaw_lg
        vels_g = (R[:2, :2] @ boxes9[:, 7:9].T).T
        # tracker layout: [x, y, z, yaw, l, w, h]
        tboxes = np.column_stack([centers_g, yaws_g,
                                  boxes9[:, 3], boxes9[:, 4], boxes9[:, 5]])
        active = tracker.step(tboxes, scores, labels, t, velocities=vels_g)

        # speed validation against GT
        gts = gt_objects(nusc, sample)
        for tr in active:
            pos = tr.kf.box[:2]
            best, best_d = None, 2.0
            for cls, gxy, gspeed in gts:
                if cls != tr.label:
                    continue
                d = float(np.linalg.norm(gxy - pos))
                if d < best_d:
                    best, best_d = gspeed, d
            if best is not None:
                head_speed = (float(np.linalg.norm(tr.last_det_velocity))
                              if tr.last_det_velocity is not None else '')
                csv_writer.writerow([scene['name'], split_name, f'{t:.2f}',
                                     tr.track_id, tr.label,
                                     f'{np.linalg.norm(tr.kf.velocity[:2]):.3f}',
                                     f'{head_speed:.3f}' if head_speed != '' else '',
                                     f'{best:.3f}'])

        if logger:
            logger.set_time(t)
            pts_g = (R @ pts[pts[:, 4] == 0.0][:, :3].T).T + T  # keyframe sweep only
            logger.log_frame(pts_g, active)

        n_frames += 1
        if not sample['next']:
            break
        sample = nusc.get('sample', sample['next'])

    dt = (time.perf_counter() - wall0) / max(n_frames, 1)
    trajs = tracker.trajectories
    with open(out_dir / f"tracks_{scene['name']}.pkl", 'wb') as f:
        pickle.dump(trajs, f)
    print(f"{scene['name']}: {n_frames} keyframes, {len(trajs)} confirmed tracks, "
          f"{dt * 1000:.0f} ms/frame")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/mnt/T9/parham/lidar-pilot/data/nuscenes')
    ap.add_argument('--out-dir', default='/mnt/T9/parham/lidar-pilot/outputs/nuscenes_mini')
    ap.add_argument('--config', default='/mnt/T9/parham/lidar-pilot/mmdetection3d/configs/'
                    'centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py')
    ap.add_argument('--checkpoint', default='/mnt/T9/parham/lidar-pilot/checkpoints/'
                    'centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth')
    ap.add_argument('--scenes', nargs='*', default=None,
                    help='scene names; default mini_val; --all for all 10')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--rrd', action='store_true', help='write rerun recordings')
    ap.add_argument('--score-thr', type=float, default=0.35)
    args = ap.parse_args()

    from mmdet3d.apis import LidarDet3DInferencer
    inferencer = LidarDet3DInferencer(model=args.config, weights=args.checkpoint,
                                      device='cuda:0')
    inferencer.show_progress = False

    nusc = NuScenes(version='v1.0-mini', dataroot=args.data_root, verbose=False)
    wanted = (None if args.all else (set(args.scenes) if args.scenes else set(mini_val)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'speeds.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['scene', 'split', 't', 'track_id', 'class',
                             'speed_kf', 'speed_head', 'speed_gt'])
        for scene in nusc.scene:
            if wanted is not None and scene['name'] not in wanted:
                continue
            split = 'val' if scene['name'] in mini_val else 'train'
            run_scene(nusc, inferencer, scene, out_dir, args.rrd,
                      args.score_thr, writer, split)


if __name__ == '__main__':
    main()
