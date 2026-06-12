#!/usr/bin/env python
"""Hard-braking recovery experiments: evaluate and sweep the e2e pipeline.

The label-free pipeline reproduces ground-truth TTC and PET conflict counts
almost exactly but recovers only ~half of the hard-braking events (HBE).
This script measures that gap properly and sweeps the levers that should
close it (track survival, association gating, Kalman responsiveness,
braking-estimator settings), each re-run from a detection cache in CPU
minutes instead of GPU passes.

Beyond the dashboard's count ratio and per-cell spatial correlation, it
matches events one-to-one (Hungarian on time+location) for precision and
recall, and attributes every missed GT event to a cause:
  no_track        no pipeline track near the braking vehicle at that moment
  filtered_track  a track was there but failed the moving-road-user or
                  class-plausibility filters
  under_threshold a valid track was there but its estimated deceleration
                  never crossed the -3 m/s^2 definition

Usage:
  # diagnose two existing conflict sets
  python scripts/sweep_hbe_recovery.py eval \
      --gt-conflicts outputs/lumpi/conflicts.csv \
      --pipe-conflicts outputs/lumpi_e2e/conflicts.csv \
      --pipe-tracks outputs/lumpi_e2e/tracks_Measurement5_e2e.pkl

  # sweep configurations from a detection cache
  python scripts/sweep_hbe_recovery.py sweep \
      --detections outputs/lumpi_e2e/detections_Measurement5.pkl \
      --gt-conflicts outputs/lumpi/conflicts.csv \
      --out-dir outputs/hbe_sweep --jobs 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# The tracker's linear algebra is all tiny (10x10 KF updates); threaded
# BLAS only adds dispatch overhead, and with several sweep workers the
# spinning BLAS pools oversubscribe every core. Must be set before numpy
# loads, and applies to spawned workers re-importing this module too.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from run_lumpi_pipeline import FPS, LUMPI_TRAIN_CLASSES, MATCH_GATES  # noqa: E402

from lidar_pilot.conflicts import (mine_conflicts,  # noqa: E402
                                   select_moving_road_users)
from lidar_pilot.kinematics import longitudinal_accel  # noqa: E402
from lidar_pilot.tracking import Tracker3D  # noqa: E402

R_MAX = 45.0      # sensing envelope, as in plot_e2e_validation.py
CELL = 8.0        # agreement-map cell size, as in plot_e2e_validation.py
MATCH_DT = 2.0    # s: max onset-time offset for one-to-one event matching
MATCH_DXY = 6.0   # m: max location offset for one-to-one event matching


# ---------------------------------------------------------------- events --

def load_events(path) -> list[dict]:
    out = []
    for r in csv.DictReader(open(path)):
        if not r['x']:
            continue
        out.append(dict(metric=r['metric'], value=float(r['value']),
                        t=float(r['t']), x=float(r['x']), y=float(r['y']),
                        id_a=r['id_a'], class_a=r['class_a']))
    return out


def rows_to_events(rows) -> list[dict]:
    return [dict(metric=r[1], value=float(r[2]), t=float(r[3]),
                 x=float(r[8]), y=float(r[9]), id_a=str(r[4]), class_a=r[5])
            for r in rows]


def in_envelope(events, r_max=R_MAX):
    return [e for e in events if np.hypot(e['x'], e['y']) <= r_max]


def ratio_and_spatial_r(gt_ev, pipe_ev) -> tuple[float, float]:
    """Count ratio and per-cell correlation, exactly as the dashboard."""
    lim = R_MAX + 3
    bins = np.arange(-lim, lim + CELL, CELL)
    maps = []
    for evs in (gt_ev, pipe_ev):
        pts = np.array([(e['x'], e['y']) for e in evs]).reshape(-1, 2)
        h, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[bins, bins])
        maps.append(h.ravel())
    m = (maps[0] > 0) | (maps[1] > 0)
    r = float(np.corrcoef(maps[0][m], maps[1][m])[0, 1]) if m.any() else float('nan')
    return len(pipe_ev) / max(len(gt_ev), 1), r


def match_events(gt_ev, pipe_ev, trajs_by_id=None,
                 dt_tol=MATCH_DT, dxy_tol=MATCH_DXY):
    """One-to-one matching of braking events. Returns
    (matched index pairs, precision, recall, f1).

    With trajs_by_id (pipeline track id -> Trajectory) the test is
    same-vehicle: the pipeline event's own track must pass within dxy_tol
    of the GT vehicle position at GT onset time. This is robust to the
    onset-time shift a damped estimator introduces, which moves the event
    *position* tens of metres at driving speed. Without tracks it falls
    back to comparing event positions directly.
    """
    n_g, n_p = len(gt_ev), len(pipe_ev)
    if n_g == 0 or n_p == 0:
        return [], 0.0, 0.0, 0.0
    cost = np.full((n_g, n_p), 1e6)
    for i, g in enumerate(gt_ev):
        for j, p in enumerate(pipe_ev):
            dt = abs(g['t'] - p['t'])
            if dt > dt_tol:
                continue
            tr = trajs_by_id.get(p['id_a']) if trajs_by_id else None
            if tr is not None:
                if not (tr.t[0] - 1.0 <= g['t'] <= tr.t[-1] + 1.0):
                    continue
                x = np.interp(g['t'], tr.t, tr.xy[:, 0])
                y = np.interp(g['t'], tr.t, tr.xy[:, 1])
                d = float(np.hypot(x - g['x'], y - g['y']))
            else:
                d = float(np.hypot(g['x'] - p['x'], g['y'] - p['y']))
            if d > dxy_tol:
                continue
            cost[i, j] = (dt / dt_tol) ** 2 + (d / dxy_tol) ** 2
    rows, cols = linear_sum_assignment(cost)
    pairs = [(i, j) for i, j in zip(rows, cols) if cost[i, j] < 1e6]
    prec = len(pairs) / n_p
    rec = len(pairs) / n_g
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return pairs, prec, rec, f1


def attribute_misses(missed_gt, all_trajs, moving_trajs,
                     dxy_tol=MATCH_DXY, window=MATCH_DT):
    """Why was each missed GT braking event not recovered?

    Misses with a valid track nearby are sub-bucketed by that track's
    estimated peak deceleration around the event: 'damped' never crossed
    -3 m/s^2 (estimator too smooth / filter lag), 'noise_spike' crossed
    the -12 m/s^2 plausibility bound (tracking glitches, discarded as
    unphysical), 'crossed_unmatched' crossed -3 plausibly but produced no
    matched event (duration/start-speed filters, or matched elsewhere).
    """
    moving_ids = {tr.track_id for tr in moving_trajs}
    buckets = dict(no_track=0, filtered_track=0, damped=0, noise_spike=0,
                   crossed_unmatched=0)
    for ev in missed_gt:
        t0 = ev['t']
        best = None  # (dist, traj)
        for tr in all_trajs:
            if not (tr.t[0] - window <= t0 <= tr.t[-1] + window):
                continue
            x = np.interp(t0, tr.t, tr.xy[:, 0])
            y = np.interp(t0, tr.t, tr.xy[:, 1])
            d = float(np.hypot(x - ev['x'], y - ev['y']))
            if d <= dxy_tol and (best is None or d < best[0]):
                best = (d, tr)
        if best is None:
            buckets['no_track'] += 1
        elif best[1].track_id not in moving_ids:
            buckets['filtered_track'] += 1
        else:
            tr = best[1]
            sel = (tr.t >= t0 - window) & (tr.t <= t0 + window)
            min_a = 0.0
            if sel.sum() >= 3:
                a = longitudinal_accel(tr, smooth_window=3)
                min_a = float(a[sel].min())
            if min_a > -3.0:
                buckets['damped'] += 1
            elif min_a < -12.0:
                buckets['noise_spike'] += 1
            else:
                buckets['crossed_unmatched'] += 1
    return buckets


# ----------------------------------------------------------------- sweep --

@dataclass(frozen=True)
class Config:
    name: str
    max_age: int = 5
    min_hits: int = 3
    min_score: float = 0.35
    gate_growth: float = 0.0
    accel_std: float = 3.0
    record: str = 'posterior'
    max_range: float = 50.0
    smooth: int = 3          # HBE estimator: moving-average window
    min_dur: float = 0.2     # HBE estimator: sustained duration
    speed_source: str = 'positions'
    despike: bool = False    # HBE estimator: median-filter speed first


DEFAULT_SWEEP = [
    Config('baseline'),
    # track survival / association levers
    Config('age10', max_age=10),
    Config('age10_gg03', max_age=10, gate_growth=0.3),
    Config('age15_gg03', max_age=15, gate_growth=0.3),
    Config('score025', min_score=0.25),
    Config('score020', min_score=0.20),
    # Kalman responsiveness levers
    Config('accel6', accel_std=6.0),
    Config('accel9', accel_std=9.0),
    # bypass the filter for the recorded positions entirely
    Config('det_s3', record='detection'),
    Config('det_s5', record='detection', smooth=5),
    Config('det_s5_accel6', record='detection', smooth=5, accel_std=6.0),
    # estimator-side variants (preview on frozen tracks said: minor)
    Config('smooth1', smooth=1),
    Config('smooth5', smooth=5),
    Config('kfvel', speed_source='kf_velocity'),
    Config('mindur01', min_dur=0.1),
    Config('range60', max_range=60.0),
    # early combos of the likely winners
    Config('accel6_age10_gg03', accel_std=6.0, max_age=10, gate_growth=0.3),
    Config('det_s5_age10_gg03', record='detection', smooth=5,
           max_age=10, gate_growth=0.3),
]


def retrack(frames, cfg: Config):
    tracker = Tracker3D(max_match_distance=MATCH_GATES, max_age=cfg.max_age,
                        min_hits=cfg.min_hits, min_score=cfg.min_score,
                        gate_growth=cfg.gate_growth,
                        kf_params=dict(accel_std=cfg.accel_std),
                        record_source=cfg.record)
    for fr in frames:
        boxes9, scores, labels = fr['boxes9'], fr['scores'], fr['labels']
        rng_ok = np.linalg.norm(boxes9[:, :2].astype(float), axis=1) <= cfg.max_range
        boxes9 = boxes9[rng_ok].astype(float)
        scores = scores[rng_ok].astype(float)
        names = [LUMPI_TRAIN_CLASSES[c] for c in labels[rng_ok]]
        tboxes = np.column_stack([boxes9[:, :3], boxes9[:, 6],
                                  boxes9[:, 3], boxes9[:, 4], boxes9[:, 5]])
        tracker.step(tboxes, scores, names, fr['fidx'] / FPS,
                     velocities=boxes9[:, 7:9])
    return tracker.trajectories


def evaluate(gt_events, pipe_events, all_trajs, moving_trajs) -> dict:
    """All comparison numbers for one pipeline conflict set."""
    res = {}
    gt_env = {m: in_envelope([e for e in gt_events if e['metric'] == m])
              for m in ('TTC', 'PET', 'HBE')}
    pipe_env = {m: in_envelope([e for e in pipe_events if e['metric'] == m])
                for m in ('TTC', 'PET', 'HBE')}
    for m in ('TTC', 'PET', 'HBE'):
        ratio, r = ratio_and_spatial_r(gt_env[m], pipe_env[m])
        res[f'{m.lower()}_ratio'] = round(ratio, 3)
        res[f'{m.lower()}_r'] = round(r, 3)
    trajs_by_id = None
    if all_trajs is not None:
        trajs_by_id = {str(tr.track_id): tr for tr in all_trajs}
    pairs, prec, rec, f1 = match_events(gt_env['HBE'], pipe_env['HBE'],
                                        trajs_by_id)
    res.update(hbe_n_gt=len(gt_env['HBE']), hbe_n_pipe=len(pipe_env['HBE']),
               hbe_precision=round(prec, 3), hbe_recall=round(rec, 3),
               hbe_f1=round(f1, 3))
    if all_trajs is not None:
        matched_gt = {i for i, _ in pairs}
        missed = [e for i, e in enumerate(gt_env['HBE']) if i not in matched_gt]
        buckets = attribute_misses(missed, all_trajs, moving_trajs)
        res.update({f'miss_{k}': v for k, v in buckets.items()})
    return res


_FRAMES = None  # per-worker detection cache


def _init_worker(cache_path):
    global _FRAMES
    with open(cache_path, 'rb') as f:
        _FRAMES = pickle.load(f)['frames']


def _run_config(cfg: Config, gt_path: str, out_dir: str | None) -> dict:
    t0 = time.perf_counter()
    trajs = retrack(_FRAMES, cfg)
    rows, st = mine_conflicts(trajs, cfg.name,
                              hbe_smooth_window=cfg.smooth,
                              hbe_min_duration=cfg.min_dur,
                              hbe_speed_source=cfg.speed_source,
                              hbe_despike=cfg.despike)
    gt_events = load_events(gt_path)
    res = dict(asdict(cfg))
    res.update(evaluate(gt_events, rows_to_events(rows),
                        trajs, st.moving_tracks))
    durs = np.array([tr.duration for tr in st.moving_tracks])
    res.update(n_tracks=len(trajs), n_moving=st.n_moving,
               moving_dur_med=round(float(np.median(durs)), 1) if durs.size else 0.0,
               runtime_s=round(time.perf_counter() - t0, 1))
    if out_dir:
        with open(Path(out_dir) / f'conflicts_{cfg.name}.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['scene', 'metric', 'value', 't', 'id_a', 'class_a',
                        'id_b', 'class_b', 'x', 'y'])
            w.writerows(rows)
    return res


TABLE_COLS = ['name', 'hbe_ratio', 'hbe_r', 'hbe_recall', 'hbe_precision',
              'hbe_f1', 'miss_no_track', 'miss_filtered_track', 'miss_damped',
              'miss_noise_spike', 'miss_crossed_unmatched',
              'ttc_ratio', 'ttc_r', 'pet_ratio', 'pet_r',
              'n_moving', 'moving_dur_med', 'runtime_s']


def print_table(results):
    widths = {c: max(len(c), *(len(str(r.get(c, ''))) for r in results))
              for c in TABLE_COLS}
    print('  '.join(c.ljust(widths[c]) for c in TABLE_COLS))
    for r in results:
        print('  '.join(str(r.get(c, '')).ljust(widths[c]) for c in TABLE_COLS))


def cmd_sweep(args):
    if args.configs:
        cfgs = [Config(**d) for d in json.load(open(args.configs))]
    else:
        cfgs = DEFAULT_SWEEP
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 initializer=_init_worker,
                                 initargs=(args.detections,)) as ex:
            futs = {ex.submit(_run_config, c, args.gt_conflicts,
                              str(out_dir)): c for c in cfgs}
            for f in as_completed(futs):
                res = f.result()
                results.append(res)
                print(f'[{len(results)}/{len(cfgs)}] {res["name"]}: '
                      f'HBE ratio {res["hbe_ratio"]} recall {res["hbe_recall"]} '
                      f'precision {res["hbe_precision"]} ({res["runtime_s"]}s)',
                      flush=True)
    else:
        _init_worker(args.detections)
        for c in cfgs:
            res = _run_config(c, args.gt_conflicts, str(out_dir))
            results.append(res)
            print(f'[{len(results)}/{len(cfgs)}] {res["name"]}: '
                  f'HBE ratio {res["hbe_ratio"]} recall {res["hbe_recall"]} '
                  f'precision {res["hbe_precision"]} ({res["runtime_s"]}s)',
                  flush=True)

    order = {c.name: k for k, c in enumerate(cfgs)}
    results.sort(key=lambda r: order[r['name']])
    cols = list(results[0].keys())
    with open(out_dir / 'results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)
    print(f'\nwrote {out_dir / "results.csv"}\n')
    print_table(results)


def cmd_eval(args):
    gt_events = load_events(args.gt_conflicts)
    pipe_events = load_events(args.pipe_conflicts)
    all_trajs = moving = None
    if args.pipe_tracks:
        all_trajs = pickle.load(open(args.pipe_tracks, 'rb'))
        moving, _ = select_moving_road_users(all_trajs)
    res = evaluate(gt_events, pipe_events, all_trajs, moving)
    for k, v in res.items():
        print(f'{k:28s} {v}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    ev = sub.add_parser('eval', help='diagnose two existing conflict csvs')
    ev.add_argument('--gt-conflicts', required=True)
    ev.add_argument('--pipe-conflicts', required=True)
    ev.add_argument('--pipe-tracks', help='tracks pkl for miss attribution')
    ev.set_defaults(func=cmd_eval)

    sw = sub.add_parser('sweep', help='sweep configs from a detection cache')
    sw.add_argument('--detections', required=True)
    sw.add_argument('--gt-conflicts', required=True)
    sw.add_argument('--out-dir', required=True)
    sw.add_argument('--configs', help='JSON list of Config kwargs (else default sweep)')
    sw.add_argument('--jobs', type=int, default=1)
    sw.set_defaults(func=cmd_sweep)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
