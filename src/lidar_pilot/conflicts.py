"""Artifact-hardened surrogate-safety mining over a set of trajectories.

This is the library core behind ``scripts/extract_conflicts.py``: it mines
every plausible track pair for TTC and PET conflicts and every moving track
for hard-braking events, with filters against the label/tracking artifacts
that otherwise fake extreme conflicts (duplicate tracks, fragmented ids,
kinematically implausible class labels, parked-object jitter).

The filters that *define* the metrics (severity thresholds, plausibility
bounds, the moving-road-user test) are module constants and apply to every
data source identically, so ground-truth and pipeline conflict sets stay
comparable. Estimator details for hard braking (smoothing window, speed
source) are parameters: noisier sources may need different estimation to
measure the same physical quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from lidar_pilot.kinematics import speed
from lidar_pilot.metrics import hard_braking_events, min_pet, min_ttc

MIN_TRACK_LEN = 5          # samples
MIN_DISPLACEMENT = 3.0     # m net displacement: separates genuinely moving
                           # users from association jitter on parked objects
                           # (peak instantaneous speed is fooled by jitter)
MIN_PATH_EFFICIENCY = 0.4  # net displacement / path length: a real mover
                           # goes somewhere (~1.0; U-turn ~0.5); a track
                           # wandering across a bike rack does not (~0.2)
MIN_BRAKE_SPEED = 3.0      # m/s: braking from walking pace is not an HBE
MAX_PLAUSIBLE_DECEL = -12.0  # m/s^2: beyond emergency braking on dry
                             # asphalt; sharper events are label/tracking
                             # glitches (teleporting boxes), not physics
SEVERE_TTC = 3.0           # s, report TTC conflicts below this
SEVERE_PET = 3.0           # s, report PET events below this
# Sustained speed (90th percentile, m/s) beyond which a class label is
# physically implausible and the track's class cannot be trusted (e.g. a
# "pedestrian" at vehicle speed is a mislabeled vehicle).
MAX_CLASS_SPEED = {'pedestrian': 4.0, 'bicycle': 12.0, 'scooter': 14.0,
                   'motorcycle': 42.0}
PET_MAX_GAP = 10.0         # s: users this far apart in time cannot interact
PET_MINING_DT = 0.5        # s: PET polyline crossing runs on coarse copies
                           # (long 10 Hz roadside tracks make the segment-
                           # pair loop quadratic; 2 Hz preserves paths)


def is_moving_road_user(tr) -> bool:
    disp = np.linalg.norm(tr.xy - tr.xy[0], axis=1).max()
    if disp < MIN_DISPLACEMENT:
        return False
    path_len = float(np.linalg.norm(np.diff(tr.xy, axis=0), axis=1).sum())
    return disp / max(path_len, 1e-9) >= MIN_PATH_EFFICIENCY


def class_speed_plausible(tr) -> bool:
    limit = MAX_CLASS_SPEED.get(tr.label)
    if limit is None or len(tr) < 5:
        return True
    return float(np.percentile(speed(tr, smooth_window=5), 90)) <= limit


def same_object_pair(a, b) -> bool:
    """Two tracks that are really one physical object.

    Catches the signature of label/tracking artifacts that fake extreme
    conflicts: (1) shadow duplicates, where two ids trace near-identical
    paths simultaneously (e.g. one object labeled both pedestrian and car);
    (2) fragmentation, where one id ends and another starts at the same
    place and time. Real conflict pairs do neither: two road users cannot
    occupy the same footprint for their entire encounter.
    """
    w = a.overlap_window(b)
    if w is not None and w[1] - w[0] >= 1.0:
        grid = np.arange(w[0], w[1], 0.2)
        if grid.size >= 3:
            d = np.linalg.norm(a.resample(grid).xy - b.resample(grid).xy, axis=1)
            if float(np.median(d)) < 2.5:
                return True
    for first, second in ((a, b), (b, a)):
        gap = second.t[0] - first.t[-1]
        if -1.0 <= gap <= 2.0 and \
                float(np.linalg.norm(first.xy[-1] - second.xy[0])) < 3.0:
            return True
    return False


def bboxes_overlap(a, b, margin=3.0) -> bool:
    """Cheap spatial prefilter: inflated BEV bounding boxes must intersect."""
    a_lo, a_hi = a.xy.min(0) - margin, a.xy.max(0) + margin
    b_lo, b_hi = b.xy.min(0) - margin, b.xy.max(0) + margin
    return bool(np.all(a_hi >= b_lo) and np.all(b_hi >= a_lo))


def temporally_interacting(a, b, max_gap=PET_MAX_GAP) -> bool:
    """Time windows must overlap or lie within max_gap seconds."""
    return a.t[0] <= b.t[-1] + max_gap and b.t[0] <= a.t[-1] + max_gap


def coarse_copy(tr, dt=PET_MINING_DT):
    """Resampled copy for PET mining; native rate kept for everything else."""
    if len(tr) < 3 or np.median(np.diff(tr.t)) >= dt * 0.8:
        return tr
    grid = np.arange(tr.t[0], tr.t[-1], dt)
    return tr.resample(grid) if grid.size >= 2 else tr


def interp_position(tr, t: float) -> tuple[float, float]:
    return (float(np.interp(t, tr.t, tr.xy[:, 0])),
            float(np.interp(t, tr.t, tr.xy[:, 1])))


@dataclass
class MiningStats:
    n_tracks: int = 0
    n_moving: int = 0
    n_implausible: int = 0
    n_pairs: int = 0
    n_same_object: int = 0
    n_ttc: int = 0
    n_pet: int = 0
    n_hbe: int = 0
    moving_tracks: list = field(default_factory=list)


def select_moving_road_users(trajs) -> tuple[list, int]:
    """The miner's population: moving road users with trustworthy classes.

    Returns (kept tracks, count of moving-but-class-implausible excluded).
    """
    long_enough = [tr for tr in trajs if len(tr) >= MIN_TRACK_LEN]
    kept = [tr for tr in long_enough
            if is_moving_road_user(tr) and class_speed_plausible(tr)]
    n_implausible = sum(1 for tr in long_enough
                        if is_moving_road_user(tr)
                        and not class_speed_plausible(tr))
    return kept, n_implausible


def mine_conflicts(all_trajs, scene: str,
                   hbe_smooth_window: int = 3,
                   hbe_min_duration: float = 0.2,
                   hbe_speed_source: str = 'positions',
                   hbe_despike: bool = False,
                   ) -> tuple[list[list[str]], MiningStats]:
    """Mine one scene's trajectories for TTC, PET and hard-braking events.

    Returns CSV-ready rows [scene, metric, value, t, id_a, class_a, id_b,
    class_b, x, y] plus counters for reporting. The hbe_* parameters tune
    the hard-braking *estimator* (see module docstring); event thresholds
    themselves are module constants.
    """
    stats = MiningStats(n_tracks=len(all_trajs))
    trajs, stats.n_implausible = select_moving_road_users(all_trajs)
    stats.n_moving = len(trajs)
    stats.moving_tracks = trajs
    rows: list[list[str]] = []

    coarse = {tr.track_id: coarse_copy(tr) for tr in trajs}

    for a, b in combinations(trajs, 2):
        if not bboxes_overlap(a, b) or not temporally_interacting(a, b):
            continue
        if same_object_pair(a, b):
            stats.n_same_object += 1
            continue
        stats.n_pairs += 1
        if a.overlap_window(b) is not None:
            res = min_ttc(a, b)
            # TTC == 0 means overlapping boxes; with no real collisions
            # in the data that is a detection artifact by construction.
            if res is not None and 0.0 < res.min_ttc < SEVERE_TTC:
                stats.n_ttc += 1
                x, y = interp_position(a, res.t_at_min)
                rows.append([scene, 'TTC', f'{res.min_ttc:.2f}', f'{res.t_at_min:.1f}',
                             a.track_id, a.label, b.track_id, b.label,
                             f'{x:.2f}', f'{y:.2f}'])
        pet = min_pet(coarse[a.track_id], coarse[b.track_id])
        if pet is not None and pet.pet < SEVERE_PET:
            stats.n_pet += 1
            rows.append([scene, 'PET', f'{pet.pet:.2f}', f'{pet.t_second:.1f}',
                         pet.first_id, a.label if a.track_id == pet.first_id else b.label,
                         pet.second_id, b.label if b.track_id == pet.second_id else a.label,
                         f'{pet.conflict_point[0]:.2f}', f'{pet.conflict_point[1]:.2f}'])

    hbe_rows = mine_hbe(trajs, scene,
                        smooth_window=hbe_smooth_window,
                        min_duration=hbe_min_duration,
                        speed_source=hbe_speed_source,
                        despike=hbe_despike)
    stats.n_hbe = len(hbe_rows)
    return rows + hbe_rows, stats


def mine_hbe(moving_trajs, scene: str,
             smooth_window: int = 3,
             min_duration: float = 0.2,
             speed_source: str = 'positions',
             despike: bool = False) -> list[list[str]]:
    """Hard-braking rows for an already-selected moving-road-user set.

    Split out from mine_conflicts so braking-estimator variants can be
    re-mined without repeating the (much slower) pairwise TTC/PET pass.
    """
    rows: list[list[str]] = []
    for tr in moving_trajs:
        for ev in hard_braking_events(tr, min_duration=min_duration,
                                      smooth_window=smooth_window,
                                      speed_source=speed_source,
                                      despike=despike):
            if ev.speed_before < MIN_BRAKE_SPEED:
                continue
            if ev.peak_decel < MAX_PLAUSIBLE_DECEL:
                continue
            x, y = interp_position(tr, ev.t_start)
            rows.append([scene, 'HBE', f'{ev.peak_decel:.2f}', f'{ev.t_start:.1f}',
                         tr.track_id, tr.label, '', '',
                         f'{x:.2f}', f'{y:.2f}'])
    return rows
