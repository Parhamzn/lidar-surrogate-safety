"""3D multi-object tracker: Kalman prediction + Hungarian association.

Follows the AB3DMOT recipe (Weng et al., 2020) with the nuScenes-style
center-distance cost instead of 3D IoU: predicted track centers are matched
to detection centers in the ground plane with the Hungarian algorithm,
gated by a maximum distance. Tracks are confirmed after `min_hits` matches
and dropped after `max_age` consecutive misses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from lidar_pilot.tracking.kalman import KalmanBox3D
from lidar_pilot.trajectory import Trajectory


@dataclass
class Track:
    track_id: int
    label: str
    kf: KalmanBox3D
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    # History rows: [t, x, y, z, yaw, l, w, h, vx, vy, vz]
    history: list = field(default_factory=list)

    def record(self, t: float) -> None:
        self.history.append([t, *self.kf.box, *self.kf.velocity])

    def to_trajectory(self) -> Trajectory:
        h = np.asarray(self.history)
        return Trajectory(
            track_id=self.track_id,
            t=h[:, 0],
            xy=h[:, 1:3],
            label=self.label,
            yaw=h[:, 4],
            size_lw=np.median(h[:, 5:7], axis=0),
            extras={"z": h[:, 3], "velocity": h[:, 8:11]},
        )


class Tracker3D:
    def __init__(self,
                 max_match_distance: float = 2.5,
                 max_age: int = 3,
                 min_hits: int = 2,
                 min_score: float = 0.3):
        """max_match_distance: BEV gating radius in metres.
        max_age: consecutive missed frames before a track is terminated.
        min_hits: matches needed before a track counts as confirmed.
        min_score: detections below this confidence are ignored.
        """
        self.max_match_distance = max_match_distance
        self.max_age = max_age
        self.min_hits = min_hits
        self.min_score = min_score
        self._next_id = 1
        self._active: list[Track] = []
        self._finished: list[Track] = []
        self._last_t: float | None = None

    def step(self,
             boxes: np.ndarray,
             scores: np.ndarray,
             labels: list[str],
             t: float) -> list[Track]:
        """Advance one frame.

        boxes: (N, 7) [x, y, z, yaw, l, w, h]; scores: (N,); labels: N class
        names; t: frame timestamp in seconds. Returns currently confirmed
        tracks (after this frame's update).
        """
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 7)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        keep = scores >= self.min_score
        boxes, labels = boxes[keep], [l for l, k in zip(labels, keep) if k]

        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        for tr in self._active:
            tr.kf.predict(dt)

        matches, unmatched_dets, unmatched_tracks = self._associate(boxes, labels)

        for ti, di in matches:
            tr = self._active[ti]
            tr.kf.update(boxes[di])
            tr.hits += 1
            tr.misses = 0
            if tr.hits >= self.min_hits:
                tr.confirmed = True
            tr.record(t)

        for ti in unmatched_tracks:
            self._active[ti].misses += 1

        for di in unmatched_dets:
            tr = Track(self._next_id, labels[di], KalmanBox3D(boxes[di]))
            self._next_id += 1
            if self.min_hits <= 1:
                tr.confirmed = True
            tr.record(t)
            self._active.append(tr)

        still_active = []
        for tr in self._active:
            if tr.misses > self.max_age:
                if tr.confirmed:
                    self._finished.append(tr)
            else:
                still_active.append(tr)
        self._active = still_active

        return [tr for tr in self._active if tr.confirmed and tr.misses == 0]

    def _associate(self, boxes: np.ndarray, labels: list[str]):
        n_t, n_d = len(self._active), len(boxes)
        if n_t == 0 or n_d == 0:
            return [], list(range(n_d)), list(range(n_t))

        cost = np.full((n_t, n_d), 1e6)
        for i, tr in enumerate(self._active):
            pred_xy = tr.kf.box[:2]
            for j in range(n_d):
                if labels[j] != tr.label:
                    continue  # never match across classes
                d = float(np.linalg.norm(pred_xy - boxes[j, :2]))
                if d <= self.max_match_distance:
                    cost[i, j] = d

        rows, cols = linear_sum_assignment(cost)
        matches = [(i, j) for i, j in zip(rows, cols) if cost[i, j] < 1e6]
        matched_t = {i for i, _ in matches}
        matched_d = {j for _, j in matches}
        return (matches,
                [j for j in range(n_d) if j not in matched_d],
                [i for i in range(n_t) if i not in matched_t])

    @property
    def trajectories(self) -> list[Trajectory]:
        """All confirmed tracks (finished and still active) as trajectories."""
        out = [tr.to_trajectory() for tr in self._finished]
        out += [tr.to_trajectory() for tr in self._active if tr.confirmed]
        return out
