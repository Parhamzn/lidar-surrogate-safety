"""Post-Encroachment Time (PET) between two tracked road users.

Definition used (Allen, Shin & Cooper, 1978): the conflict point is where
the two users' paths cross in space. PET is the elapsed time between the
first user leaving that point and the second user arriving at it. Unlike
TTC it is a measured (not predicted) quantity and needs no collision
course, which makes it the standard metric for path-crossing conflicts at
intersections.

Implementation: both BEV paths are treated as polylines; every segment pair
intersection yields a conflict point, and each user's passage time at that
point is linearly interpolated within its segment. Multiple crossings (e.g.
looping trajectories) yield multiple PET events.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lidar_pilot.trajectory import Trajectory


@dataclass
class PETEvent:
    pet: float                  # seconds, >= 0
    conflict_point: np.ndarray  # (2,) metres
    first_id: int               # track that cleared the point first
    second_id: int
    t_first: float              # passage times at the conflict point
    t_second: float


def _segment_intersection(p1, p2, q1, q2):
    """Intersection of segments p1-p2 and q1-q2.

    Returns (point, s, u) with s, u in [0, 1] the fractional positions along
    each segment, or None if they do not cross.
    """
    d1, d2 = p2 - p1, q2 - q1
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:      # parallel or collinear: no single crossing point
        return None
    diff = q1 - p1
    s = (diff[0] * d2[1] - diff[1] * d2[0]) / denom
    u = (diff[0] * d1[1] - diff[1] * d1[0]) / denom
    if -1e-9 <= s <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return p1 + np.clip(s, 0, 1) * d1, float(np.clip(s, 0, 1)), float(np.clip(u, 0, 1))
    return None


def pet_events(a: Trajectory, b: Trajectory, max_pet: float = 10.0) -> list[PETEvent]:
    """All path-crossing PET events between two trajectories.

    Events with PET above `max_pet` are discarded: very large gaps are
    ordinary passages, not interactions (thresholds in the literature for
    *conflicts* are typically a few seconds at most).
    """
    events: list[PETEvent] = []
    eps = 1e-9
    for i in range(len(a) - 1):
        for j in range(len(b) - 1):
            hit = _segment_intersection(a.xy[i], a.xy[i + 1], b.xy[j], b.xy[j + 1])
            if hit is None:
                continue
            point, s, u = hit
            # Treat segments as half-open [start, end) so a crossing that
            # falls exactly on a shared polyline vertex is counted once,
            # not once per adjacent segment. The final segment stays closed.
            if s >= 1 - eps and i < len(a) - 2:
                continue
            if u >= 1 - eps and j < len(b) - 2:
                continue
            t_a = a.t[i] + s * (a.t[i + 1] - a.t[i])
            t_b = b.t[j] + u * (b.t[j + 1] - b.t[j])
            pet = abs(t_b - t_a)
            if pet > max_pet:
                continue
            first, second = (a, b) if t_a <= t_b else (b, a)
            events.append(PETEvent(
                pet=float(pet),
                conflict_point=np.asarray(point),
                first_id=first.track_id,
                second_id=second.track_id,
                t_first=float(min(t_a, t_b)),
                t_second=float(max(t_a, t_b)),
            ))
    return events


def min_pet(a: Trajectory, b: Trajectory, **kwargs) -> PETEvent | None:
    """The most severe (smallest) PET event of the pair, or None."""
    events = pet_events(a, b, **kwargs)
    return min(events, key=lambda e: e.pet) if events else None
