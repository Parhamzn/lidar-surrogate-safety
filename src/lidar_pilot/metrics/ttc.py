"""Time-to-Collision (TTC) between two tracked road users.

Definition used (point mass with combined collision radius): at each common
timestamp, assume both users keep their current velocity. TTC is the time
until their center distance first reaches the combined radius `d`:

    || r + v_rel * t || = d,   r = p2 - p1,  v_rel = v2 - v1

i.e. the smallest non-negative root of
    |v|^2 t^2 + 2 (r.v) t + (|r|^2 - d^2) = 0.

If the discriminant is negative the predicted paths miss each other and TTC
is undefined at that instant. For two users closing head-on this reduces to
the classic gap / closing-speed definition (Hayward, 1972). The combined
radius defaults to half the mean of the two users' box diagonals, a simple
isotropic stand-in for oriented-box contact that suits a pilot; the
literature's caveat applies (definitions vary, state the one you use).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lidar_pilot.kinematics import velocity
from lidar_pilot.trajectory import Trajectory


def _combined_radius(a: Trajectory, b: Trajectory, default_diagonal: float = 1.5) -> float:
    """Sum of the two users' half box diagonals (circumscribing circles)."""
    radii = []
    for traj in (a, b):
        diag = float(np.hypot(*traj.size_lw)) if traj.size_lw is not None else default_diagonal
        radii.append(diag / 2)
    return float(sum(radii))


def ttc_series(a: Trajectory, b: Trajectory,
               dt: float = 0.1,
               collision_radius: float | None = None,
               smooth_window: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """TTC over the common time window of two trajectories.

    Returns (t_grid, ttc) where ttc is NaN wherever undefined (not on a
    collision course). Both trajectories are resampled to a common grid with
    spacing `dt` seconds.
    """
    window = a.overlap_window(b)
    if window is None:
        return np.array([]), np.array([])
    t_grid = np.arange(window[0], window[1] + 1e-9, dt)
    t_grid = t_grid[t_grid <= window[1]]
    ra, rb = a.resample(t_grid), b.resample(t_grid)
    va, vb = velocity(ra, smooth_window), velocity(rb, smooth_window)

    d = collision_radius if collision_radius is not None else _combined_radius(a, b)
    r = rb.xy - ra.xy
    v = vb - va
    vv = np.einsum("ij,ij->i", v, v)
    rv = np.einsum("ij,ij->i", r, v)
    rr = np.einsum("ij,ij->i", r, r)

    ttc = np.full(t_grid.shape, np.nan)
    already = rr <= d * d           # contact: TTC is zero
    ttc[already] = 0.0

    disc = rv**2 - vv * (rr - d * d)
    valid = (~already) & (vv > 1e-12) & (disc >= 0)
    t_hit = np.full(t_grid.shape, np.nan)
    t_hit[valid] = (-rv[valid] - np.sqrt(disc[valid])) / vv[valid]
    hit_future = valid & (t_hit >= 0)
    ttc[hit_future] = t_hit[hit_future]
    return t_grid, ttc


@dataclass
class TTCResult:
    min_ttc: float
    t_at_min: float
    t_grid: np.ndarray
    ttc: np.ndarray


def min_ttc(a: Trajectory, b: Trajectory, **kwargs) -> TTCResult | None:
    """Minimum TTC over the encounter (the conventional severity score).

    Returns None if the pair is never on a predicted collision course.
    """
    t_grid, ttc = ttc_series(a, b, **kwargs)
    if t_grid.size == 0 or np.all(np.isnan(ttc)):
        return None
    i = int(np.nanargmin(ttc))
    return TTCResult(float(ttc[i]), float(t_grid[i]), t_grid, ttc)
