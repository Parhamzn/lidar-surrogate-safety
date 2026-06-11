"""Trajectory container shared by the tracker, kinematics and metrics layers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trajectory:
    """A time-stamped 2D (bird's-eye-view) trajectory of one road user.

    Heights matter for detection but not for the planar conflict metrics
    (TTC/PET are defined on ground-plane motion), so the metrics layer works
    in BEV. `label` carries the road-user class (e.g. "car", "pedestrian").
    """

    track_id: int
    t: np.ndarray          # (N,) timestamps, seconds, strictly increasing
    xy: np.ndarray         # (N, 2) ground-plane positions, metres
    label: str = "unknown"
    yaw: np.ndarray | None = None    # (N,) heading, radians
    size_lw: np.ndarray | None = None  # (2,) median box length/width, metres
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.xy = np.asarray(self.xy, dtype=float)
        if self.t.ndim != 1 or self.xy.shape != (self.t.size, 2):
            raise ValueError("t must be (N,) and xy must be (N, 2)")
        if self.t.size >= 2 and np.any(np.diff(self.t) <= 0):
            raise ValueError("timestamps must be strictly increasing")

    def __len__(self) -> int:
        return self.t.size

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    def resample(self, t_new: np.ndarray) -> "Trajectory":
        """Linearly interpolate the trajectory onto new timestamps.

        Timestamps outside the original range are not extrapolated; callers
        must pass t_new within [t[0], t[-1]].
        """
        t_new = np.asarray(t_new, dtype=float)
        if t_new.size and (t_new[0] < self.t[0] - 1e-9 or t_new[-1] > self.t[-1] + 1e-9):
            raise ValueError("resample timestamps outside trajectory range")
        xy_new = np.column_stack(
            [np.interp(t_new, self.t, self.xy[:, k]) for k in range(2)]
        )
        return Trajectory(self.track_id, t_new, xy_new, label=self.label,
                          size_lw=self.size_lw)

    def overlap_window(self, other: "Trajectory") -> tuple[float, float] | None:
        """Common time interval of two trajectories, or None if disjoint."""
        lo = max(self.t[0], other.t[0])
        hi = min(self.t[-1], other.t[-1])
        return (lo, hi) if hi > lo else None
