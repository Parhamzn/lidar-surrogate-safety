"""Kinematics extraction from trajectories: speed, heading, acceleration.

Works on any Trajectory (tracker output, ground-truth annotations, or probe
data), so the same code path serves nuScenes validation and the roadside
study. Derivatives are computed with central finite differences on the
(already Kalman-smoothed) positions; an optional moving-average pass guards
against residual jitter, since acceleration amplifies noise twice.
"""

from __future__ import annotations

import numpy as np

from lidar_pilot.trajectory import Trajectory


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average along axis 0; window <= 1 is a no-op."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    if values.ndim == 1:
        return np.convolve(values, kernel, mode="same")
    return np.column_stack(
        [np.convolve(values[:, k], kernel, mode="same") for k in range(values.shape[1])]
    )


def velocity(traj: Trajectory, smooth_window: int = 1) -> np.ndarray:
    """(N, 2) ground-plane velocity in m/s via central differences."""
    if len(traj) < 2:
        return np.zeros((len(traj), 2))
    xy = _smooth(traj.xy, smooth_window)
    v = np.empty_like(xy)
    v[0] = (xy[1] - xy[0]) / (traj.t[1] - traj.t[0])
    v[-1] = (xy[-1] - xy[-2]) / (traj.t[-1] - traj.t[-2])
    if len(traj) > 2:
        dt = (traj.t[2:] - traj.t[:-2])[:, None]
        v[1:-1] = (xy[2:] - xy[:-2]) / dt
    return v


def speed(traj: Trajectory, smooth_window: int = 1) -> np.ndarray:
    """(N,) scalar speed in m/s."""
    return np.linalg.norm(velocity(traj, smooth_window), axis=1)


def heading(traj: Trajectory, smooth_window: int = 1) -> np.ndarray:
    """(N,) course-over-ground in radians (atan2 of velocity)."""
    v = velocity(traj, smooth_window)
    return np.arctan2(v[:, 1], v[:, 0])


def longitudinal_accel(traj: Trajectory, smooth_window: int = 1) -> np.ndarray:
    """(N,) rate of change of scalar speed in m/s^2.

    Negative values are braking. This is d|v|/dt (the quantity hard-braking
    thresholds are defined on), not the magnitude of the acceleration
    vector, which would also count pure cornering.
    """
    s = speed(traj, smooth_window)
    if len(traj) < 2:
        return np.zeros(len(traj))
    a = np.empty_like(s)
    a[0] = (s[1] - s[0]) / (traj.t[1] - traj.t[0])
    a[-1] = (s[-1] - s[-2]) / (traj.t[-1] - traj.t[-2])
    if len(traj) > 2:
        a[1:-1] = (s[2:] - s[:-2]) / (traj.t[2:] - traj.t[:-2])
    return a
