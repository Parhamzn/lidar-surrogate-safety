"""Offline RTS (Rauch-Tung-Striebel) smoothing of finished trajectories.

The online tracker must be causal, so its constant-velocity filter lags
real maneuvers: at a braking onset the filter still believes in the old
velocity and its posterior smears the deceleration peak. Recorded data
has no such constraint — once a track is finished, a backward pass can
revise every estimate with knowledge of what happened next ("one second
later this car is at standstill, so it must be braking hard now").

`rts_smooth` runs an independent ground-plane constant-velocity Kalman
filter over a trajectory's recorded positions (use tracker
``record_source='detection'`` so these are raw detections, not already-
filtered posteriors) and applies the RTS backward recursion. The result
is both smoother *and* less lagged than any causal filter — the
responsiveness/smoothness trade-off that limits forward filtering does
not apply.
"""

from __future__ import annotations

import numpy as np

from lidar_pilot.trajectory import Trajectory


def _cv_matrices(dt: float, accel_std: float):
    """Transition and process noise for a [x, y, vx, vy] CV model."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q_pos = 0.25 * dt**4 * accel_std**2
    q_cross = 0.5 * dt**3 * accel_std**2
    q_vel = dt**2 * accel_std**2
    Q = np.zeros((4, 4))
    for p, v in ((0, 2), (1, 3)):
        Q[p, p] = q_pos
        Q[p, v] = Q[v, p] = q_cross
        Q[v, v] = q_vel
    return F, Q


def rts_smooth(traj: Trajectory,
               accel_std: float = 4.5,
               meas_std: float = 0.4) -> Trajectory:
    """Forward KF + RTS backward pass over one trajectory.

    Returns a new Trajectory with smoothed ground-plane positions and the
    smoothed velocity state in ``extras['velocity']`` (vz = 0), leaving
    every other field untouched. Timestamp gaps (missed frames) are
    handled by the per-step dt in the transition model.

    accel_std (m/s^2) is the smoother's maneuver allowance; meas_std (m)
    the per-detection localization noise.
    """
    n = len(traj)
    if n < 3:
        return traj

    H = np.zeros((2, 4))
    H[0, 0] = H[1, 1] = 1.0
    R = np.eye(2) * meas_std**2

    # forward filter, storing what the backward pass needs
    xs_f = np.empty((n, 4))           # filtered states
    Ps_f = np.empty((n, 4, 4))
    xs_p = np.empty((n, 4))           # predicted states (x_{k|k-1})
    Ps_p = np.empty((n, 4, 4))
    Fs = np.empty((n, 4, 4))

    v0 = (traj.xy[1] - traj.xy[0]) / (traj.t[1] - traj.t[0])
    x = np.array([traj.xy[0, 0], traj.xy[0, 1], v0[0], v0[1]])
    P = np.diag([meas_std**2, meas_std**2, 36.0, 36.0])
    xs_f[0], Ps_f[0] = x, P
    xs_p[0], Ps_p[0], Fs[0] = x, P, np.eye(4)   # unused at k=0

    for k in range(1, n):
        F, Q = _cv_matrices(float(traj.t[k] - traj.t[k - 1]), accel_std)
        x = F @ x
        P = F @ P @ F.T + Q
        xs_p[k], Ps_p[k], Fs[k] = x, P, F

        y = traj.xy[k] - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P
        xs_f[k], Ps_f[k] = x, P

    # RTS backward recursion
    xs_s = xs_f.copy()
    Ps_s = Ps_f.copy()
    for k in range(n - 2, -1, -1):
        C = Ps_f[k] @ Fs[k + 1].T @ np.linalg.inv(Ps_p[k + 1])
        xs_s[k] = xs_f[k] + C @ (xs_s[k + 1] - xs_p[k + 1])
        Ps_s[k] = Ps_f[k] + C @ (Ps_s[k + 1] - Ps_p[k + 1]) @ C.T

    extras = dict(traj.extras)
    extras['velocity'] = np.column_stack([xs_s[:, 2:4], np.zeros(n)])
    return Trajectory(traj.track_id, traj.t.copy(), xs_s[:, :2].copy(),
                      label=traj.label, yaw=traj.yaw, size_lw=traj.size_lw,
                      extras=extras)
