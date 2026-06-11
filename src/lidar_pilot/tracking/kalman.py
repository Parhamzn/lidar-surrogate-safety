"""Constant-velocity Kalman filter for one 3D bounding box (AB3DMOT-style).

State (10,): [x, y, z, yaw, l, w, h, vx, vy, vz]
Measurement (7,): [x, y, z, yaw, l, w, h]

Position evolves with constant velocity; yaw and box size are modelled as
(noisy) constants. Implemented directly (no filterpy) so every step is
inspectable: predict/update are the textbook KF equations.
"""

from __future__ import annotations

import numpy as np

_DIM_X = 10
_DIM_Z = 7


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float(np.arctan2(np.sin(a), np.cos(a)))


class KalmanBox3D:
    def __init__(self, box: np.ndarray,
                 init_velocity: np.ndarray | None = None,
                 pos_var: float = 1.0,
                 vel_var: float = 10.0,
                 accel_std: float = 3.0,
                 meas_std: float = 0.5):
        """box: (7,) [x, y, z, yaw, l, w, h] first detection of the object.

        init_velocity: optional (2,) or (3,) m/s seed, e.g. from a
        detector's velocity head. Without it a new track starts at zero
        velocity, which at low frame rates (nuScenes keyframes are 2 Hz)
        makes the first re-association fail for fast objects.
        accel_std (m/s^2) drives process noise: how much velocity may change
        between frames. meas_std (m) is detector localization noise.
        """
        self.x = np.zeros(_DIM_X)
        self.x[:_DIM_Z] = box
        # Covariance: confident in observed pose, uninformed about velocity.
        self.P = np.eye(_DIM_X) * pos_var
        self.P[7:, 7:] = np.eye(3) * vel_var
        if init_velocity is not None:
            v = np.asarray(init_velocity, dtype=float).ravel()
            self.x[7:7 + v.size] = v
            # Seeded velocity is far better than uninformed: shrink its var.
            self.P[7:, 7:] = np.eye(3) * 2.0
        self._accel_std = accel_std
        self.R = np.eye(_DIM_Z) * meas_std**2
        self.H = np.zeros((_DIM_Z, _DIM_X))
        self.H[:_DIM_Z, :_DIM_Z] = np.eye(_DIM_Z)

    def predict(self, dt: float) -> np.ndarray:
        F = np.eye(_DIM_X)
        F[0, 7] = F[1, 8] = F[2, 9] = dt
        # Piecewise-constant white acceleration model on (x, y, z).
        q_pos = 0.25 * dt**4 * self._accel_std**2
        q_cross = 0.5 * dt**3 * self._accel_std**2
        q_vel = dt**2 * self._accel_std**2
        Q = np.zeros((_DIM_X, _DIM_X))
        for p, v in ((0, 7), (1, 8), (2, 9)):
            Q[p, p] = q_pos
            Q[p, v] = Q[v, p] = q_cross
            Q[v, v] = q_vel
        # Small drift on yaw and size so the filter can follow slow change.
        Q[3, 3] = (0.1 * dt) ** 2
        Q[4:7, 4:7] = np.eye(3) * (0.05 * dt) ** 2

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x.copy()

    def update(self, box: np.ndarray) -> None:
        z = np.asarray(box, dtype=float).copy()
        # Detectors are ambiguous about heading by pi (a box looks the same
        # facing forward or backward). If the innovation exceeds pi/2, flip
        # the measured yaw so we correct the small error, not the flip.
        innov_yaw = wrap_angle(z[3] - self.x[3])
        if abs(innov_yaw) > np.pi / 2:
            z[3] = wrap_angle(z[3] + np.pi)

        y = z - self.H @ self.x
        y[3] = wrap_angle(y[3])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[3] = wrap_angle(self.x[3])
        self.P = (np.eye(_DIM_X) - K @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        """Current (7,) box estimate [x, y, z, yaw, l, w, h]."""
        return self.x[:_DIM_Z].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current (3,) velocity estimate [vx, vy, vz] in m/s."""
        return self.x[7:].copy()
