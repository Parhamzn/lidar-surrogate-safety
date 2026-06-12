"""Hard-braking event (HBE) detection from a single trajectory.

A hard-braking event is a sustained deceleration sharper than a threshold,
the connected-vehicle literature's leading indicator of elevated risk. The
default threshold of -3 m/s^2 (~0.3 g) follows the Google Research
hard-braking work; `min_duration` filters single-sample noise spikes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lidar_pilot.kinematics import speed
from lidar_pilot.trajectory import Trajectory


@dataclass
class BrakingEvent:
    track_id: int
    t_start: float
    t_end: float
    peak_decel: float       # most negative m/s^2 during the event
    speed_before: float     # m/s at event start
    speed_after: float      # m/s at event end


def _central_diff(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    d = np.empty_like(values)
    d[0] = (values[1] - values[0]) / (t[1] - t[0])
    d[-1] = (values[-1] - values[-2]) / (t[-1] - t[-2])
    if len(values) > 2:
        d[1:-1] = (values[2:] - values[:-2]) / (t[2:] - t[:-2])
    return d


def _median3(values: np.ndarray) -> np.ndarray:
    """3-point median despike: removes single-sample outliers (e.g. a box
    teleporting for one frame) while leaving monotone ramps untouched."""
    if len(values) < 3:
        return values
    out = values.copy()
    out[1:-1] = np.median(
        np.stack([values[:-2], values[1:-1], values[2:]]), axis=0)
    return out


def hard_braking_events(traj: Trajectory,
                        threshold: float = -3.0,
                        min_duration: float = 0.2,
                        smooth_window: int = 3,
                        speed_source: str = 'positions',
                        despike: bool = False) -> list[BrakingEvent]:
    """All hard-braking events in a trajectory.

    threshold: deceleration boundary in m/s^2 (negative; default -3.0).
    min_duration: minimum seconds below threshold to count as an event.
    speed_source: 'positions' differentiates (smoothed) positions twice;
    'kf_velocity' differentiates the tracker's Kalman velocity state
    (``traj.extras['velocity']``) once, which avoids the second
    noise-amplifying derivative on tracker output. Ground-truth
    trajectories have no filter state, so 'positions' is their only option.
    despike: median-filter the speed series before differentiating, for
    sources with occasional single-frame glitches.
    """
    if len(traj) < 3:
        return []
    if speed_source == 'kf_velocity':
        from lidar_pilot.kinematics import _smooth
        v = np.asarray(traj.extras['velocity'], dtype=float)
        s = _smooth(np.linalg.norm(v[:, :2], axis=1), smooth_window)
    elif speed_source == 'positions':
        s = speed(traj, smooth_window)
    else:
        raise ValueError(f'unknown speed_source: {speed_source!r}')
    if despike:
        s = _median3(s)
    a = _central_diff(s, traj.t)
    below = a <= threshold
    # The first and last samples use one-sided differences and reflect
    # track birth/death transients, not vehicle dynamics: never start or
    # end an event on them.
    below[0] = below[-1] = False

    events: list[BrakingEvent] = []
    start = None
    for k in range(len(below)):
        if below[k] and start is None:
            start = k
        elif (not below[k] or k == len(below) - 1) and start is not None:
            end = k if below[k] else k - 1
            # epsilon: at 10 Hz a 3-sample run is exactly min_duration long,
            # and float timestamp differences (k+2)/10 - k/10 land on either
            # side of 0.2 depending on k — without tolerance ~40% of
            # exact-length events vanish on start-frame parity.
            if traj.t[end] - traj.t[start] >= min_duration - 1e-9:
                seg = slice(start, end + 1)
                events.append(BrakingEvent(
                    track_id=traj.track_id,
                    t_start=float(traj.t[start]),
                    t_end=float(traj.t[end]),
                    peak_decel=float(a[seg].min()),
                    speed_before=float(s[start]),
                    speed_after=float(s[end]),
                ))
            start = None
    return events
