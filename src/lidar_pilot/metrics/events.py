"""Hard-braking event (HBE) detection from a single trajectory.

A hard-braking event is a sustained deceleration sharper than a threshold,
the connected-vehicle literature's leading indicator of elevated risk. The
default threshold of -3 m/s^2 (~0.3 g) follows the Google Research
hard-braking work; `min_duration` filters single-sample noise spikes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lidar_pilot.kinematics import longitudinal_accel, speed
from lidar_pilot.trajectory import Trajectory


@dataclass
class BrakingEvent:
    track_id: int
    t_start: float
    t_end: float
    peak_decel: float       # most negative m/s^2 during the event
    speed_before: float     # m/s at event start
    speed_after: float      # m/s at event end


def hard_braking_events(traj: Trajectory,
                        threshold: float = -3.0,
                        min_duration: float = 0.2,
                        smooth_window: int = 3) -> list[BrakingEvent]:
    """All hard-braking events in a trajectory.

    threshold: deceleration boundary in m/s^2 (negative; default -3.0).
    min_duration: minimum seconds below threshold to count as an event.
    """
    if len(traj) < 3:
        return []
    a = longitudinal_accel(traj, smooth_window)
    s = speed(traj, smooth_window)
    below = a <= threshold

    events: list[BrakingEvent] = []
    start = None
    for k in range(len(below)):
        if below[k] and start is None:
            start = k
        elif (not below[k] or k == len(below) - 1) and start is not None:
            end = k if below[k] else k - 1
            if traj.t[end] - traj.t[start] >= min_duration:
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
