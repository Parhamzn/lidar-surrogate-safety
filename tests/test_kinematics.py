"""Kinematics tests against closed-form motion profiles."""

import numpy as np
import pytest

from lidar_pilot.kinematics import heading, longitudinal_accel, speed
from lidar_pilot.metrics import hard_braking_events
from lidar_pilot.trajectory import Trajectory


def constant_speed_traj(v=13.9, duration=10.0, dt=0.1):
    t = np.arange(0, duration, dt)
    xy = np.column_stack([v * t, np.zeros_like(t)])
    return Trajectory(1, t, xy, label="car")


def braking_traj(v0=15.0, decel=4.0, t_brake=5.0, brake_dur=2.0, dt=0.05):
    """Constant speed, then brake at `decel` m/s^2 for brake_dur, then hold."""
    t = np.arange(0, t_brake + brake_dur + 5.0, dt)
    spd = np.where(t < t_brake, v0,
                   np.where(t < t_brake + brake_dur,
                            v0 - decel * (t - t_brake),
                            v0 - decel * brake_dur))
    x = np.concatenate([[0], np.cumsum(spd[:-1] * dt)])
    return Trajectory(2, t, np.column_stack([x, np.zeros_like(t)]), label="car")


def test_speed_constant():
    traj = constant_speed_traj(v=13.9)
    assert np.allclose(speed(traj), 13.9, atol=0.01)


def test_heading_east():
    traj = constant_speed_traj()
    assert np.allclose(heading(traj), 0.0, atol=0.01)


def test_accel_zero_when_cruising():
    traj = constant_speed_traj()
    assert np.allclose(longitudinal_accel(traj), 0.0, atol=0.01)


def test_hard_braking_detected():
    traj = braking_traj(decel=4.0)
    events = hard_braking_events(traj, threshold=-3.0, min_duration=0.2)
    assert len(events) == 1
    ev = events[0]
    assert ev.t_start == pytest.approx(5.0, abs=0.3)
    assert ev.t_end == pytest.approx(7.0, abs=0.3)
    assert ev.peak_decel == pytest.approx(-4.0, abs=0.5)
    assert ev.speed_before == pytest.approx(15.0, abs=0.5)
    assert ev.speed_after == pytest.approx(7.0, abs=0.6)


def test_gentle_braking_not_flagged():
    traj = braking_traj(decel=1.5)
    assert hard_braking_events(traj, threshold=-3.0) == []
