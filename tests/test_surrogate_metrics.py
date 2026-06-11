"""TTC and PET tests against closed-form encounter geometry."""

import numpy as np
import pytest

from lidar_pilot.metrics import min_pet, min_ttc, pet_events, ttc_series
from lidar_pilot.trajectory import Trajectory


def straight_traj(track_id, start, v, duration=10.0, dt=0.1, label="car"):
    t = np.arange(0, duration, dt)
    xy = np.asarray(start, float)[None] + np.asarray(v, float)[None] * t[:, None]
    return Trajectory(track_id, t, xy, label=label)


def test_ttc_head_on():
    """Closing head-on at 20 m/s from 100 m apart with radius 5 m:
    first TTC sample must be (100 - 5) / 20 = 4.75 s."""
    a = straight_traj(1, (0, 0), (10, 0), duration=4.0)
    b = straight_traj(2, (100, 0), (-10, 0), duration=4.0)
    t_grid, ttc = ttc_series(a, b, collision_radius=5.0)
    assert ttc[0] == pytest.approx(4.75, abs=0.05)
    # TTC must fall roughly linearly while both keep speed
    assert ttc[10] == pytest.approx(4.75 - 1.0, abs=0.1)


def test_ttc_undefined_when_diverging():
    a = straight_traj(1, (0, 0), (10, 0), duration=5.0)
    b = straight_traj(2, (50, 0), (15, 0), duration=5.0)   # running away faster
    _, ttc = ttc_series(a, b, collision_radius=5.0)
    assert np.all(np.isnan(ttc))


def test_ttc_parallel_paths_no_collision_course():
    """Parallel same-speed traffic 10 m apart: never a collision course."""
    a = straight_traj(1, (0, 0), (10, 0))
    b = straight_traj(2, (0, 10), (10, 0))
    res = min_ttc(a, b, collision_radius=3.0)
    assert res is None


def test_ttc_undefined_for_crossing_with_comfortable_gap():
    """Orthogonal paths through the same point, arriving 1 s apart at
    10 m/s: closest approach is ~7 m, never a predicted collision, so TTC
    is undefined. This is exactly the geometry PET exists for."""
    a = straight_traj(1, (-50, 0), (10, 0))      # reaches (0,0) at t=5
    b = straight_traj(2, (0, -60), (0, 10))      # reaches (0,0) at t=6
    assert min_ttc(a, b, collision_radius=2.0) is None


def test_min_ttc_orthogonal_tight_crossing():
    """Same crossing but arriving only 0.2 s apart (2 m gap): inside the
    collision radius, so a finite TTC must exist."""
    a = straight_traj(1, (-50, 0), (10, 0))      # reaches (0,0) at t=5
    b = straight_traj(2, (0, -52), (0, 10))      # reaches (0,0) at t=5.2
    res = min_ttc(a, b, collision_radius=2.0)
    assert res is not None
    assert np.isfinite(res.min_ttc)
    assert res.min_ttc >= 0


def test_pet_orthogonal_crossing():
    """A passes (50, 0) at t=5; B passes it at t=3. PET = 2 s, B first."""
    a = straight_traj(1, (0, 0), (10, 0))            # east along y=0
    b = straight_traj(2, (50, -30), (0, 10), label="bicycle")  # north through x=50
    events = pet_events(a, b)
    assert len(events) == 1
    ev = events[0]
    assert ev.pet == pytest.approx(2.0, abs=0.02)
    assert ev.first_id == 2
    assert ev.second_id == 1
    assert np.allclose(ev.conflict_point, [50, 0], atol=0.1)
    assert ev.t_first == pytest.approx(3.0, abs=0.02)
    assert ev.t_second == pytest.approx(5.0, abs=0.02)


def test_pet_no_crossing():
    a = straight_traj(1, (0, 0), (10, 0))
    b = straight_traj(2, (0, 5), (10, 0))    # parallel, never crosses
    assert min_pet(a, b) is None


def test_pet_above_threshold_discarded():
    a = straight_traj(1, (0, 0), (10, 0), duration=30.0)       # at (50,0) t=5
    b = straight_traj(2, (50, -200), (0, 10), duration=30.0)   # at (50,0) t=20
    assert pet_events(a, b, max_pet=10.0) == []
    assert min_pet(a, b, max_pet=20.0).pet == pytest.approx(15.0, abs=0.05)
