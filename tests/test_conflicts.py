"""Tests for the conflict-mining library and braking-estimator variants."""

import numpy as np
import pytest

from lidar_pilot.conflicts import mine_conflicts, select_moving_road_users
from lidar_pilot.metrics import hard_braking_events
from lidar_pilot.trajectory import Trajectory

DT = 0.1


def braking_car(track_id=1, v0=12.0, decel=-5.0, t_brake=2.0, n=60,
                with_kf_velocity=False):
    """Straight track: constant v0, then braking at `decel` until stopped."""
    t = np.arange(n) * DT
    v = np.where(t < t_brake, v0, np.maximum(v0 + decel * (t - t_brake), 0.0))
    x = np.concatenate([[0.0], np.cumsum(v[:-1] * DT)])
    xy = np.column_stack([x, np.zeros(n)])
    extras = {}
    if with_kf_velocity:
        extras['velocity'] = np.column_stack([v, np.zeros(n), np.zeros(n)])
    return Trajectory(track_id, t, xy, label='car', extras=extras)


def parked_jitter(track_id=2, n=60):
    rng = np.random.default_rng(3)
    xy = rng.normal(0, 0.15, (n, 2)) + np.array([30.0, 30.0])
    return Trajectory(track_id, np.arange(n) * DT, xy, label='car')


def test_kf_velocity_speed_source_finds_braking():
    tr = braking_car(with_kf_velocity=True)
    evs = hard_braking_events(tr, speed_source='kf_velocity')
    assert len(evs) == 1
    assert evs[0].peak_decel == pytest.approx(-5.0, abs=1.0)
    assert evs[0].t_start == pytest.approx(2.0, abs=0.4)


def test_kf_velocity_source_requires_filter_state():
    with pytest.raises(KeyError):
        hard_braking_events(braking_car(), speed_source='kf_velocity')
    with pytest.raises(ValueError):
        hard_braking_events(braking_car(), speed_source='nonsense')


def test_select_moving_road_users_drops_parked_jitter():
    kept, n_implausible = select_moving_road_users(
        [braking_car(), parked_jitter()])
    assert [tr.track_id for tr in kept] == [1]
    assert n_implausible == 0


def test_mine_conflicts_reports_hbe():
    rows, stats = mine_conflicts([braking_car(), parked_jitter()], 'scene')
    assert stats.n_moving == 1
    assert stats.n_hbe == 1
    hbe = [r for r in rows if r[1] == 'HBE']
    assert len(hbe) == 1
    assert hbe[0][5] == 'car'
    # event position is where the vehicle was at braking onset (~24 m)
    assert float(hbe[0][8]) == pytest.approx(24.0, abs=3.0)


def test_mine_conflicts_estimator_params_propagate():
    """A 0.3 s brake pulse passes min_duration=0.2 but not 0.5."""
    t = np.arange(60) * DT
    v = np.where((t >= 2.0) & (t < 2.3), 12.0 - 5.0 * (t - 2.0), 12.0)
    v = np.where(t >= 2.3, 10.5, v)
    x = np.concatenate([[0.0], np.cumsum(v[:-1] * DT)])
    tr = Trajectory(5, t, np.column_stack([x, np.zeros(60)]), label='car')
    _, st_loose = mine_conflicts([tr], 's', hbe_min_duration=0.2,
                                 hbe_smooth_window=1)
    _, st_tight = mine_conflicts([tr], 's', hbe_min_duration=0.5,
                                 hbe_smooth_window=1)
    assert st_loose.n_hbe == 1
    assert st_tight.n_hbe == 0


def test_despike_drops_outliers_keeps_ramps():
    """The 3-point median kills single-sample speed outliers but must not
    soften a genuine braking ramp."""
    from lidar_pilot.metrics.events import _median3

    ramp = np.linspace(12.0, 2.0, 21)
    assert np.allclose(_median3(ramp), ramp)

    spiked = ramp.copy()
    spiked[8] = 25.0
    assert np.allclose(_median3(spiked)[8], ramp[8], atol=0.51)


def test_despike_clean_track_events_unchanged():
    """On glitch-free data the despiked estimator finds the same event."""
    tr = braking_car()
    plain, despiked = (hard_braking_events(tr, despike=d) for d in (False, True))
    assert len(plain) == len(despiked) == 1
    assert despiked[0].t_start == pytest.approx(plain[0].t_start, abs=0.11)
    assert despiked[0].peak_decel == pytest.approx(plain[0].peak_decel, abs=0.5)


def test_min_duration_immune_to_float_timestamp_dust():
    """(k+2)/10 - k/10 floats to either side of 0.2 depending on k; an
    exact-length run must count as min_duration regardless of start frame."""
    found = []
    for k0 in range(2, 12):
        n = 40
        t = np.array([(k0 + k) / 10 for k in range(n)])
        v = np.full(n, 12.0)
        v[20:23] = 12.0 - 4.0 * np.arange(1, 4) / 3   # 3 accel samples below
        x = np.concatenate([[0.0], np.cumsum(v[:-1] * np.diff(t))])
        tr = Trajectory(9, t, np.column_stack([x, np.zeros(n)]), label='car')
        evs = hard_braking_events(tr, smooth_window=1, min_duration=0.2)
        found.append(len(evs))
    assert all(n == found[0] for n in found), found
    assert found[0] >= 1
