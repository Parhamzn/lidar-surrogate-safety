"""RTS smoother tests: noise reduction without causal lag."""

import numpy as np
import pytest

from lidar_pilot.tracking import Tracker3D, rts_smooth
from lidar_pilot.trajectory import Trajectory

RNG = np.random.default_rng(11)
DT = 0.1


def noisy_braking_track(v0=12.0, decel=-5.0, t_brake=3.0, n=80, sigma=0.15):
    """Straight drive then hard brake, with detection-like position noise.
    Returns (noisy trajectory, true speed profile)."""
    t = np.arange(n) * DT
    v = np.where(t < t_brake, v0, np.maximum(v0 + decel * (t - t_brake), 0.0))
    x = np.concatenate([[0.0], np.cumsum(v[:-1] * DT)])
    xy = np.column_stack([x, np.zeros(n)]) + RNG.normal(0, sigma, (n, 2))
    return Trajectory(1, t, xy, label='car'), v


def test_rts_reduces_position_noise():
    tr, _ = noisy_braking_track()
    sm = rts_smooth(tr)
    true_x = np.concatenate([[0.0], np.cumsum(
        np.where(tr.t < 3.0, 12.0, np.maximum(12.0 - 5.0 * (tr.t - 3.0), 0.0))[:-1] * DT)])
    err_raw = np.abs(tr.xy[:, 0] - true_x).mean()
    err_sm = np.abs(sm.xy[:, 0] - true_x).mean()
    assert err_sm < err_raw * 0.7


def test_rts_velocity_tracks_braking_without_lag():
    """The smoothed velocity must follow the true decel profile closely —
    including right at the onset, where a causal filter lags."""
    tr, v_true = noisy_braking_track()
    sm = rts_smooth(tr)
    v_sm = np.linalg.norm(sm.extras['velocity'][:, :2], axis=1)
    # overall fit
    assert np.abs(v_sm - v_true).mean() < 0.35
    # at onset +0.3 s the true speed is 10.5; a causal KF (accel_std 4.5)
    # overestimates it by ~1 m/s, the smoother must not
    k = np.searchsorted(tr.t, 3.3)
    assert abs(v_sm[k] - v_true[k]) < 0.5


def test_rts_beats_causal_filter_at_onset():
    """Same data through the online tracker: RTS speed error in the first
    second of braking must be clearly smaller than the causal filter's."""
    tr, v_true = noisy_braking_track()
    tracker = Tracker3D(max_match_distance=5.0, min_hits=1,
                        kf_params=dict(accel_std=4.5))
    for k in range(len(tr)):
        box = np.array([tr.xy[k, 0], tr.xy[k, 1], 0, 0, 4.5, 1.9, 1.7])
        tracker.step(box[None], np.array([0.9]), ['car'], float(tr.t[k]))
    (causal,) = tracker.trajectories
    v_causal = np.linalg.norm(np.asarray(causal.extras['velocity'])[:, :2], axis=1)
    v_sm = np.linalg.norm(rts_smooth(tr).extras['velocity'][:, :2], axis=1)

    win = (tr.t >= 3.0) & (tr.t <= 4.0)
    err_causal = np.abs(v_causal[win] - v_true[win]).mean()
    err_sm = np.abs(v_sm[win] - v_true[win]).mean()
    assert err_sm < err_causal * 0.6


def test_rts_handles_timestamp_gaps():
    tr, _ = noisy_braking_track()
    keep = np.ones(len(tr), bool)
    keep[30:36] = False          # 0.6 s occlusion gap
    gappy = Trajectory(2, tr.t[keep], tr.xy[keep], label='car')
    sm = rts_smooth(gappy)
    assert len(sm) == keep.sum()
    assert np.array_equal(sm.t, gappy.t)
    v = np.linalg.norm(sm.extras['velocity'][:, :2], axis=1)
    assert np.all(v < 14.0)      # no blow-up across the gap


def test_rts_short_track_passthrough():
    tr = Trajectory(3, [0.0, 0.1], [[0, 0], [1, 0]], label='car')
    assert rts_smooth(tr) is tr


def test_rts_preserves_identity_fields():
    tr, _ = noisy_braking_track()
    tr.extras['z'] = np.zeros(len(tr))
    sm = rts_smooth(tr)
    assert sm.track_id == tr.track_id and sm.label == tr.label
    assert 'z' in sm.extras
    assert sm.extras['velocity'].shape == (len(tr), 3)
