"""Tracker tests on synthetic scenes with known ground truth."""

import numpy as np
import pytest

from lidar_pilot.tracking import KalmanBox3D, Tracker3D

RNG = np.random.default_rng(7)
DT = 0.1


def make_box(x, y, yaw=0.0, size=(4.5, 1.9, 1.7)):
    return np.array([x, y, 0.0, yaw, *size])


def test_kalman_velocity_converges():
    """A constant-velocity object: the filter should recover its speed."""
    kf = KalmanBox3D(make_box(0, 0))
    v_true = np.array([10.0, -2.0])   # m/s
    for k in range(1, 40):
        kf.predict(DT)
        noisy = make_box(v_true[0] * k * DT, v_true[1] * k * DT)
        noisy[:2] += RNG.normal(0, 0.1, 2)
        kf.update(noisy)
    assert np.allclose(kf.velocity[:2], v_true, atol=0.5)


def test_kalman_yaw_flip_correction():
    """A detection flipped by pi must not yank the heading estimate."""
    kf = KalmanBox3D(make_box(0, 0, yaw=0.1))
    kf.predict(DT)
    kf.update(make_box(1, 0, yaw=0.1 + np.pi))
    assert abs(kf.box[3] - 0.1) < 0.2


def test_two_crossing_objects_keep_ids():
    """Two objects whose paths cross at different times: two stable IDs.

    A passes (12, 0) at t=1.5; B passes it at t=1.0. Closest approach
    between the objects is ~2.8 m, outside the 2 m gate, so association
    must never swap them.
    """
    tracker = Tracker3D(max_match_distance=2.0, min_hits=2)
    id_history = {"a": set(), "b": set()}
    for k in range(30):
        t = k * DT
        pos_a = np.array([k * DT * 8.0, 0.0])          # east along y=0
        pos_b = np.array([12.0, (k * DT - 1.0) * 8.0])  # north through x=12
        boxes = np.stack([make_box(*pos_a), make_box(*pos_b, yaw=np.pi / 2)])
        boxes[:, :2] += RNG.normal(0, 0.05, (2, 2))
        active = tracker.step(boxes, np.array([0.9, 0.9]), ["car", "car"], t)
        for tr in active:
            d_a = np.linalg.norm(tr.kf.box[:2] - pos_a)
            d_b = np.linalg.norm(tr.kf.box[:2] - pos_b)
            id_history["a" if d_a < d_b else "b"].add(tr.track_id)
    assert len(id_history["a"]) == 1
    assert len(id_history["b"]) == 1
    assert id_history["a"] != id_history["b"]


def test_track_survives_missed_detections():
    """Dropping 2 frames must not kill a track (max_age=3)."""
    tracker = Tracker3D(max_match_distance=3.0, min_hits=2, max_age=3)
    ids = set()
    for k in range(25):
        t = k * DT
        if 10 <= k < 12:    # occlusion: no detection
            tracker.step(np.zeros((0, 7)), np.array([]), [], t)
            continue
        box = make_box(k * DT * 10.0, 0.0)
        active = tracker.step(box[None], np.array([0.9]), ["car"], t)
        ids.update(tr.track_id for tr in active)
    assert len(ids) == 1


def test_no_cross_class_match():
    """A pedestrian detection near a car track must spawn a new track."""
    tracker = Tracker3D(max_match_distance=5.0, min_hits=1)
    tracker.step(make_box(0, 0)[None], np.array([0.9]), ["car"], 0.0)
    active = tracker.step(make_box(0.5, 0)[None], np.array([0.9]), ["pedestrian"], DT)
    labels = {tr.label for tr in tracker._active}
    assert labels == {"car", "pedestrian"}


def test_velocity_seeding_enables_low_framerate_tracking():
    """At 2 Hz a 14 m/s car moves 7 m between frames, far beyond a 3 m
    gate. Seeding new tracks with the detector's velocity head must keep
    the track alive; without seeding it fragments."""
    def run(seed: bool):
        tracker = Tracker3D(max_match_distance=3.0, min_hits=2)
        ids = set()
        for k in range(10):
            t = k * 0.5
            box = make_box(14.0 * t, 0.0)
            vel = np.array([[14.0, 0.0]]) if seed else None
            active = tracker.step(box[None], np.array([0.9]), ["car"], t,
                                  velocities=vel)
            ids.update(tr.track_id for tr in active)
        return ids

    assert len(run(seed=True)) == 1
    # Unseeded, no association ever succeeds: tracks die at 1 hit and never
    # confirm, so not a single stable track comes out.
    assert len(run(seed=False)) == 0


def test_per_class_gates():
    """With a dict gate, a 3 m jump matches a car but not a pedestrian."""
    gates = {"car": 4.0, "pedestrian": 2.0}
    for label, expect_ids in (("car", 1), ("pedestrian", 2)):
        tracker = Tracker3D(max_match_distance=gates, min_hits=1)
        tracker.step(make_box(0, 0)[None], np.array([0.9]), [label], 0.0)
        tracker.step(make_box(3.0, 0)[None], np.array([0.9]), [label], 0.5)
        assert len({tr.track_id for tr in tracker._active}) == expect_ids, label


def test_trajectory_export():
    tracker = Tracker3D(min_hits=2)
    for k in range(10):
        tracker.step(make_box(k * 1.0, 0.0)[None], np.array([0.9]), ["car"], k * DT)
    trajs = tracker.trajectories
    assert len(trajs) == 1
    assert trajs[0].label == "car"
    assert len(trajs[0]) == 10
    # ~10 m/s east; allow the filter a few frames to converge
    assert trajs[0].extras["velocity"][-1][0] == pytest.approx(10.0, abs=1.0)


def test_gate_growth_reacquires_after_occlusion():
    """A car braking hard while occluded for 1.1 s ends up ~5 m behind the
    constant-velocity prediction; a fixed 2 m gate loses it, a miss-grown
    gate re-acquires it."""
    def run(gate_growth: float):
        tracker = Tracker3D(max_match_distance=2.0, min_hits=2, max_age=15,
                            gate_growth=gate_growth)
        ids = set()
        v0, decel, x = 12.0, -8.0, 0.0
        for k in range(35):
            t = k * DT
            v = v0 if t < 1.0 else max(v0 + decel * (t - 1.0), 0.0)
            x += v * DT
            if 10 <= k < 21:   # occluded through the braking onset
                tracker.step(np.zeros((0, 7)), np.array([]), [], t)
                continue
            active = tracker.step(make_box(x, 0.0)[None], np.array([0.9]),
                                  ["car"], t, velocities=np.array([[v, 0.0]]))
            ids.update(tr.track_id for tr in active)
        return ids

    assert len(run(gate_growth=0.5)) == 1
    assert len(run(gate_growth=0.0)) == 2


def test_kf_params_accel_std_responsiveness():
    """Higher accel_std lets the velocity estimate follow a hard brake
    with less lag, so the tracked speed drops sooner."""
    def final_speed(accel_std: float) -> float:
        tracker = Tracker3D(max_match_distance=5.0, min_hits=1,
                            kf_params=dict(accel_std=accel_std))
        v0, decel = 12.0, -8.0
        x = 0.0
        for k in range(30):
            t = k * DT
            v = v0 if t < 1.0 else max(v0 + decel * (t - 1.0), 0.0)
            x += v * DT
            tracker.step(make_box(x, 0.0)[None], np.array([0.9]), ["car"], t,
                         velocities=np.array([[v0, 0.0]]) if k == 0 else None)
        (tr,) = tracker._active
        return float(np.linalg.norm(tr.kf.velocity[:2]))

    # true speed at t=2.9 is 0; the sluggish filter overestimates it more
    assert final_speed(9.0) < final_speed(3.0) - 0.5


def test_record_source_detection_keeps_raw_positions():
    """With record_source='detection' the history holds the matched boxes
    verbatim, not filter-smoothed positions."""
    raw = Tracker3D(min_hits=2, record_source='detection')
    smooth = Tracker3D(min_hits=2)
    xs = []
    for k in range(12):
        x = k * 1.0 + (0.3 if k % 2 else -0.3)   # zigzag measurement noise
        xs.append(x)
        for tracker in (raw, smooth):
            tracker.step(make_box(x, 0.0)[None], np.array([0.9]), ["car"], k * DT)
    (tr_raw,), (tr_smooth,) = raw.trajectories, smooth.trajectories
    assert np.allclose(tr_raw.xy[:, 0], xs)
    assert not np.allclose(tr_smooth.xy[:, 0], xs)


def test_record_source_validated():
    with pytest.raises(ValueError):
        Tracker3D(record_source='raw')
