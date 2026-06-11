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
