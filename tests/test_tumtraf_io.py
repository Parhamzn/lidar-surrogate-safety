"""TUMTraf OpenLABEL + PCD adapter tests on synthetic data."""

import json
import struct

import numpy as np
import pytest

from lidar_pilot.io.tumtraf import (TUMTRAF_CLASS_MAP, load_tumtraf_trajectories,
                                    quat_to_yaw, read_openlabel_boxes, read_pcd)


def yaw_quat(yaw):
    """Unit quaternion [qx,qy,qz,qw] for a rotation about +z."""
    return [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]


def write_frame(path, fid, objects, timestamp=None):
    """objects: list of (uuid, TYPE, x, y, z, l, w, h, yaw)."""
    frame = {'objects': {}}
    if timestamp is not None:
        frame['frame_properties'] = {'timestamp': timestamp}
    for uuid, typ, x, y, z, l, w, h, yaw in objects:
        frame['objects'][uuid] = {'object_data': {
            'type': typ,
            'cuboid': {'name': '3d', 'val': [x, y, z, *yaw_quat(yaw), l, w, h]}}}
    path.write_text(json.dumps({'openlabel': {'frames': {str(fid): frame}}}))


def test_quat_to_yaw_roundtrip():
    for y in (-3.0, -1.0, 0.0, 0.7, 2.9):
        assert quat_to_yaw(*yaw_quat(y)) == pytest.approx(y, abs=1e-6)


def test_read_openlabel_boxes_parses_and_maps(tmp_path):
    f = tmp_path / "frame.json"
    write_frame(f, 0, [
        ("uuid-a", "CAR", 10.0, 5.0, 0.0, 4.5, 2.0, 1.6, 0.5),
        ("uuid-b", "PEDESTRIAN", -3.0, 2.0, 0.0, 0.8, 0.8, 1.8, 0.0),
        ("uuid-c", "VAN", 1.0, 1.0, 0.0, 5.0, 2.1, 2.2, 1.0),
    ])
    boxes = read_openlabel_boxes(f)
    assert len(boxes) == 3
    by = {b['track_id']: b for b in boxes}
    assert by['uuid-a']['label'] == 'car'
    assert by['uuid-b']['label'] == 'pedestrian'
    assert by['uuid-c']['label'] == 'car'      # VAN -> car
    assert by['uuid-c']['raw_type'] == 'VAN'   # original preserved
    assert np.allclose(by['uuid-a']['xyz'], [10, 5, 0])
    assert np.allclose(by['uuid-a']['lwh'], [4.5, 2.0, 1.6])
    assert by['uuid-a']['yaw'] == pytest.approx(0.5, abs=1e-6)


def test_class_map_covers_all_ten_categories():
    cats = {'CAR', 'TRUCK', 'TRAILER', 'VAN', 'MOTORCYCLE', 'BUS',
            'PEDESTRIAN', 'BICYCLE', 'EMERGENCY_VEHICLE', 'OTHER'}
    assert cats <= set(TUMTRAF_CLASS_MAP)
    assert TUMTRAF_CLASS_MAP['OTHER'] == 'unknown'


def test_load_trajectories_stitches_uuid_across_frames(tmp_path):
    # one car moving east over 8 frames; uuid is the track link
    for k in range(8):
        write_frame(tmp_path / f"{k:04d}.json", k,
                    [("car-1", "CAR", k * 1.0, 0.0, 0.0, 4.5, 2.0, 1.6, 0.0)],
                    timestamp=100.0 + k * 0.1)
    trajs = load_tumtraf_trajectories(tmp_path)
    assert len(trajs) == 1
    tr = trajs[0]
    assert tr.label == 'car'
    assert len(tr) == 8
    assert tr.extras['uuid'] == 'car-1'
    # timestamps honored, positions stitched in order
    assert tr.t[0] == pytest.approx(100.0)
    assert np.allclose(tr.xy[:, 0], np.arange(8.0))


def test_load_trajectories_drops_short_tracks(tmp_path):
    for k in range(3):
        write_frame(tmp_path / f"{k:04d}.json", k,
                    [("blip", "CAR", k, 0, 0, 4.5, 2, 1.6, 0)])
    assert load_tumtraf_trajectories(tmp_path, min_len=5) == []


def _write_pcd(path, xyz, intensity, binary):
    n = len(xyz)
    header = (
        "VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
        "TYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\nDATA {'binary' if binary else 'ascii'}\n")
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        if binary:
            for (x, y, z), i in zip(xyz, intensity):
                f.write(struct.pack('<ffff', x, y, z, i))
        else:
            for (x, y, z), i in zip(xyz, intensity):
                f.write(f"{x} {y} {z} {i}\n".encode('ascii'))


@pytest.mark.parametrize("binary", [False, True])
def test_read_pcd_ascii_and_binary(tmp_path, binary):
    xyz = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0], [0.0, 0.0, 1.5]])
    inten = np.array([10.0, 20.0, 30.0])
    p = tmp_path / "cloud.pcd"
    _write_pcd(p, xyz, inten, binary)
    pts = read_pcd(p, z_shift=0.5)
    assert pts.shape == (3, 5)
    assert np.allclose(pts[:, :2], xyz[:, :2], atol=1e-5)
    assert np.allclose(pts[:, 2], xyz[:, 2] + 0.5, atol=1e-5)
    assert np.allclose(pts[:, 3], inten, atol=1e-5)
    assert np.all(pts[:, 4] == 0)


def test_read_pcd_filters_nonfinite(tmp_path):
    xyz = np.array([[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0]])
    _write_pcd(tmp_path / "c.pcd", xyz, np.array([1.0, 2.0]), binary=True)
    pts = read_pcd(tmp_path / "c.pcd")
    assert len(pts) == 1
