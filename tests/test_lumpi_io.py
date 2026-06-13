"""LUMPI loader tests on a synthetic Label.csv."""

import numpy as np
import pytest

from lidar_pilot.io import load_lumpi_trajectories

HEADER = ("time,object id, 2d rectangle: top left x,top left y, width,height,"
          "score,class_id,visibility,3D box: center x,y,z,length, width ,"
          "height,heading,[optional]\n")


def make_csv(tmp_path, rows):
    p = tmp_path / "Label.csv"
    p.write_text(HEADER + "\n".join(rows) + "\n")
    return p


def row(t, oid, cls, x, y, l=4.5, w=2.0, h=1.7, heading=0.0):
    return (f"{t:.3f},{oid},0,0,1,1,0.9,{cls},100,"
            f"{x:.3f},{y:.3f},-1.5,{l},{w},{h},{heading}")


def test_basic_parsing(tmp_path):
    rows = [row(k * 0.1, 7, 1, 10.0 * k * 0.1, 0.0) for k in range(20)]
    trajs = load_lumpi_trajectories(make_csv(tmp_path, rows))
    assert len(trajs) == 1
    tr = trajs[0]
    assert tr.track_id == 7
    assert tr.label == "car"
    assert len(tr) == 20
    # 10 m/s motion east
    v = (tr.xy[-1] - tr.xy[0]) / (tr.t[-1] - tr.t[0])
    assert v[0] == pytest.approx(10.0, abs=0.01)
    assert tr.size_lw == pytest.approx([4.5, 2.0])


def test_class_majority_vote_and_vru_classes(tmp_path):
    rows = [row(k * 0.1, 1, 0 if k < 12 else 2, 0.5 * k, 0, l=0.8, w=0.7, h=1.8)
            for k in range(20)]
    rows += [row(k * 0.1, 2, 6, 0, 0.5 * k, l=1.9, w=0.9, h=1.7)
             for k in range(20)]
    trajs = {tr.track_id: tr for tr in load_lumpi_trajectories(make_csv(tmp_path, rows))}
    assert trajs[1].label == "pedestrian"   # 12 of 20 votes
    assert trajs[2].label == "scooter"      # undocumented class 6


def test_short_and_duplicate_rows(tmp_path):
    rows = [row(0.0, 3, 1, 0, 0), row(0.0, 3, 1, 0, 0), row(0.1, 3, 1, 1, 0)]
    rows += [row(k * 0.1, 4, 1, k * 1.0, 5.0) for k in range(10)]
    trajs = load_lumpi_trajectories(make_csv(tmp_path, rows), min_len=5)
    assert [tr.track_id for tr in trajs] == [4]
    assert np.all(np.diff(trajs[0].t) > 0)


def test_class7_maps_to_car(tmp_path):
    """Measurement4's undocumented class id 7 (a single van-sized track)
    must map to 'car', not fall through to 'unknown'."""
    from lidar_pilot.io import load_lumpi_trajectories
    csv = tmp_path / "label.csv"
    header = "time,id,x,y,w,h,score,class,vis,cx,cy,cz,l,bw,bh,head\n"
    rows = "".join(
        f"{i*0.1:.1f},1,0,0,0,0,0.9,7,1,{i*0.5:.2f},0,0,4.88,2.1,2.17,0\n"
        for i in range(8))
    csv.write_text(header + rows)
    trajs = load_lumpi_trajectories(csv)
    assert len(trajs) == 1
    assert trajs[0].label == "car"
