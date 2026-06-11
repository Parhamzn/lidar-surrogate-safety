"""LUMPI label loader (Leibniz University Multi-Perspective Intersection).

Label.csv format (per the dataset README, one row per object per 10 Hz
frame):

    time, object id, 2D box (tl.x, tl.y, w, h), score, class_id,
    visibility, 3D box (center x, y, z, length, width, height, heading),
    [optional extra columns]

Coordinates are metres in the measurement's local frame (the per-sensor
extrinsics in meta.json map to UTM; a local metric frame is all the
conflict metrics need). Time is seconds from measurement start.

Class ids: the README documents the COCO-subset classes 1-based
(person=1 ... truck=6), but the shipped CSVs are 0-based, verified
against median box dimensions per class (class 0 is person-sized, 1 is
car-sized, ...). Class 6 is undocumented and bike-sized; we call it
"scooter" and treat it as a VRU.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from lidar_pilot.trajectory import Trajectory

LUMPI_CLASSES = {
    0: "pedestrian", 1: "car", 2: "bicycle", 3: "motorcycle",
    4: "bus", 5: "truck", 6: "scooter",
}


def load_lumpi_trajectories(csv_path: str | Path,
                            min_len: int = 5) -> list[Trajectory]:
    """Parse a LUMPI Label.csv into one Trajectory per tracked object.

    The object's class is taken as the per-track majority vote (labels can
    flicker frame to frame), its footprint as the median box size. Tracks
    shorter than min_len samples are dropped.
    """
    cols = np.loadtxt(csv_path, delimiter=",", skiprows=1,
                      usecols=range(16), ndmin=2)
    if cols.size == 0:
        return []
    t_all = cols[:, 0]
    ids = cols[:, 1].astype(int)
    cls_all = cols[:, 7].astype(int)
    centers = cols[:, 9:12]
    lwh = cols[:, 12:15]
    heading = cols[:, 15]

    by_id: dict[int, list[int]] = defaultdict(list)
    for row, oid in enumerate(ids):
        by_id[oid].append(row)

    out: list[Trajectory] = []
    for oid, rows in by_id.items():
        rows = np.asarray(rows)
        order = np.argsort(t_all[rows], kind="stable")
        rows = rows[order]
        # collapse duplicate timestamps (keep the first occurrence)
        keep = np.concatenate([[True], np.diff(t_all[rows]) > 1e-9])
        rows = rows[keep]
        if rows.size < min_len:
            continue
        votes = np.bincount(cls_all[rows])
        label = LUMPI_CLASSES.get(int(np.argmax(votes)), "unknown")
        out.append(Trajectory(
            track_id=int(oid),
            t=t_all[rows],
            xy=centers[rows, :2],
            label=label,
            yaw=heading[rows],
            size_lw=np.median(lwh[rows, :2], axis=0),
            extras={"z": centers[rows, 2],
                    "lwh_median": np.median(lwh[rows], axis=0)},
        ))
    return out
