"""TUMTraf Intersection adapter (TUM Traffic Dataset, OpenLABEL format).

A roadside intersection dataset (Munich/A9 testbed) used here as an
out-of-distribution test site for the LUMPI-fine-tuned detector: "trained
on Hanover, tested on Munich". Two LiDARs + two cameras on gantries,
~10 Hz, 10 object classes, CC BY-NC-ND 4.0.

Two label conventions differ from LUMPI and are handled here:

* **Labels are OpenLABEL JSON, one file per frame.** Per the dev kit
  (conversion_openlabel_to_nuscenes.py), the schema is::

      data["openlabel"]["frames"][fid]["objects"][uuid]
          ["object_data"]["type"]          -> class string (e.g. "CAR")
          ["object_data"]["cuboid"]["val"] -> [x, y, z,
                                               qx, qy, qz, qw,   # quaternion
                                               l, w, h]          # dimensions

  The object ``uuid`` is the persistent track id across a sequence's
  per-frame files; heading is recovered from the quaternion's z-rotation.

* **Point clouds are .pcd, not .ply.** ``read_pcd`` reads ascii and
  uncompressed-binary PCD (x, y, z, intensity); binary_compressed needs
  open3d and raises a clear error.

Class mapping to the pipeline's canonical set is a documented dict; per
the LUMPI lesson it MUST be re-verified against median box dimensions on
the real data before trusting it (``summarize_class_dims`` helps).
Coordinate frame: labels are in the registered ``s110_base`` metric frame
(metres); confirm the chosen point cloud is co-registered before eval.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lidar_pilot.trajectory import Trajectory

# TUMTraf's 10 categories -> the pipeline's canonical classes. VAN and
# EMERGENCY_VEHICLE have no own bucket and are car-like; TRAILER is
# truck-like; OTHER is unusable. VERIFY against box dimensions on real
# data before trusting (summarize_class_dims).
TUMTRAF_CLASS_MAP = {
    'CAR': 'car',
    'VAN': 'car',
    'EMERGENCY_VEHICLE': 'car',
    'TRUCK': 'truck',
    'TRAILER': 'truck',
    'BUS': 'bus',
    'MOTORCYCLE': 'motorcycle',
    'BICYCLE': 'bicycle',
    'PEDESTRIAN': 'pedestrian',
    'OTHER': 'unknown',
}


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Heading (rotation about +z) from a unit quaternion, in radians."""
    return float(np.arctan2(2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz)))


def _frame_objects(doc: dict) -> tuple[str, dict]:
    """(frame_id, objects-dict) for a single-frame OpenLABEL document."""
    frames = doc['openlabel']['frames']
    fid = next(iter(frames))
    return fid, frames[fid].get('objects', {}) or {}


def _frame_time(doc: dict, fid: str, fallback_index: int, fps: float) -> float:
    """Seconds; prefer the frame's recorded timestamp, else index / fps."""
    props = doc['openlabel']['frames'][fid].get('frame_properties', {})
    ts = props.get('timestamp')
    if ts is not None:
        try:
            return float(ts)
        except (TypeError, ValueError):
            pass
    return fallback_index / fps


def read_openlabel_boxes(json_path: str | Path,
                         class_map: dict | None = None) -> list[dict]:
    """Per-frame ground-truth boxes from one OpenLABEL JSON.

    Returns one dict per object: ``label`` (mapped class), ``raw_type``
    (original TUMTraf category), ``xyz``, ``lwh``, ``yaw``, ``track_id``.
    Used by the detection evaluation as ground truth.
    """
    class_map = class_map or TUMTRAF_CLASS_MAP
    doc = json.load(open(json_path))
    _, objects = _frame_objects(doc)
    out = []
    for uuid, obj in objects.items():
        od = obj.get('object_data', {})
        cub = od.get('cuboid')
        if not cub:
            continue
        v = cub['val']
        raw_type = (od.get('type') or obj.get('type') or 'OTHER').upper()
        out.append(dict(
            label=class_map.get(raw_type, 'unknown'),
            raw_type=raw_type,
            xyz=np.array(v[:3], float),
            lwh=np.array(v[7:10], float),
            yaw=quat_to_yaw(*v[3:7]),
            track_id=uuid,
        ))
    return out


def load_tumtraf_trajectories(seq_dir: str | Path,
                              fps: float = 10.0,
                              min_len: int = 5,
                              class_map: dict | None = None) -> list[Trajectory]:
    """Stitch a sequence's per-frame OpenLABEL files into trajectories.

    Objects are linked across frames by their OpenLABEL uuid (the dataset's
    track id). Sequences are short (30-120 s), so this is mainly for
    tracking sanity checks; the dataset's primary use here is per-frame
    detection evaluation (``read_openlabel_boxes``).
    """
    class_map = class_map or TUMTRAF_CLASS_MAP
    files = sorted(Path(seq_dir).glob('*.json'))
    by_id: dict[str, list[tuple]] = defaultdict(list)
    cls_votes: dict[str, list[str]] = defaultdict(list)
    for k, jp in enumerate(files):
        doc = json.load(open(jp))
        fid, objects = _frame_objects(doc)
        t = _frame_time(doc, fid, k, fps)
        for uuid, obj in objects.items():
            od = obj.get('object_data', {})
            cub = od.get('cuboid')
            if not cub:
                continue
            v = cub['val']
            by_id[uuid].append((t, v[0], v[1], v[2],
                                quat_to_yaw(*v[3:7]), v[7], v[8]))
            cls_votes[uuid].append((od.get('type') or obj.get('type')
                                    or 'OTHER').upper())

    out: list[Trajectory] = []
    for uuid, rows in by_id.items():
        if len(rows) < min_len:
            continue
        rows.sort(key=lambda r: r[0])
        a = np.asarray(rows, float)
        t = a[:, 0]
        keep = np.concatenate([[True], np.diff(t) > 1e-9])  # dedup times
        a = a[keep]
        if len(a) < min_len:
            continue
        votes = cls_votes[uuid]
        raw = max(set(votes), key=votes.count)
        out.append(Trajectory(
            track_id=abs(hash(uuid)) % (10 ** 9),
            t=a[:, 0],
            xy=a[:, 1:3],
            label=class_map.get(raw, 'unknown'),
            yaw=a[:, 4],
            size_lw=np.median(a[:, 5:7], axis=0),
            extras={'z': a[:, 3], 'uuid': uuid, 'raw_type': raw},
        ))
    return out


# --------------------------------------------------------------- PCD I/O --

def read_pcd(pcd_path: str | Path, z_shift: float = 0.0) -> np.ndarray:
    """Read an ascii or uncompressed-binary PCD into (N, 5):
    [x, y, z + z_shift, intensity, 0]. The trailing 0 is the per-point
    sweep dt the detector expects for a single static frame.

    binary_compressed is not handled here (raises with a hint to open3d);
    the TUMTraf Ouster clouds ship as binary, which is supported.
    """
    path = Path(pcd_path)
    with open(path, 'rb') as f:
        fields, sizes, types, counts = [], [], [], []
        npts, data_fmt = 0, 'ascii'
        while True:
            line = f.readline().decode('ascii', 'replace').strip()
            if not line or line.startswith('#'):
                continue
            key, *vals = line.split()
            key = key.upper()
            if key == 'FIELDS':
                fields = vals
            elif key == 'SIZE':
                sizes = [int(x) for x in vals]
            elif key == 'TYPE':
                types = vals
            elif key == 'COUNT':
                counts = [int(x) for x in vals]
            elif key == 'POINTS':
                npts = int(vals[0])
            elif key == 'WIDTH' and npts == 0:
                npts = int(vals[0])
            elif key == 'DATA':
                data_fmt = vals[0].lower()
                break
        counts = counts or [1] * len(fields)
        if data_fmt == 'binary_compressed':
            raise NotImplementedError(
                f'{path.name}: binary_compressed PCD needs open3d/pypcd; '
                'convert with the dev kit or `pcl_convert_pcd_ascii_binary`.')

        idx = {name: i for i, name in enumerate(fields)}
        need = [idx.get(k) for k in ('x', 'y', 'z')]
        if any(i is None for i in need):
            raise ValueError(f'{path.name}: PCD lacks x/y/z fields {fields}')
        i_int = idx.get('intensity', idx.get('i'))

        if data_fmt == 'ascii':
            arr = np.loadtxt(f, ndmin=2)
            cols = [sum(counts[:k]) for k in range(len(fields))]
            xyz = arr[:, [cols[i] for i in need]]
            inten = arr[:, cols[i_int]] if i_int is not None else np.zeros(len(arr))
        else:  # uncompressed binary: one interleaved record per point
            np_types = {('F', 4): '<f4', ('F', 8): '<f8',
                        ('U', 1): '<u1', ('U', 2): '<u2', ('U', 4): '<u4',
                        ('I', 1): '<i1', ('I', 2): '<i2', ('I', 4): '<i4'}
            dt = []
            for name, t, s, c in zip(fields, types, sizes, counts):
                base = np_types[(t.upper(), s)]
                dt += [(f'{name}_{j}', base) for j in range(c)] if c > 1 \
                    else [(name, base)]
            rec = np.frombuffer(f.read(npts * sum(sizes[k] * counts[k]
                                for k in range(len(fields)))), dtype=np.dtype(dt))
            xyz = np.column_stack([rec['x'], rec['y'], rec['z']]).astype(np.float32)
            inten = (rec['intensity'].astype(np.float32)
                     if 'intensity' in rec.dtype.names else np.zeros(len(rec), np.float32))

    pts = np.column_stack([
        xyz[:, 0], xyz[:, 1], xyz[:, 2] + z_shift,
        inten, np.zeros(len(xyz), np.float32)]).astype(np.float32)
    return pts[np.isfinite(pts[:, :3]).all(axis=1)]


def camera_projection_matrix(json_path: str | Path, camera_id: str) -> np.ndarray:
    """3x4 matrix projecting south-LiDAR homogeneous points to image pixels.

    Calibration is embedded in the OpenLABEL: the camera intrinsics
    (``streams[cam].stream_properties.intrinsics_pinhole.camera_matrix_3x4``)
    and the camera's ``coordinate_systems[cam].pose_wrt_parent``. Despite
    the OpenLABEL spec's child->parent convention, TUMTraf stores this pose
    as the LiDAR->camera transform (verified empirically: K·pose lands GT
    boxes on the vehicles, K·inv(pose) does not). The projection is
    therefore K · pose — lidar point (4,) homogeneous -> pixel (3,).
    """
    ol = json.load(open(json_path))['openlabel']
    K = np.array(ol['streams'][camera_id]['stream_properties']
                 ['intrinsics_pinhole']['camera_matrix_3x4'], float)        # 3x4
    pose_lidar_to_cam = np.array(
        ol['coordinate_systems'][camera_id]['pose_wrt_parent']['matrix4x4'],
        float).reshape(4, 4)
    return K @ pose_lidar_to_cam                                            # 3x4


def project_lidar_points(P: np.ndarray, xyz: np.ndarray,
                         min_depth: float = 0.5):
    """Project (N,3) LiDAR points with a 3x4 matrix. Returns (uv (M,2),
    depth (M,), mask) for points in front of the camera (depth>min_depth)."""
    homog = np.column_stack([xyz, np.ones(len(xyz))])
    cam = (P @ homog.T).T                       # (N, 3): [u*z, v*z, z]
    depth = cam[:, 2]
    front = depth > min_depth
    uv = cam[front, :2] / depth[front, None]
    return uv, depth[front], front


def summarize_class_dims(json_paths) -> dict[str, dict]:
    """Median box L/W/H per raw TUMTraf category — the empirical check that
    the class mapping is sane (a 'BICYCLE' must be bike-sized, etc.)."""
    dims: dict[str, list] = defaultdict(list)
    for jp in json_paths:
        for b in read_openlabel_boxes(jp):
            dims[b['raw_type']].append(b['lwh'])
    out = {}
    for raw, lst in dims.items():
        a = np.asarray(lst, float)
        out[raw] = dict(n=len(a), median_lwh=np.median(a, axis=0).round(2).tolist(),
                        mapped=TUMTRAF_CLASS_MAP.get(raw, 'unknown'))
    return out
