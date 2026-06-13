#!/usr/bin/env python
"""Georeference the TUMTraf s110 label frame to UTM32N + fetch an orthophoto.

Unlike LUMPI (whose sensor origins were sphere-fitted), TUMTraf's OpenLABEL
already carries the rigid chain south_lidar -> s110_base -> hd_map_origin,
and hd_map_origin is a local z-up (ENU-style) frame. So a label point p
maps to UTM by
    UTM(p) = sensor_utm + Rz(residual_deg) @ (R @ p)[:2]
where R is the south_lidar->hd_map rotation (translation cancels: it only
places the sensor). What the OpenLABEL does NOT give is where hd_map sits
in UTM, nor whether its x/y axes are exactly East/North. Those two unknowns
(sensor_utm from the gantry location, residual_deg from the East/North
alignment) are pinned by overlaying LiDAR ground points on the fetched
orthophoto and nudging until the roads register, the same visual check
used for LUMPI.

Writes data/tumtraf/georef.json (R, sensor_utm, residual_deg, epsg),
data/tumtraf/orthophoto.jpg (+ .json extent) and, with --pcd, an overlay
PNG for the visual check.

Usage (iterate lat/lon and residual until the overlay registers):
  python scripts/solve_tumtraf_georef.py --lat 48.2501 --lon 11.6360 \
      --residual-deg 0 --pcd /tmp/sample.pcd
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from pyproj import Transformer

WMS = ('https://geoservices.bayern.de/od/wms/dop/v1/dop20'
       '?service=WMS&version=1.3.0&request=GetMap&layers=by_dop20c'
       '&styles=&crs=EPSG:25832&bbox={minE},{minN},{maxE},{maxN}'
       '&width={w}&height={h}&format=image/jpeg')
# EPSG:25832 bbox axis order is Easting,Northing here (as for the LUMPI WMS).

LL_TO_UTM = Transformer.from_crs('EPSG:4326', 'EPSG:25832', always_xy=True)


def lidar_to_hdmap_rotation(label_json):
    cs = json.load(open(label_json))['openlabel']['coordinate_systems']
    def mat(n): return np.array(cs[n]['pose_wrt_parent']['matrix4x4'], float).reshape(4, 4)
    T = mat('s110_base') @ mat('s110_lidar_ouster_south')
    return T[:3, :3]


def read_pcd_xyz(path):
    from lidar_pilot.io.tumtraf import read_pcd
    return read_pcd(path)[:, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label-json', default=None,
                    help='any OpenLABEL frame (for the transform chain)')
    ap.add_argument('--lat', type=float, required=True, help='gantry latitude')
    ap.add_argument('--lon', type=float, required=True, help='gantry longitude')
    ap.add_argument('--residual-deg', type=float, default=0.0,
                    help='extra yaw to align hd_map x/y to UTM East/North')
    ap.add_argument('--span', type=float, default=160.0, help='tile size [m]')
    ap.add_argument('--px-per-m', type=float, default=5.0)
    ap.add_argument('--pcd', default=None, help='a .pcd to overlay for checking')
    ap.add_argument('--ground-z', type=float, default=-7.25,
                    help='label-frame ground height (for road points)')
    ap.add_argument('--data-dir', default='data/tumtraf')
    args = ap.parse_args()

    label_json = args.label_json or f'{args.data_dir}/sample_label.json'
    R = lidar_to_hdmap_rotation(label_json)
    th = np.radians(args.residual_deg)
    Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    e0, n0 = LL_TO_UTM.transform(args.lon, args.lat)   # sensor UTM (E, N)
    sensor_utm = np.array([e0, n0])

    def to_utm(xyz):
        d = (R @ np.asarray(xyz).T).T[:, :2]   # ENU displacement from sensor
        return sensor_utm + (Rz @ d.T).T

    out = Path(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    georef = dict(epsg=25832, sensor_utm=[float(e0), float(n0)],
                  R=R.tolist(), residual_deg=args.residual_deg,
                  note='UTM(p)=sensor_utm + Rz(residual_deg)@(R@p)[:2]; '
                       'p in s110_lidar_ouster_south metres')
    json.dump(georef, open(out / 'georef.json', 'w'), indent=1)

    half = args.span / 2
    ext = dict(minE=e0 - half, minN=n0 - half, maxE=e0 + half, maxN=n0 + half)
    px = int(args.span * args.px_per_m)
    url = WMS.format(**ext, w=px, h=px)
    ortho = out / 'orthophoto.jpg'
    subprocess.run(['curl', '-fsSL', '-o', str(ortho), url], check=True)
    json.dump(dict(extent_utm=[ext['minE'], ext['maxE'], ext['minN'], ext['maxN']],
                   sensor_utm=[e0, n0], px=px, epsg=25832,
                   source='Bayerische Vermessungsverwaltung DOP20 (CC BY 4.0)'),
              open(ortho.with_suffix('.json'), 'w'), indent=1)
    print(f'sensor UTM ({e0:.1f}, {n0:.1f}) | tile {args.span} m, {px}px | residual {args.residual_deg} deg')

    if args.pcd:
        import matplotlib.pyplot as plt
        pts = read_pcd_xyz(args.pcd)
        ground = pts[np.abs(pts[:, 2] - args.ground_z) < 1.0]
        if len(ground) > 40000:
            ground = ground[np.random.default_rng(0).choice(len(ground), 40000, replace=False)]
        utm = to_utm(ground)
        img = plt.imread(str(ortho))
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, extent=[ext['minE'], ext['maxE'], ext['minN'], ext['maxN']],
                  origin='upper')
        ax.scatter(utm[:, 0], utm[:, 1], s=0.5, c='cyan', alpha=0.4)
        ax.plot(e0, n0, 'r+', ms=14, mew=2)
        ax.set_title(f'LiDAR ground points (cyan) over DOP20 | '
                     f'lat {args.lat} lon {args.lon} residual {args.residual_deg} deg')
        ax.set_xlabel('UTM32N Easting'), ax.set_ylabel('Northing')
        fig.savefig(out / 'georef_overlay.png', dpi=130, bbox_inches='tight')
        print(f"wrote {out/'georef_overlay.png'} ({len(ground)} ground points)")


if __name__ == '__main__':
    main()
