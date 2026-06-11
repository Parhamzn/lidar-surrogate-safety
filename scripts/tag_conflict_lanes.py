#!/usr/bin/env python
"""Tag conflict events with the road facility they occurred on.

Uses the OpenDRIVE lane map to label every TTC/PET/HBE event with its
facility (driving lane, junction area, cycle path, sidewalk, ...),
writes conflicts_tagged.csv, prints facility cross-tabs, and renders a
facility map with the severe VRU conflicts overlaid (which doubles as a
visual check that the lane polygons align with the data).

Usage: python scripts/tag_conflict_lanes.py outputs/lumpi \
           data/lumpi/lumpi_lines_arcs.xodr
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from lidar_pilot.io.opendrive import LaneMap
from lidar_pilot.viz import load_orthophoto

VRU = {'pedestrian', 'bicycle', 'motorcycle', 'scooter'}
# saturated, semantically distinct facility styles that survive being
# drawn semi-transparent over an orthophoto; draw order puts the thin
# strips (cycle paths, sidewalks) on top of the wide carriageways
FACILITY_STYLE = {  # facility: (face color, alpha)
    'driving': ('#3d76c2', 0.40),
    'junction': ('#8a4fc8', 0.45),
    'median/other': ('#9e9e9e', 0.40),
    'shoulder': ('#8d7350', 0.50),
    'restricted': ('#e3d51f', 0.55),
    'parking': ('#34b8c4', 0.55),
    'sidewalk': ('#ff9d1c', 0.60),
    'biking': ('#11c24a', 0.70),
}
FACILITY_DRAW_ORDER = list(FACILITY_STYLE)


def main(outputs_dir: str, xodr_path: str):
    out_dir = Path(outputs_dir)
    lane_map = LaneMap(xodr_path)
    print(f'{len(lane_map.lanes)} lane polygons parsed')

    rows = list(csv.DictReader(open(out_dir / 'conflicts.csv')))
    for r in rows:
        r['facility'] = (lane_map.tag(float(r['x']), float(r['y']))
                         if r['x'] else 'unknown')

    with open(out_dir / 'conflicts_tagged.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out_dir / "conflicts_tagged.csv"}')

    # ---- cross-tab: facility x metric ----
    facilities = sorted({r['facility'] for r in rows},
                        key=lambda v: -sum(1 for r in rows if r['facility'] == v))
    print(f'\n{"facility":<16}' + ''.join(f'{m:>8}' for m in ('TTC', 'PET', 'HBE'))
          + f'{"total":>8}')
    for fac in facilities:
        cnt = Counter(r['metric'] for r in rows if r['facility'] == fac)
        total = sum(cnt.values())
        print(f'{fac:<16}' + ''.join(f'{cnt.get(m, 0):>8}'
                                     for m in ('TTC', 'PET', 'HBE'))
              + f'{total:>8}')

    # ---- VRU-involved conflicts (TTC/PET with at least one VRU) ----
    vru_rows = [r for r in rows if r['metric'] in ('TTC', 'PET')
                and (r['class_a'] in VRU or r['class_b'] in VRU)]
    print(f'\nVRU-involved conflicts (n={len(vru_rows)}):')
    for fac, n in Counter(r['facility'] for r in vru_rows).most_common():
        print(f'  {fac:<16}{n:>5}  ({100 * n / len(vru_rows):.0f}%)')

    # ---- facility map + severe VRU conflicts ----
    fig, ax = plt.subplots(figsize=(10.5, 9))
    ortho, ortho_ext = load_orthophoto()
    if ortho is not None:
        ax.imshow(ortho, extent=ortho_ext, zorder=0)
    by_type = defaultdict(list)
    for ln in lane_map.lanes:
        fac = ('junction' if ln.lane_type == 'driving' and ln.in_junction
               else ('median/other' if ln.lane_type == 'none' else ln.lane_type))
        by_type[fac].append(ln.polygon.vertices)
    for z, fac in enumerate(FACILITY_DRAW_ORDER):
        if fac not in by_type:
            continue
        face, alpha = FACILITY_STYLE[fac]
        ax.add_collection(PolyCollection(
            by_type[fac], facecolors=face, alpha=alpha,
            edgecolors=face, linewidths=0.8, zorder=2 + 0.1 * z,
            label=f'{fac} ({len(by_type[fac])})'))

    vru_sev = sorted(vru_rows, key=lambda r: float(r['value']))[:40]
    x = [float(r['x']) for r in vru_sev]
    y = [float(r['y']) for r in vru_sev]
    # crosses keep the spot itself visible on the orthophoto; the white
    # pass underneath is a halo for contrast on busy imagery
    ax.scatter(x, y, s=110, marker='x', color='white', linewidths=4.0,
               zorder=5)
    ax.scatter(x, y, s=90, marker='x', color='crimson', linewidths=1.8,
               zorder=6, label='40 most severe VRU conflicts')

    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]'), ax.set_ylabel('y [m]')
    ax.set_title('Road facilities (OpenDRIVE) and severe VRU conflicts')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.92)
    fig.tight_layout()
    Path('figures').mkdir(exist_ok=True)
    fig.savefig('figures/lane_facility_map_Measurement5.png', dpi=200)
    print('wrote figures/lane_facility_map_Measurement5.png')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'outputs/lumpi',
         sys.argv[2] if len(sys.argv) > 2 else 'data/lumpi/lumpi_lines_arcs.xodr')
