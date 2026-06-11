"""Minimal OpenDRIVE lane map for facility tagging.

Parses an OpenDRIVE 1.x file (line/arc reference geometry, e.g. the
LUMPI `lumpi_lines_arcs.xodr`) into per-lane ground polygons, and tags
points with the lane type they fall in (driving, biking, sidewalk, ...).
Junction-internal driving lanes are reported as "junction".

Scope: enough of the standard for tagging, not a general importer —
supported: <line>/<arc> plan view, <laneOffset> polynomials, multiple
<laneSection>s, per-lane <width> polynomials. Not supported: spirals,
poly3 geometry, superelevation (irrelevant for 2D tagging).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path as FsPath

import numpy as np
from matplotlib.path import Path as MplPath

SAMPLE_DS = 0.5
# when polygons overlap, the more specific facility wins
TYPE_PRIORITY = ['biking', 'sidewalk', 'restricted', 'shoulder',
                 'driving', 'parking', 'none']


@dataclass
class Lane:
    road_id: str
    lane_id: int
    lane_type: str
    in_junction: bool
    polygon: MplPath
    bbox: tuple  # (xmin, ymin, xmax, ymax)


def _poly3(records, s):
    """Evaluate the last polynomial record starting at or before s.
    records: list of (s_start, a, b, c, d), sorted by s_start."""
    rec = records[0]
    for r in records:
        if r[0] <= s + 1e-9:
            rec = r
        else:
            break
    ds = s - rec[0]
    return rec[1] + rec[2] * ds + rec[3] * ds ** 2 + rec[4] * ds ** 3


def _sample_reference(geoms, s_values):
    """Reference-line position and heading at the requested s values."""
    out = np.zeros((len(s_values), 3))
    for i, s in enumerate(s_values):
        g = geoms[0]
        for cand in geoms:
            if cand['s'] <= s + 1e-9:
                g = cand
            else:
                break
        ds = s - g['s']
        x, y, hdg = g['x'], g['y'], g['hdg']
        if g['curv'] == 0.0:
            out[i] = (x + ds * np.cos(hdg), y + ds * np.sin(hdg), hdg)
        else:
            k = g['curv']
            out[i] = (x + (np.sin(hdg + k * ds) - np.sin(hdg)) / k,
                      y - (np.cos(hdg + k * ds) - np.cos(hdg)) / k,
                      hdg + k * ds)
    return out


def parse_lanes(xodr_path: str | FsPath) -> list[Lane]:
    root = ET.parse(xodr_path).getroot()
    lanes: list[Lane] = []

    for road in root.iter('road'):
        length = float(road.get('length'))
        in_junction = road.get('junction', '-1') != '-1'
        geoms = []
        for g in road.find('planView').iter('geometry'):
            curv = 0.0
            if g.find('arc') is not None:
                curv = float(g.find('arc').get('curvature'))
            elif g.find('line') is None:
                continue  # unsupported primitive (none in the LUMPI map)
            geoms.append(dict(s=float(g.get('s')), x=float(g.get('x')),
                              y=float(g.get('y')), hdg=float(g.get('hdg')),
                              curv=curv))
        if not geoms:
            continue
        geoms.sort(key=lambda d: d['s'])

        lanes_el = road.find('lanes')
        offsets = [(float(o.get('s')), float(o.get('a')), float(o.get('b')),
                    float(o.get('c')), float(o.get('d')))
                   for o in lanes_el.findall('laneOffset')] or [(0, 0, 0, 0, 0)]
        offsets.sort()

        sections = lanes_el.findall('laneSection')
        sec_starts = [float(s.get('s')) for s in sections]
        sec_ends = sec_starts[1:] + [length]

        for sec, s0, s1 in zip(sections, sec_starts, sec_ends):
            if s1 - s0 < SAMPLE_DS:
                continue
            s_values = np.linspace(s0, s1, max(int((s1 - s0) / SAMPLE_DS), 2))
            ref = _sample_reference(geoms, s_values)
            normal = np.column_stack([-np.sin(ref[:, 2]), np.cos(ref[:, 2])])
            center_t = np.array([_poly3(offsets, s) for s in s_values])

            for side, sign in (('left', 1), ('right', -1)):
                side_el = sec.find(side)
                if side_el is None:
                    continue
                side_lanes = sorted(
                    side_el.findall('lane'),
                    key=lambda ln: abs(int(ln.get('id'))))
                inner_t = center_t.copy()
                for ln in side_lanes:
                    widths = [(float(w.get('sOffset')), float(w.get('a')),
                               float(w.get('b')), float(w.get('c')),
                               float(w.get('d')))
                              for w in ln.findall('width')]
                    if not widths:
                        continue
                    widths.sort()
                    w = np.array([_poly3(widths, s - s0) for s in s_values])
                    outer_t = inner_t + sign * w
                    inner_xy = ref[:, :2] + normal * inner_t[:, None]
                    outer_xy = ref[:, :2] + normal * outer_t[:, None]
                    poly = np.vstack([inner_xy, outer_xy[::-1]])
                    lanes.append(Lane(
                        road_id=road.get('id'),
                        lane_id=int(ln.get('id')),
                        lane_type=ln.get('type', 'none'),
                        in_junction=in_junction,
                        polygon=MplPath(poly),
                        bbox=(poly[:, 0].min(), poly[:, 1].min(),
                              poly[:, 0].max(), poly[:, 1].max()),
                    ))
                    inner_t = outer_t
    return lanes


class LaneMap:
    def __init__(self, xodr_path: str | FsPath):
        self.lanes = parse_lanes(xodr_path)
        boxes = np.array([ln.bbox for ln in self.lanes])
        # overall mapped extent (the map covers only the junction vicinity)
        self.extent = (boxes[:, 0].min(), boxes[:, 1].min(),
                       boxes[:, 2].max(), boxes[:, 3].max())

    def in_extent(self, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.extent
        return x0 <= x <= x1 and y0 <= y <= y1

    def tag(self, x: float, y: float) -> str:
        """Facility at (x, y): lane type, 'junction' for junction-internal
        driving lanes, or 'off-map'."""
        hits = [ln for ln in self.lanes
                if ln.bbox[0] <= x <= ln.bbox[2]
                and ln.bbox[1] <= y <= ln.bbox[3]
                and ln.polygon.contains_point((x, y))]
        if not hits:
            return 'unmapped island' if self.in_extent(x, y) else 'beyond map'
        best = min(hits, key=lambda ln: TYPE_PRIORITY.index(ln.lane_type)
                   if ln.lane_type in TYPE_PRIORITY else 99)
        if best.lane_type == 'driving' and best.in_junction:
            return 'junction'
        if best.lane_type == 'none':
            return 'median/other'
        return best.lane_type
