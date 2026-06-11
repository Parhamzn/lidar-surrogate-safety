"""LaneMap tests on a synthetic minimal OpenDRIVE file."""

import pytest

from lidar_pilot.io.opendrive import LaneMap

MINIMAL_XODR = """<?xml version="1.0"?>
<OpenDRIVE>
  <road name="r" length="20.0" id="1" junction="-1">
    <planView>
      <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="20.0"><line/></geometry>
    </planView>
    <lanes>
      <laneSection s="0.0">
        <left>
          <lane id="1" type="biking">
            <width sOffset="0.0" a="2.0" b="0.0" c="0.0" d="0.0"/>
          </lane>
        </left>
        <center><lane id="0" type="none"/></center>
        <right>
          <lane id="-1" type="driving">
            <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
          </lane>
          <lane id="-2" type="sidewalk">
            <width sOffset="0.0" a="1.5" b="0.0" c="0.0" d="0.0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
"""


@pytest.fixture()
def lane_map(tmp_path):
    p = tmp_path / 'mini.xodr'
    p.write_text(MINIMAL_XODR)
    return LaneMap(p)


def test_lane_polygons_parsed(lane_map):
    assert len(lane_map.lanes) == 3  # biking, driving, sidewalk


def test_tagging(lane_map):
    # road runs along +x; left = +y, right = -y
    assert lane_map.tag(10.0, 1.0) == 'biking'
    assert lane_map.tag(10.0, -1.5) == 'driving'
    assert lane_map.tag(10.0, -4.2) == 'sidewalk'   # 3.5 .. 5.0 below center
    # the minimal fixture's lanes tile its whole bounding box, so points
    # off the lanes are also outside the extent -> 'beyond map'
    assert lane_map.tag(10.0, 8.0) == 'beyond map'
    assert lane_map.tag(40.0, 0.0) == 'beyond map'  # beyond road end
