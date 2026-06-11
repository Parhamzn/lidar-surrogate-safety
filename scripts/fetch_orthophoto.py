#!/usr/bin/env python
"""Fetch the DOP20 orthophoto tile around the LUMPI intersection.

Source: LGLN Lower Saxony open geodata WMS (ni_dop20, CC-BY), the same
imagery the LUMPI README references. The tile extent is computed from
georef.json so the image registers with the label frame.

Usage: python scripts/fetch_orthophoto.py [--span 300] [--px-per-m 5]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

WMS = ('https://opendata.lgln.niedersachsen.de/doorman/noauth/dop_wms'
       '?service=WMS&version=1.3.0&request=GetMap&layers=ni_dop20'
       '&styles=&crs=EPSG:25832&bbox={minE},{minN},{maxE},{maxN}'
       '&width={w}&height={h}&format=image/jpeg')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--georef', default='data/lumpi/georef.json')
    ap.add_argument('--span', type=float, default=300.0, help='tile size [m]')
    ap.add_argument('--px-per-m', type=float, default=5.0)
    ap.add_argument('--out', default='data/lumpi/orthophoto.jpg')
    args = ap.parse_args()

    g = json.load(open(args.georef))
    # label-frame origin in absolute UTM
    e0 = g['t'][0] + g['utm_offset'][0]
    n0 = g['t'][1] + g['utm_offset'][1]
    half = args.span / 2
    ext = dict(minE=e0 - half, minN=n0 - half, maxE=e0 + half, maxN=n0 + half)
    px = int(args.span * args.px_per_m)
    url = WMS.format(**ext, w=px, h=px)
    print('GET', url)
    # curl uses the system trust store (framework pythons on macOS often
    # lack CA certificates)
    subprocess.run(['curl', '-fsSL', '-o', args.out, url], check=True)

    meta = dict(extent_utm=[ext['minE'], ext['maxE'], ext['minN'], ext['maxN']],
                extent_label=[-half + (ext['minE'] - e0) + half,  # = -half
                              half, -half, half],
                label_origin_utm=[e0, n0], px=px, epsg=25832,
                source='LGLN ni_dop20 (CC-BY), via open WMS')
    # extent in label frame is simply +-half around 0 (rotation ~0.009 deg)
    meta['extent_label'] = [-half, half, -half, half]
    json.dump(meta, open(Path(args.out).with_suffix('.json'), 'w'), indent=1)
    print(f"wrote {args.out} ({px}x{px}px) and extent json; "
          f"label origin = UTM ({e0:.1f}, {n0:.1f})")


if __name__ == '__main__':
    main()
