"""Shared plotting constants and helpers (matplotlib-based, BEV figures)."""

import json
from pathlib import Path

CLASS_COLORS = {
    'car': '#4287f5', 'truck': '#f5a742', 'trailer': '#b5651d',
    'bus': '#f542dd', 'construction_vehicle': '#828282',
    'bicycle': '#42f560', 'motorcycle': '#20c4a0',
    'pedestrian': '#f54242', 'unknown': '#aaaaaa',
}


def load_orthophoto(jpg_path='data/lumpi/orthophoto.jpg'):
    """(image array, label-frame extent) or (None, None) if not fetched.

    The extent json is written by scripts/fetch_orthophoto.py; the image
    registers with the label frame via the georef solved from sensor
    positions (rotation ~0.01 deg, treated as axis-aligned).
    """
    jpg, meta = Path(jpg_path), Path(jpg_path).with_suffix('.json')
    if not (jpg.exists() and meta.exists()):
        return None, None
    import matplotlib.image as mpimg
    return mpimg.imread(jpg), json.load(open(meta))['extent_label']
