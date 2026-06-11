"""Minimal MMDetection3D dataset for LUMPI roadside frames.

Reads the info pickles written by scripts/prepare_lumpi_training.py:
.bin point files with 5 features and bottom-center 7-DoF boxes in the
(z-shifted) local intersection frame.
"""

from __future__ import annotations

import numpy as np

from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes


@DATASETS.register_module()
class LumpiDataset(Det3DDataset):
    METAINFO = {
        'classes': ('car', 'truck', 'bus', 'pedestrian', 'bicycle',
                    'motorcycle', 'scooter'),
    }

    def parse_ann_info(self, info: dict) -> dict:
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict(
                gt_bboxes_3d=np.zeros((0, 7), dtype=np.float32),
                gt_labels_3d=np.zeros(0, dtype=np.int64),
            )
        # default LiDARInstance3DBoxes origin (0.5, 0.5, 0): bottom-center z,
        # matching how prepare_lumpi_training.py stores the boxes
        ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(
            ann_info['gt_bboxes_3d'], box_dim=7)
        return ann_info
