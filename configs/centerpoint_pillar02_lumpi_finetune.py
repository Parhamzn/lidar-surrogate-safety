# Fine-tune nuScenes-pretrained CenterPoint-pillar on LUMPI roadside frames.
#
# Starts from the pillar02 nuScenes checkpoint (backbone/neck transfer; the
# head is re-shaped to LUMPI's 7 classes, so mismatched head weights are
# skipped at load). Trains on single fused 5-LiDAR frames (no sweep
# aggregation; the 5th point feature stays 0 for input compatibility).
# Validation during training is disabled: both checkpoints are scored by
# scripts/eval_lumpi_detection.py on the held-out time slice instead, so
# pretrained and fine-tuned models share one protocol.

_base_ = ['/mnt/T9/parham/lidar-pilot/mmdetection3d/configs/centerpoint/'
          'centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py']

custom_imports = dict(
    imports=['lidar_pilot.mmdet3d_plugins.lumpi_dataset'],
    allow_failed_imports=False)

class_names = ['car', 'truck', 'bus', 'pedestrian', 'bicycle',
               'motorcycle', 'scooter']
data_root = '/mnt/T9/parham/lidar-pilot/data/lumpi/m5_kit/'
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

model = dict(
    pts_bbox_head=dict(
        tasks=[dict(num_class=7, class_names=class_names)]))

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.3925, 0.3925],
         scale_ratio_range=[0.95, 1.05],
         translation_std=[0.2, 0.2, 0.2]),
    dict(type='RandomFlip3D',
         flip_ratio_bev_horizontal=0.5,
         flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs',
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points']),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='LumpiDataset',
        data_root=data_root,
        ann_file='lumpi_infos_train.pkl',
        data_prefix=dict(pts='bins'),
        pipeline=train_pipeline,
        metainfo=dict(classes=class_names),
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR',
        test_mode=False))

# no validation loop during training (see header note)
val_dataloader = None
val_cfg = None
val_evaluator = None

# the test triple exists so inference tooling can read a pipeline from this
# config; it is not used as a benchmark
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LumpiDataset',
        data_root=data_root,
        ann_file='lumpi_infos_val.pkl',
        data_prefix=dict(pts='bins'),
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names),
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR',
        test_mode=True))
test_cfg = dict()
test_evaluator = dict(type='DumpResults',
                      out_file_path='/mnt/T9/parham/lidar-pilot/work_dirs/'
                                    'lumpi_finetune/test_results.pkl')

train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=20)

# fine-tuning: an order of magnitude below the from-scratch LR
# (base cyclic schedule scales this to a peak of ~10x)
optim_wrapper = dict(optimizer=dict(lr=2.5e-5))

load_from = ('/mnt/T9/parham/lidar-pilot/checkpoints/'
             'centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_'
             '20220811_031844-191a3822.pth')

default_hooks = dict(checkpoint=dict(interval=5))
