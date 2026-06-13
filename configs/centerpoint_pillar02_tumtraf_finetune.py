# Fine-tune the LUMPI-fine-tuned CenterPoint-pillar on TUMTraf Intersection.
#
# Starts from the LUMPI epoch_20 checkpoint (Hanover roadside) and adapts
# on TUMTraf (Munich roadside) — measuring how much of the cross-site gap
# closes with in-domain data on top of roadside pretraining. Reuses the
# LumpiDataset plugin (TUMTraf infos are written in the same format by
# scripts/prepare_tumtraf_training.py) and the same 7-class head, so the
# fine-tuned model is scored by eval_tumtraf_detection.py under the exact
# protocol used for the zero-shot numbers.

_base_ = ['/mnt/T9/parham/lidar-pilot/mmdetection3d/configs/centerpoint/'
          'centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py']

custom_imports = dict(
    imports=['lidar_pilot.mmdet3d_plugins.lumpi_dataset'],
    allow_failed_imports=False)

class_names = ['car', 'truck', 'bus', 'pedestrian', 'bicycle',
               'motorcycle', 'scooter']
data_root = '/mnt/T9/parham/lidar-pilot/data/tumtraf/kit/'
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
        ann_file='tumtraf_infos_train.pkl',
        data_prefix=dict(pts='bins'),
        pipeline=train_pipeline,
        metainfo=dict(classes=class_names),
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR',
        test_mode=False))

val_dataloader = None
val_cfg = None
val_evaluator = None

test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LumpiDataset',
        data_root=data_root,
        ann_file='tumtraf_infos_val.pkl',
        data_prefix=dict(pts='bins'),
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names),
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR',
        test_mode=True))
test_cfg = dict()
test_evaluator = dict(type='DumpResults',
                      out_file_path='/mnt/T9/parham/lidar-pilot/work_dirs/'
                                    'tumtraf_finetune/test_results.pkl')

train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=20)

optim_wrapper = dict(optimizer=dict(lr=2.5e-5))

# continue from the Hanover-fine-tuned model (same 7-class head -> full load)
load_from = '/mnt/T9/parham/lidar-pilot/work_dirs/lumpi_finetune/epoch_20.pth'

default_hooks = dict(checkpoint=dict(interval=5))
