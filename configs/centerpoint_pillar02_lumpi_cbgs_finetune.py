# CBGS class-balanced refinement of the LUMPI fine-tuned CenterPoint.
#
# The baseline fine-tune (lumpi_finetune/epoch_20) barely learned the rare
# classes (motorcycle n=34, scooter) because car/pedestrian dominate the
# frames. This continues from that checkpoint with the training set wrapped
# in CBGSDataset (class-balanced group sampling, ~9x replication driven by
# the rarest classes) for a few epochs, so motorcycle/scooter frames are
# seen far more often. Scored by eval_lumpi_detection.py under the same
# protocol: epoch_20 (no CBGS) vs this (epoch_20 + CBGS refinement) is the
# comparison. Starting from the trained head keeps the initial loss sane
# and the run to ~2 h instead of a from-scratch CBGS pass (~9 h).

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
        type='CBGSDataset',
        dataset=dict(
            type='LumpiDataset',
            data_root=data_root,
            ann_file='lumpi_infos_train.pkl',
            data_prefix=dict(pts='bins'),
            pipeline=train_pipeline,
            metainfo=dict(classes=class_names),
            modality=dict(use_lidar=True, use_camera=False),
            box_type_3d='LiDAR',
            test_mode=False)))

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
                                    'lumpi_cbgs_finetune/test_results.pkl')

# CBGS replicates the set ~9x, so a few epochs already give the rare
# classes more exposure than the baseline's full 20 epochs.
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=4)

# CBGS batches are heavy with replicated rare-class objects and produce
# occasional gradient spikes (grad_norm -> hundreds) that diverge the loss;
# clip them. Also flatten the base cyclic LR (which ramps to ~1e-3, far too
# hot for a short refinement) to a constant moderate rate.
param_scheduler = [dict(type='ConstantLR', factor=1.0, begin=0, end=4,
                        by_epoch=True)]
optim_wrapper = dict(optimizer=dict(lr=1e-4),
                     clip_grad=dict(max_norm=10, norm_type=2))

# continue from the LUMPI fine-tuned model (trained 7-class head)
load_from = '/mnt/T9/parham/lidar-pilot/work_dirs/lumpi_finetune/epoch_20.pth'

default_hooks = dict(checkpoint=dict(interval=5))
