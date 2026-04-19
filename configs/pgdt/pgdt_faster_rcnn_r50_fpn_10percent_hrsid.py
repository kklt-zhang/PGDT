_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/semi_hrsid_detection.py'  # Use the cleaned dataset file name
]

detector = _base_.model

# Rewrite AnchorGenerator specifically for SAR ship detection
detector.rpn_head.anchor_generator = dict(
    type='AnchorGenerator',
    scales=[8],
    ratios=[0.2, 0.5, 1.0, 2.0, 5.0],  # Added 0.2 and 5.0 for slender ships
    strides=[4, 8, 16, 32, 64]
)
detector.roi_head.bbox_head.num_classes = 1

detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    bgr_to_rgb=False,
    pad_size_divisor=32)

detector.backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=False),
    norm_eval=True,
    style='caffe',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='open-mmlab://detectron2/resnet50_caffe'))

model = dict(
    _delete_=True,
    type='PGDT',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=2.0,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_thr=0.9,
        cls_pseudo_thr=0.9,  # Matched with tau_sem = 0.9 in the paper
        reg_pseudo_thr=0.05,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset

labeled_dataset.ann_file = 'annotations/instances_train_10percent.json'
unlabeled_dataset.ann_file = 'annotations/instances_unlabeled_90percent.json'
unlabeled_dataset.data_prefix = dict(img='images/')

train_dataloader = dict(
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# Training schedule for HRSID
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=90000, val_interval=728)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# Learning rate policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=90000, by_epoch=False, milestones=[70000, 80000], gamma=0.1)
]

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=10000,
        max_keep_ckpts=2,
        save_best='teacher/coco/bbox_mAP',
        rule='greater'
    )
)

custom_hooks = [dict(type='MeanTeacherHook')]