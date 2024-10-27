# dataset settings
dataset_type = 'RobotaDataset'
data_root = 'data/Robot-controlled A/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)  # TODO: what should these be?
# background_color = ()

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),  # TODO: consider specifying direction=['horizontal', 'vertical']
    # TODO: consider adding a Rotate augmentation; specify img_fill_val=background_color
    # TODO: consider adding a Translate augmentation; specify img_fill_val=background_color
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='AddAugmentationID'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'], meta_keys=('ori_filename', 'filename', 'flip', 'flip_direction', 'rotation_angle')),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'as_yolo_as_coco_ann_train.json',
        img_prefix=data_root + 'as_yolo/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'as_yolo_as_coco_ann_val.json',
        img_prefix=data_root + 'as_yolo/images/',
        pipeline=val_pipeline,
        test_mode=False),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'as_yolo_as_coco_ann_test.json',
        img_prefix=data_root + 'as_yolo/images/',
        pipeline=test_pipeline,
        test_mode=False)
)
evaluation = dict(interval=1, metric='bbox')

custom_imports = dict(
    imports=[#'projects.custom_pipelines',
             'projects.custom_augmentations'],
    allow_failed_imports=False)
