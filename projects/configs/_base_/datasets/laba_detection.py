# dataset settings
dataset_type = 'RobotaDataset'
data_root = 'data/Lab A/'

# TODO: consider importing img_norm_cfg (and other commonalities) from robota_detection.py

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)  # TODO: what should these be?
# background_color = ()

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),  # TODO: consider specifying direction=['horizontal', 'vertical']
#     # TODO: consider adding a Rotate augmentation; specify img_fill_val=background_color
#     # TODO: consider adding a Translate augmentation; specify img_fill_val=background_color
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        # scale_factor = [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        flip=False,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    # train=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'as_yolo_as_coco_ann_train.json',
    #     img_prefix=data_root + 'as_yolo/images/',
    #     pipeline=train_pipeline),
    # val=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'as_yolo_as_coco_ann_val.json',
    #     img_prefix=data_root + 'as_yolo/images/',
    #     pipeline=test_pipeline),
    test=dict(type='ConcatDataset',
              datasets=[
                  # dict(
                  #     type=dataset_type,
                  #     ann_file=data_root + 'raw_images.json',
                  #     img_prefix=data_root + 'raw_images/',
                  #     pipeline=test_pipeline,
                  #     test_mode=True),
                  dict(
                      type=dataset_type,
                      ann_file=data_root + 'crop_manual_choice_one_two_as_coco_annotations.json',
                      img_prefix=data_root + 'crop_manual_choice_one_two/',
                      pipeline=test_pipeline,
                      test_mode=True),
              ],
              separate_eval=True)
)
evaluation = dict(interval=1, metric='bbox')
