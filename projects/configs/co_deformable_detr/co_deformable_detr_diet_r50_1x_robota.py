from os import environ

_base_ = ['../_base_/default_runtime.py']
if environ.get('CHOICE_DATASET') == 'LabA':
    _base_.append('../_base_/datasets/laba_detection.py')
elif environ.get('CHOICE_DATASET') == 'LabC':
    _base_.append('../_base_/datasets/laba_detection.py')
    test_data = dict(
        ann_file='data/Lab C/crop_one_as_coco_annotations.json',
        img_prefix='data/Lab C/crop_one/')
    data_root = 'data/Lab C/'
elif environ.get('CHOICE_DATASET') == 'LabD':
    _base_.append('../_base_/datasets/laba_detection.py')
    test_data = dict(
        ann_file='data/Lab D/source_over_usb_as_coco_annotations.json',
        img_prefix='data/Lab D/source_over_usb/')
    data_root = 'data/Lab D/'
elif environ.get('CHOICE_DATASET') == 'RobotA1ofeach':
    _base_.append('../_base_/datasets/robota_detection.py')
    # from .._base_.datasets.robota_detection import data_root as robota_data_root
    robota_data_root = 'data/Robot-controlled A/'
    test_data=dict(
        ann_file=robota_data_root + 'one_of_each_class_as_coco_annotations.json',
        img_prefix=robota_data_root + 'one_of_each_class/')
elif environ.get('CHOICE_DATASET') == 'RobotA':
    _base_.append('../_base_/datasets/robota_detection.py')

# model settings
num_dec_layer = 6
lambda_2 = 2.0

model = dict(
    type='CoDETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),  # TODO: if learning normalization parameters, then change requires_grad=True
        norm_eval=True,  # TODO: maybe better to learn the normalization parameters?
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    rpn_head=None,
    query_head=dict(
        type='CoDeformDETRHead',
        num_query=300,  # TODO: consider increasing this
        num_classes=80,  # TODO: does this need to be corrected?
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=True,
        mixed_selection=True,
        transformer=dict(
            type='CoDeformableDetrTransformer',
            num_co_heads=2,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256, dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='CoDeformableDetrTransformerDecoder',
                num_layers=num_dec_layer,
                return_intermediate=True,
                look_forward_twice=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            dropout=0.0)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),  # TODO: consider reducing weight
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    roi_head=[],
    bbox_head=[],
    # model training and testing settings
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),  # TODO: consider reducing weight
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
        ],
    test_cfg=[
        dict(max_per_img=100),
    ])

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# # test_pipeline, NOTE the Pad's size_divisor is different from the default
# # setting (size_divisor=32). While there is little effect on the performance
# # whether we use the default setting or use size_divisor=1.
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=1),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img'])
#         ])
# ]

data = dict(
    samples_per_gpu=3,  # TODO: consider increasing this
    workers_per_gpu=3,  # TODO: consider increasing this
    train=dict(filter_empty_gt=False, pipeline=train_pipeline),
    # val=dict(pipeline=test_pipeline),
    # test=dict(pipeline=test_pipeline)
)
if 'test_data' in locals():
    data['test'] = test_data
    # TODO: this way of doing things is entirely barbaric!
    #  better to introduce new a new key in the data dict,
    #  and switch keys within clearml_and_test.py
    #  in response to command-line arguments!
# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
