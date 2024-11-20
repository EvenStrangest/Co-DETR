# NOTE: created with this help https://chatgpt.com/share/66e7fb57-418c-8003-a7c3-95d1358c09bf


# Specify the checkpoint to load pre-trained weights
_base_ = [
    './co_deformable_detr/co_deformable_detr_r50_1x_robota.py'
]
# load_from = 'https://files.clear.ml/Co-DETR/FinetuneRobotA%20lr%201e-5%20neck%20lr%20factor%200.25.b10ab572e11444dc8b47cbddb867d420/models/epoch_1.pth'
# load_from = 'https://files.clear.ml/Co-DETR/FinetuneRobotA.bdeff4043035447c885ca769ef7673a4/models/epoch_2.pth'
# load_from = 'https://files.clear.ml/Co-DETR/FinetuneRobotA_with_Round2_backbone_lrm0.1.131b8e106c7143028650f55a8b786848/models/epoch_2.pth'
# load_from = 'https://files.clear.ml/Co-DETR/FtRA_with_Round4_only.86d5cf5726a244a4a4b77019772d9d18/models/epoch_3.pth'
# load_from = 'https://files.clear.ml/Co-DETR/FtRA_with_Round4and5.1844c666b4394c668cf6b41211754744/models/epoch_3.pth'
load_from = 'https://files.clear.ml/Co-DETR/FtRA_with_Round4and5more.45bcbfcf7d0c4fd68ab8c7a8ce838b41/models/epoch_4.pth'
# resume_from = ''

model = dict(
    type='CoDETR',
    backbone=dict(
        num_stages=4,
        frozen_stages=4
    )
)

# Optimizer settings
optimizer = dict(
    type='AdamW',
    lr=1e-5,  # A low learning rate
#   weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            # Freeze the backbone by setting its learning rate to zero
            'backbone': dict(lr_mult=0.0),
            'neck': dict(lr_mult=0.5),
            # Optionally adjust other layers
        }
    )
)
# optimizer_config = dict(
#     grad_clip=dict(max_norm=0.1, norm_type=2)
# )

# Learning rate schedule
lr_config = dict(
    policy='step',
    step=[1, 2],  # Adjust steps based on total epochs
    gamma=0.9
)

# Total epochs
total_epochs = 4
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# Runtime settings
checkpoint_config = dict(interval=1000, by_epoch=False)
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)
evaluation = dict(
    interval=100, by_epoch=False,
    metric='bbox',
    save_best='bbox_mAP'
)

# Work directory to save checkpoints and logs
work_dir = '/logs/ft_co_deformable_detr_on_robota'

# Additional hooks
custom_hooks = [
    dict(type='ClearMLCheckpointHook')
]


