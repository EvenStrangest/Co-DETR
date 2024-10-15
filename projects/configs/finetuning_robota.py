# NOTE: created with this help https://chatgpt.com/share/66e7fb57-418c-8003-a7c3-95d1358c09bf


# Specify the checkpoint to load pre-trained weights
_base_ = [
    './co_deformable_detr/co_deformable_detr_r50_1x_robota.py'
]
load_from = '/checkpoints/co_deformable_detr_r50_1x_coco.pth'

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
    lr=1e-5,  # Lower learning rate as per your request
#   weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            # Freeze the backbone by setting its learning rate to zero
            'backbone': dict(lr_mult=0.0),
            'neck': dict(lr_mult=0.0),
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
    step=[1],  # Adjust steps based on total epochs
    gamma=0.1
)

# Total epochs
total_epochs = 1  # You can adjust this as needed
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# Runtime settings
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)

# Work directory to save checkpoints and logs
work_dir = '/logs/ft_co_deformable_detr_on_robota'

# Additional hooks
custom_hooks = []

