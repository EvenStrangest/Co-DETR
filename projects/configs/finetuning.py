_base_ = [
    './configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py'
]


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

# Runtime settings
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)

# Work directory to save checkpoints and logs
work_dir = '/checkpoints/ft_co_deformable_detr_on_coco'

# Additional hooks (if any)
custom_hooks = []

# FP16 training (optional)
# fp16 = dict(loss_scale=512.)
