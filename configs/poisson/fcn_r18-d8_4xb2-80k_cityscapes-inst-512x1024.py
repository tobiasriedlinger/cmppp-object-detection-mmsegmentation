_base_ = './fcn_r50-d8_4xb2-80k_cityscapes-inst-512x1024.py'
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        act_cfg=dict(type='LeakyReLU',negative_slope=0.1),
        in_channels=512,
        channels=128,
        num_classes=1,
        loss_decode=dict(
            type='PoissonNLLLoss', use_sigmoid=False, loss_weight=1.0
        )
    ),
    auxiliary_head=dict(
        act_cfg=dict(type='LeakyReLU',negative_slope=0.1),
        in_channels=256, 
        channels=64,
        num_classes=1,
        loss_decode=dict(
            type='PoissonNLLLoss', use_sigmoid=False, loss_weight=0.4
        )
        )
    )

# optimizer
optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=30000,
        by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=150000, val_interval=130000)