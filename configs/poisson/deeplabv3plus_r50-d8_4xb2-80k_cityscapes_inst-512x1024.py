_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes_instances.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        act_cfg=dict(type='LeakyReLU',negative_slope=0.1),
        num_classes=1,
        loss_decode=dict(
            type='PoissonNLLLoss', use_sigmoid=True, loss_weight=1.0
        )
    ),
    auxiliary_head=dict(
        act_cfg=dict(type='LeakyReLU',negative_slope=0.1),
        num_classes=1,
        loss_decode=dict(
            type='PoissonNLLLoss', use_sigmoid=True, loss_weight=0.4
        )
    )
)
# optimizer
optimizer = dict(type='SGD', lr=2e-4, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-5,
        power=0.9,
        begin=0,
        end=120000,
        by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=240000, val_interval=240000)
