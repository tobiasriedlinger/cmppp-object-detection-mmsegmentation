_base_ = './fcn_hr18_4xb2-160k_cityscapes_inst-512x1024.py'
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384]),
        num_classes=1,
        loss_decode=dict(
            type='PoissonNLLLoss', use_sigmoid=True, loss_weight=1.0
        )
    ))
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
train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=160000)