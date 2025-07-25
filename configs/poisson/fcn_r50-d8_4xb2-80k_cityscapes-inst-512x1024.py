_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/cityscapes_instances.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
