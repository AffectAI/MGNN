_base_ = ['../_base_/default_runtime.py']

model = dict(
    type='Recognizer2DGNN',
    backbone=dict(
        type='ResNet',
        pretrained='./pretrained_models/resnet50_ft_weight.pth',
        depth=50,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN2d', requires_grad=True),  # dict(type='BN2d', requires_grad=True)
        norm_eval=False),
    neck=dict(
        type='MGNN',
        in_channels=(256, 512, 1024, 2048),
        out_channels=256,
        temporal_choose='last_frame',  # 'mean' or 'last_frame'
        conv_cfg=dict(type='Conv2d'),
        norm_cfg=dict(type='GN', num_groups=32),
        aux_head_cfg=dict(
            num_classes=1,
            dropout_ratio=0.4,
            loss_weight=0.5,
            loss_cls=dict(
                type='RegressionLoss',
                loss_type='BMC+MSE',
                loss_weight=[1, 1]))),
    cls_head=dict(
        type='MGNNHead',
        num_classes=1,
        in_channels=256,
        is_aux=False,
        dropout_ratio=0.4,
        conv_cfg=dict(type='Conv2d'),
        norm_cfg=dict(type='GN', num_groups=32),
        loss_cls=dict(
            type='RegressionLoss',
            loss_type='BMC+MSE',
            loss_weight=[1, 1])),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None)

# dataset settings
dataset_type = 'RawframeDataset'
data_root = './datasets/AVEC2014/data'
data_root_val = './datasets/AVEC2014/data'
ann_file_train = './datasets/AVEC2014/label/train_val.txt'
ann_file_val = './datasets/AVEC2014/label/test_fusion.txt'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    # dict(type='UniformSampleFrames', clip_len=9),
    dict(type='AdaptiveSampleFrames', max_clip_len=9, frame_rate=30, seed=0),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter', brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    # dict(type='UniformSampleFrames', clip_len=9, num_clips=5, test_mode=True),
    dict(type='AdaptiveSampleFrames', max_clip_len=9, frame_rate=30, num_clips=5, test_mode=True, seed=0),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    # dict(type='UniformSampleFrames', clip_len=9, num_clips=5, test_mode=True),
    dict(type='AdaptiveSampleFrames', max_clip_len=9, frame_rate=30, num_clips=5, test_mode=True, seed=0),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        filename_tmpl='{:04}.jpg',
        start_index=0,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:04}.jpg',
        start_index=0,
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:04}.jpg',
        start_index=0,
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='RegMetric')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(interval=3, max_keep_ckpts=3))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=200, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=200,
        by_epoch=True,
        milestones=[80, 160],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (32 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=256)
