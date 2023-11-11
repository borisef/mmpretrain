work_dir = '/home/borisef/Runs/mmpretrain/resnext50'
auto_scale_lr = dict(base_batch_size=256)
NUM_CLASSES = 7
resume = True
dataset_type = 'CustomDataset'
my_data_root = '/home/borisef/data/classification/caltech-101/small/'
my_batch = 4

data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=NUM_CLASSES,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
#dataset_type = 'ImageNet'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        depth=50,
        groups=32,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNeXt',
        width_per_group=4),
    head=dict(
        in_channels=2048,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=NUM_CLASSES,
        topk=(
            1,
            5,
        ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=0.1, momentum=0.9, type='SGD', weight_decay=0.0001))
param_scheduler = dict(
    by_epoch=True, gamma=0.1, milestones=[
        30,
        60,
        90,
    ], type='MultiStepLR')
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=my_batch,
    dataset=dict(
        data_root=my_data_root + 'test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        #split='test',
        type=dataset_type),
    num_workers=5,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', edge='short', scale=256, type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=my_batch,
    dataset=dict(
        data_root=my_data_root + 'train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', scale=224, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        #split='train',
        type=dataset_type),
    num_workers=5,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', scale=224, type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=my_batch,
    dataset=dict(
        data_root=my_data_root + 'test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        #split='test',
        type=dataset_type),
    num_workers=5,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
  dict(type='Accuracy', topk=(1, 5)),
  dict(type='SingleLabelMetric', items=['precision', 'recall']),
]
test_evaluator = val_evaluator

visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
)