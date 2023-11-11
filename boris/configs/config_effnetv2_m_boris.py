work_dir = '/home/borisef/Runs/mmpretrain/effnetv2_m'
NUM_CLASSES = 7
resume = True
my_dataset_type = 'CustomDataset'
my_data_root = '/home/borisef/data/classification/caltech-101/small/'
my_batch = 2

auto_scale_lr = dict(base_batch_size=256)
data_preprocessor = dict(
    mean=[
        127.5,
        127.5,
        127.5,
    ],
    num_classes=NUM_CLASSES,
    std=[
        127.5,
        127.5,
        127.5,
    ],
    to_rgb=True)
dataset_type = 'ImageNet'
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
    backbone=dict(arch='m', type='EfficientNetV2'),
    head=dict(
        in_channels=1280,
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
#resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=my_batch,
    dataset=dict(
        data_root=my_data_root + 'test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(crop_padding=0, crop_size=480, type='EfficientNetCenterCrop'),
            dict(type='PackInputs'),
        ],
        #split='val',
        type=my_dataset_type),
    num_workers=5,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(crop_padding=0, crop_size=480, type='EfficientNetCenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=my_batch,
    dataset=dict(
        data_root=my_data_root + 'train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(crop_padding=0, scale=384, type='EfficientNetRandomCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        #split='train',
        type=my_dataset_type),
    num_workers=5,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(crop_padding=0, scale=384, type='EfficientNetRandomCrop'),
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
            dict(crop_padding=0, crop_size=480, type='EfficientNetCenterCrop'),
            dict(type='PackInputs'),
        ],
        #split='val',
        type=my_dataset_type),
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
