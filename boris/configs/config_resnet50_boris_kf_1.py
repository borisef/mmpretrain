work_dir = '/home/borisef/Runs/mmpretrain/resnet50_kfold_try_1'
auto_scale_lr = dict(base_batch_size=256)
NUM_CLASSES = 7
resume = True


custom_imports = dict(imports=['mmpretrain.engine.hooks.atraf.boris.visualization_extra_hook'], allow_failed_imports=False)

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
dataset_type = 'CustomDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=True, type='VisualizationExtraHook', conf_mat_params = {'save_csv': True, 'save_img': True}))

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
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    head=dict(
        in_channels=2048,
        loss=dict(loss_weight=1.0, type='MyCrossEntropyLoss'),
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

test_cfg = dict()
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root='/home/borisef/data/classification/caltech-101/small/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
       # split='val',
        type=dataset_type),
    num_workers=1,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(edge='short', scale=256, type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root='/home/borisef/data/classification/caltech-101/small/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=224, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        #split='train',
        type=dataset_type),
    num_workers=1,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=224, type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root='/home/borisef/data/classification/caltech-101/small/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        #split='val',
        type=dataset_type),
    num_workers=5,
    sampler=dict(shuffle=False, type='DefaultSampler'))

val_evaluator = [
  # dict(type='Accuracy', topk=(1, 5)),
  # dict(type='ConfusionMatrix'), #??
  # dict(type='SingleLabelMetric', items=['precision', 'recall', 'support', 'f1-score'], average = 'macro', prefix = 'macro'), # 'macro', 'micro', 'none'
  # dict(type='SingleLabelMetric', items=['precision', 'recall', 'support', 'f1-score'], average = 'micro', prefix = 'micro'), # 'macro', 'micro', 'none'
  dict(type='SingleLabelMetric', items=['precision', 'recall', 'support', 'f1-score'], average = 'macro', prefix = 'metric'), # 'macro', 'micro', 'none'
]
test_evaluator = val_evaluator

visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
)