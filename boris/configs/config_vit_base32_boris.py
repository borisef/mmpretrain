work_dir = '/home/borisef/Runs/mmpretrain/vit_base32'
NUM_CLASSES = 7
resume = True
my_dataset_type = 'CustomDataset'
my_data_root = '/home/borisef/data/classification/caltech-101/small/'
my_batch = 8

auto_scale_lr = dict(base_batch_size=4096)
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
dataset_type = my_dataset_type
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
        arch='b',
        drop_rate=0.1,
        img_size=384,
        init_cfg=[
            dict(
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear',
                type='Kaiming'),
        ],
        patch_size=32,
        type='VisionTransformer'),
    head=dict(
        in_channels=768,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=NUM_CLASSES,
        topk=(
            1,
            5,
        ),
        type='VisionTransformerClsHead'),
    neck=None,
    type='ImageClassifier')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0),
    optimizer=dict(lr=0.003, type='AdamW', weight_decay=0.3),
    paramwise_cfg=dict(
        custom_keys=dict({
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        })))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=30,
        start_factor=0.0001,
        type='LinearLR'),
    dict(
        T_max=270, begin=30, by_epoch=True, end=300, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=my_batch,
    dataset=dict(
        data_root=my_data_root + 'test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', edge='short', scale=384, type='ResizeEdge'),
            dict(crop_size=384, type='CenterCrop'),
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
    dict(backend='pillow', edge='short', scale=384, type='ResizeEdge'),
    dict(crop_size=384, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
train_dataloader = dict(
    batch_size=my_batch,
    dataset=dict(
        data_root=my_data_root + 'train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', scale=384, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        #split='train',
        type=my_dataset_type),
    num_workers=5,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', scale=384, type='RandomResizedCrop'),
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
            dict(backend='pillow', edge='short', scale=384, type='ResizeEdge'),
            dict(crop_size=384, type='CenterCrop'),
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
