work_dir = '/home/borisef/Runs/mmpretrain/levit256'
NUM_CLASSES = 7
resume = True
my_dataset_type = 'CustomDataset'
my_data_root = '/home/borisef/data/classification/caltech-101/small/'
my_batch = 4

auto_scale_lr = dict(base_batch_size=2048)

bgr_mean = [
    103.53,
    116.28,
    123.675,
]
bgr_std = [
    57.375,
    57.12,
    58.395,
]
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=1000,
    std=[
        58.395,
        57.12,
        57.375,
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
        arch='256',
        attn_ratio=2,
        drop_path_rate=0,
        img_size=224,
        mlp_ratio=2,
        out_indices=(2, ),
        patch_size=16,
        type='LeViT'),
    head=dict(
        distillation=False,
        in_channels=512,
        loss=dict(
            label_smooth_val=0.1, loss_weight=1.0, type='LabelSmoothLoss'),
        num_classes=1000,
        topk=(
            1,
            5,
        ),
        type='LeViTClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    train_cfg=dict(augments=[
        dict(alpha=0.8, type='Mixup'),
        dict(alpha=1.0, type='CutMix'),
    ]),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.002,
        type='AdamW',
        weight_decay=0.025),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys=dict({'.attention_biases': dict(decay_mult=0.0)}),
        norm_decay_mult=0.0))
param_scheduler = [
    dict(
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=0.0005,
        type='LinearLR'),
    dict(begin=5, by_epoch=True, eta_min=1e-05, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=None)

test_cfg = dict()
test_dataloader = dict(
    batch_size=my_batch,
    dataset=dict(
        data_root=my_data_root + 'test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=256,
                type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
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
    dict(
        backend='pillow',
        edge='short',
        interpolation='bicubic',
        scale=256,
        type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=1000)
train_dataloader = dict(
    batch_size=my_batch,
    dataset=dict(
        data_root=my_data_root + 'train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=224,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(
                hparams=dict(
                    interpolation='bicubic', pad_val=[
                        104,
                        116,
                        124,
                    ]),
                magnitude_level=9,
                magnitude_std=0.5,
                num_policies=2,
                policies='timm_increasing',
                total_level=10,
                type='RandAugment'),
            dict(
                erase_prob=0.25,
                fill_color=[
                    103.53,
                    116.28,
                    123.675,
                ],
                fill_std=[
                    57.375,
                    57.12,
                    58.395,
                ],
                max_area_ratio=0.3333333333333333,
                min_area_ratio=0.02,
                mode='rand',
                type='RandomErasing'),
            dict(type='PackInputs'),
        ],
        #split='train',
        type=my_dataset_type),
    num_workers=5,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=224,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(
        hparams=dict(interpolation='bicubic', pad_val=[
            104,
            116,
            124,
        ]),
        magnitude_level=9,
        magnitude_std=0.5,
        num_policies=2,
        policies='timm_increasing',
        total_level=10,
        type='RandAugment'),
    dict(
        erase_prob=0.25,
        fill_color=[
            103.53,
            116.28,
            123.675,
        ],
        fill_std=[
            57.375,
            57.12,
            58.395,
        ],
        max_area_ratio=0.3333333333333333,
        min_area_ratio=0.02,
        mode='rand',
        type='RandomErasing'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=my_batch,
    dataset=dict(
        data_root=my_data_root + 'test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=256,
                type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
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
