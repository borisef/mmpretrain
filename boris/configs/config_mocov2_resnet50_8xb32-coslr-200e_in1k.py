work_dir = '/home/borisef/Runs/mmpretrain/mocov2_load'
resume = False
dataset_type = 'CustomDataset'
my_data_root = '/home/borisef/data/classification/caltech-101/small/self'
my_batch = 8
load_from = "/home/borisef/models/mmpretrain/mocov2_resnet50_8xb32-coslr-200e_in1k_20220825-b6d23c86.pth"


#data_root = 'data/imagenet/'
#dataset_type = 'ImageNet'


auto_scale_lr = dict(base_batch_size=256, enable = True)
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True,
    type='SelfSupDataPreprocessor')
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, type='CheckpointHook'),
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

log_level = 'INFO'
model = dict(
    backbone=dict(
        depth=50,
        norm_cfg=dict(type='BN'),
        type='ResNet',
        zero_init_residual=False),
    feat_dim=128,
    head=dict(
        loss=dict(type='CrossEntropyLoss'),
        temperature=0.2,
        type='ContrastiveHead'),
    momentum=0.001,
    neck=dict(
        hid_channels=2048,
        in_channels=2048,
        out_channels=128,
        type='MoCoV2Neck',
        with_avg_pool=True),
    queue_len=65536,
    type='MoCo')
optim_wrapper = dict(
    optimizer=dict(lr=0.03, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(T_max=200, begin=0, by_epoch=True, end=200, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=None)
train_cfg = dict(max_epochs=200, type='EpochBasedTrainLoop')
train_dataloader = dict(
    batch_size=my_batch,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root=my_data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                num_views=2,
                transforms=[
                    [
                        dict(
                            backend='pillow',
                            crop_ratio_range=(
                                0.2,
                                1.0,
                            ),
                            scale=224,
                            type='RandomResizedCrop'),
                        dict(
                            prob=0.8,
                            transforms=[
                                dict(
                                    brightness=0.4,
                                    contrast=0.4,
                                    hue=0.1,
                                    saturation=0.4,
                                    type='ColorJitter'),
                            ],
                            type='RandomApply'),
                        dict(
                            channel_weights=(
                                0.114,
                                0.587,
                                0.2989,
                            ),
                            keep_channels=True,
                            prob=0.2,
                            type='RandomGrayscale'),
                        dict(
                            magnitude_range=(
                                0.1,
                                2.0,
                            ),
                            magnitude_std='inf',
                            prob=0.5,
                            type='GaussianBlur'),
                        dict(prob=0.5, type='RandomFlip'),
                    ],
                ],
                type='MultiView'),
            dict(type='PackInputs'),
        ],
        #split='train', #?
        type='CustomDataset'),
    drop_last=True,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        num_views=2,
        transforms=[
            [
                dict(
                    backend='pillow',
                    crop_ratio_range=(
                        0.2,
                        1.0,
                    ),
                    scale=224,
                    type='RandomResizedCrop'),
                dict(
                    prob=0.8,
                    transforms=[
                        dict(
                            brightness=0.4,
                            contrast=0.4,
                            hue=0.1,
                            saturation=0.4,
                            type='ColorJitter'),
                    ],
                    type='RandomApply'),
                dict(
                    channel_weights=(
                        0.114,
                        0.587,
                        0.2989,
                    ),
                    keep_channels=True,
                    prob=0.2,
                    type='RandomGrayscale'),
                dict(
                    magnitude_range=(
                        0.1,
                        2.0,
                    ),
                    magnitude_std='inf',
                    prob=0.5,
                    type='GaussianBlur'),
                dict(prob=0.5, type='RandomFlip'),
            ],
        ],
        type='MultiView'),
    dict(type='PackInputs'),
]
view_pipeline = [
    dict(
        backend='pillow',
        crop_ratio_range=(
            0.2,
            1.0,
        ),
        scale=224,
        type='RandomResizedCrop'),
    dict(
        prob=0.8,
        transforms=[
            dict(
                brightness=0.4,
                contrast=0.4,
                hue=0.1,
                saturation=0.4,
                type='ColorJitter'),
        ],
        type='RandomApply'),
    dict(
        channel_weights=(
            0.114,
            0.587,
            0.2989,
        ),
        keep_channels=True,
        prob=0.2,
        type='RandomGrayscale'),
    dict(
        magnitude_range=(
            0.1,
            2.0,
        ),
        magnitude_std='inf',
        prob=0.5,
        type='GaussianBlur'),
    dict(prob=0.5, type='RandomFlip'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')
    ])

