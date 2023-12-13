_base_ = '../_base_/default_runtime.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
ood_index = 1
find_unused_parameters = True
model = dict(
    type='PatchEncoderDecoder',
    pretrained='./mmsegmentation/work_dirs/segformer_mit-b5_8x1_1024x1024_160k_cityscapes/latest.pth',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=320,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),

    patch_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    logit_loss_cfg=dict(
         type="LikelihoodLoss",
         id_index=0,
         ood_index=ood_index,
         id_weight=1.0,
         ood_weight=1.0,
         loss_weight=0.05),
    # model setting for patch generation
    patch_cfg=dict(
        noise_ratio=0.1,
        area=(0.03, 0.06),
        patch_index=ood_index,
    ),
    train_cfg=dict(),
    uncertain_cfg=dict(type="energy", probability=False),
    test_cfg=dict(mode='whole', with_ood=False, merge_weight=0.5))
dataset_type = 'CityscapesDataset'
data_root = './data/cityscape/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        #img_ratios=[0.75, 1.0, 1.25, 1.5],
        flip=False,#True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        img_suffix='_leftImg8bit.png',
        seg_map_suffix='_gtFine_labelTrainIds.png',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        img_suffix='_leftImg8bit.png',
        seg_map_suffix='_gtFine_labelTrainIds.png',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        img_suffix='_leftImg8bit.png',
        seg_map_suffix='_gtFine_labelTrainIds.png',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    #_delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))#dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)


optimizer_config = dict()
# learning policy
lr_config = dict(
    #_delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)#dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
work_dir = './results/day928/segformerb4_nofix10_003006_weight005_adv_weightnew6k/'

#load_from = './segformer_mit-b4_8x1_1024x1024_160k_cityscapes_20211207_080709-07f6c333.pth'
load_from = './mmsegmentation/work_dirs/segformer_mit-b5_8x1_1024x1024_160k_cityscapes/latest.pth'
    # model training and testing settings




