_base_ = '../_base_/default_runtime.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
ood_index = 1
model = dict(
    type='PatchRenderCascadedEncoderDecoder',
    num_stages=2,
    pretrained='./ocrnet_hr48_512x1024_40k_cityscapes_20200601_033336-55b32491.pth',
    backbone=dict(
        type='HRNet',
        norm_cfg=norm_cfg,
        norm_eval=True,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384)))),            
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    patch_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=3,
        channels=256,
        num_convs=3,
        concat_input=False,
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
        img_ratios=[0.75, 1.0, 1.25, 1.5],
        flip=True,
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
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
work_dir = './results/day927/ocrhrnet48_4c_fix10_copypaste003006_gradientnew/'

load_from = './ocrnet_hr48_512x1024_40k_cityscapes_20200601_033336-55b32491.pth'

    # model training and testing settings




