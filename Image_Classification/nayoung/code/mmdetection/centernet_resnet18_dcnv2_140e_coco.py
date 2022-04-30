###############################
# Dataset
###############################

dataset_type = 'CocoDataset'
data_root = 'data/coco/'

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_norm_cfg = dict(
    mean=[122.6902, 116.4859, 109.2194], std=[60.9837, 59.9108, 61.8820], to_rgb=True)


# train_pipeline = [
#     dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PhotoMetricDistortion',
#         brightness_delta=32,
#         contrast_range=(0.5, 1.5),
#         saturation_range=(0.5, 1.5),
#         hue_delta=18),
#     dict(
#         type='RandomCenterCropPad',
#         crop_size=(512, 512),
#         ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
#         mean=[0, 0, 0],
#         std=[1, 1, 1],
#         to_rgb=True,
#         test_pad_mode=None),
#     dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(
#         type='Normalize',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         to_rgb=True),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile', to_float32=True),
#     dict(
#         type='MultiScaleFlipAug',
#         scale_factor=1.0,
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(
#                 type='RandomCenterCropPad',
#                 ratios=None,
#                 border=None,
#                 mean=[0, 0, 0],
#                 std=[1, 1, 1],
#                 to_rgb=True,
#                 test_mode=True,
#                 test_pad_mode=['logical_or', 31],
#                 test_pad_add_pix=1),
#             dict(type='RandomFlip'),
#             dict(
#                 type='Normalize',
#                 mean=[123.675, 116.28, 103.53],
#                 std=[58.395, 57.12, 57.375],
#                 to_rgb=True),
#             dict(type='DefaultFormatBundle'),
#             dict(
#                 type='Collect',
#                 meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
#                            'scale_factor', 'flip', 'flip_direction',
#                            'img_norm_cfg', 'border'),
#                 keys=['img'])
#         ])
# ]

image_size = 1024
size_min, size_max = map(int, (image_size * 0.5, image_size * 1.5))


multi_scale = [(x, x) for x in range(size_min, size_max + 1, 32)]
multi_scale_test = [(x, x) for x in range(size_min, size_max + 1, 256)]
multi_scale_val = [(x, x) for x in range(size_min, size_max + 1, 512)]
multi_scale_light = [(512, 512), (768, 768), (1024, 1024)]

albu_transform = [
    dict(type='VerticalFlip', p=0.1),
    dict(type='HorizontalFlip', p=0.3),
    dict(type='OneOf', transforms=[
        dict(type='GaussNoise', p=1.0),
        dict(type='GaussianBlur', p=1.0),
        dict(type='Blur', p=1.0)
    ], p=0.1),
    dict(type='OneOf', 
         transforms=[
             dict(type='ShiftScaleRotate', p=1.0),
             dict(type='RandomRotate90', p=1.0),
         ], p=0.1),

    # color 관련 transform 
    dict(type='ColorJitter', 
        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.1)
]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         img_scale=multi_scale,
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='Albu',
         transforms=albu_transform,
         bbox_params=dict(
             type='BboxParams',
             format='pascal_voc',
             label_fields=['gt_labels'],
             min_visibility=0.0,
             filter_lost_elements=True
         ),
         keymap={
             'img': 'image',
             'gt_bboxes': 'bboxes'
         },
         update_pad_shape=False,
         skip_img_without_anno=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=multi_scale_light,
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=multi_scale_light, multiscale_mode='value', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type='CocoDataset',
            classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                     'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                     'Clothing'),
            ann_file='/opt/ml/detection/baseline/Swin-Transformer-Object-Detection/custom_configs/STRATIFIEDKFOLD/cv_train_3_add.json',
            img_prefix='/opt/ml/detection/dataset/',
            pipeline=[
                dict(
                    type='LoadImageFromFile',
                    to_float32=True,
                    color_type='color'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='PhotoMetricDistortion',
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                dict(
                    type='RandomCenterCropPad',
                    crop_size=(512, 512),
                    ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
                    mean=[0, 0, 0],
                    std=[1, 1, 1],
                    to_rgb=True,
                    test_pad_mode=None),
                dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ])),

    val=dict(
        type='CocoDataset',
        ann_file='/opt/ml/detection/baseline/Swin-Transformer-Object-Detection/custom_configs/STRATIFIEDKFOLD/cv_val_3.json',
        img_prefix='/opt/ml/detection/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='RandomCenterCropPad',
                        ratios=None,
                        border=None,
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        to_rgb=True,
                        test_mode=True,
                        test_pad_mode=['logical_or', 31],
                        test_pad_add_pix=1),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect',
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'flip_direction', 'img_norm_cfg', 'border'),
                        keys=['img'])
                ])
        ],
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing')),

    test=dict(
        type='CocoDataset',
        ann_file='/opt/ml/detection/dataset/test.json',
        img_prefix='/opt/ml/detection/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='RandomCenterCropPad',
                        ratios=None,
                        border=None,
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        to_rgb=True,
                        test_mode=True,
                        test_pad_mode=['logical_or', 31],
                        test_pad_add_pix=1),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect',
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'flip_direction', 'img_norm_cfg', 'border'),
                        keys=['img'])
                ])
        ],
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing')))

###############################
# Scheduler
###############################

evaluation = dict(interval=1, metric='bbox')


# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(
    type='AdamW',
    lr=0.0001 / 2,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0)))
)


optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=0.001,
#     step=[18, 24])

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=0.1,
    min_lr_ratio=5e-6
)

runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(max_keep_ckpts=2, interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), 
            dict(type='WandbLoggerHook',interval=50,
            init_kwargs=dict(
            project='level2-object-detection',
            entity = 'dudskrla',
            name = f"0407_CenterNet_ResNet18" # 날짜 변경 
            ),
            ),])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'


###############################
# Model
###############################


model = dict(
    type='CenterNet',
    backbone=dict(
        type='ResNet',
        depth=18,
        norm_eval=False,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='CTResNetNeck',
        in_channel=512,
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=True),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=10,
        in_channel=64,
        feat_channel=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100))
work_dir = f'./work_dirs/CenterNet_ResNet18_L1loss_coco'
auto_resume = True
gpu_ids = [0]
