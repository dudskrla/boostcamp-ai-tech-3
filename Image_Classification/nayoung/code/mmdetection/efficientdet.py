_base_ = [
    "/opt/ml/detection/baseline/Swin-Transformer-Object-Detection/custom_configs/__base__/swin_runtime.py",
]

model_name = 'RetinaNet + EfficientNet'
model_loss = 'SmoothL1loss'
model_input = 224

# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='EfficientNet',
        model_name='tf_efficientnet_b4'),
    neck=dict(
        type='BIFPN',
        in_channels=[56, 112, 160, 272, 448],
        out_channels=224,
        start_level=0,
        stack=6,
        add_extra_convs=True,
        num_outs=5,
        norm_cfg=dict(type='BN', requires_grad=False),
        activation='relu'),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=81,
        in_channels=224,#256->224
        stacked_convs=4,
        feat_channels=224,#256->224
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.5, #2->1.5
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))

# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
#####################################
# Dataset
#####################################

data_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
anno_root = '/opt/ml/detection/baseline/efficientdet-pytorch/custom_config/stratified_kfold_eff_oversampled/'

image_size = 1024
size_min, size_max = map(int, (image_size * 0.5, image_size * 1.5))

multi_scale = [(x, x) for x in range(size_min, size_max + 1, 32)]
multi_scale_test = [(x, x) for x in range(size_min, size_max + 1, 256)]

# multi_scale_val # 512
# multi_scale_test # 256
# mAP 점수는 올라가나, inference 시간도 늘어남 

multi_scale_light = [(512, 512), (768, 768), (1024, 1024)]

img_norm_cfg = dict(
    mean=[122.6902, 116.4859, 109.2194], std=[60.9837, 59.9108, 61.8820], to_rgb=True)

classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
           'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')

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
    samples_per_gpu=4,  # batch size 변경
    workers_per_gpu=4,
    train=dict(
        type=data_type,
        classes=classes,
        ann_file=anno_root + 'cv_train_1.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=data_type,
        classes=classes,
        ann_file=anno_root + 'cv_val_1.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=data_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline)
)

##################################
# optimizer
##################################

runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)

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

optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
    type='DistOptimizerHook',
    update_interval=1,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=0.1,
    min_lr_ratio=5e-6
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',interval=50,
            init_kwargs=dict(
            project='level2-object-detection',
            entity = 'dudskrla',
            name = f"2022_03_31_Large_anchor_ratio" # 날짜 변경 
            ),
            ),
        dict(type='MlflowLoggerHook',
             exp_name=f'{model_name}',
             tags=dict(
                 optim='AdamW',
                 bbox_loss=model_loss,
                 rpn_loss='LabelSmoothing',
                 fold=4
             ),
             )
    ]
)

work_dir = f'./work_dirs/{model_name}/{model_loss}_full_fold1_cv2'
seed = 2022
gpu_ids = 0
