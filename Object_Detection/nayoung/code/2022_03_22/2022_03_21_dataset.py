# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# normalize 
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# train dataset # transform + image load
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# test dataset 
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 900),
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# json 파일 
json_root = '/opt/ml/detection/dataset/seed2022/'
data = dict(
    samples_per_gpu=6, # GPU 당 batch size 설정 
    workers_per_gpu=4, # num_worker 설정 
    train=dict(
        type=dataset_type,
        classes=classes, # inference 할 때 필요 
        ann_file=json_root + 'train.json', # split 완료 
        img_prefix=data_root,
        pipeline=train_pipeline), # transform 설정 
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=json_root + 'val.json', # split 완료
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')