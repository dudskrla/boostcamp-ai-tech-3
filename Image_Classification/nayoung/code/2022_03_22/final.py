# final.py

'''
model settings
'''

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'

# fp16 = dict(loss_scale=512.) # ?

# model settings
model = dict(
    type='FasterRCNN', # model type
    
    # backbone model 설정
    backbone=dict(
        # 에러 발생 # TypeError: FasterRCNN: SwinTransformer: __init__() got an unexpected keyword argument '_delete_'
        # _delete_=True,  ## 기존에 백본을 Resnet을 썼는데 Swin으로 쓰겠다. lr과 같은 다른 config에도 같이 사용이 가능한 인자
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),

    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768], # in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),

    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),

        # class loss settings
        loss_cls=dict( 
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),

        # bbox loss settings 
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10, 
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=1.0))), # loss 변경 

    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),

    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

'''
dataset settings
'''

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# train dataset
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
json_root = '/opt/ml/detection/nayoung/utils/stratified_kfold/'
data = dict(
    samples_per_gpu=6, # GPU 당 batch size 설정 
    workers_per_gpu=4, # num_worker 설정 
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=json_root + 'cv_train_1.json', # split 완료 
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=json_root + 'cv_val_1.json', # split 완료
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')


'''
optimizer settings
'''

# optimizer # configs/_base_/schedules/schedule_1x.py 참고 

# learning rate, optimizer type, weight decay settings 
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001) 
optimizer_config = dict(grad_clip=None)

# learning rate policy
lr_config = dict(
    policy='step', # scheduler 설정
    warmup='linear', # warmup을 할건지
    warmup_iters=500, # warmup iteration 얼마나 줄건지
    warmup_ratio=0.001, # step은 얼마마다 update 
    step=[8, 11])

runner = dict(type='EpochBasedRunnerAmp', max_epochs=12) # epoch 설정 # AMP

'''
run time settings
'''

# runtime # configs/_base_/default_runtime.py 참고 
checkpoint_config = dict(interval=1) # ?

# wandb 연결 
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook', interval=50),
        dict(type='WandbLoggerHook',interval=50,
            init_kwargs=dict(
                project='level2-object-detection',
                entity = 'dudskrla',
                name = f"2022_03_22_faster_rcnn_swin" # 날짜 변경 
            ),
            )
    ])




# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
# 1 epoch에 train과 validation을 모두 하고 싶으면 workflow = [('train', 1), ('val', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
