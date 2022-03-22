# runtime # configs/_base_/default_runtime.py 참고 # 경로 ?
checkpoint_config = dict(interval=1) # ? # max_keep_ckpts ?

# wandb 연결 
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook', interval=50), # CLI 창에서 띄울 수 있는 log 
        dict(type='WandbLoggerHook',interval=50,
            init_kwargs=dict(
                project='level2-object-detection',
                entity = 'dudskrla',
                name = f"2022_03_22_faster_rcnn_swin" # 날짜 변경 
            ),
            )
    ])


    # ml flow 사용 코드 # mlruns 폴더에서 실행 가능 (?)
    # hooks=[
    #     dict(type='MLflowLoggerHook',
    #         exp_name='2022_03_22',
    #         tags=dict(
    #             lr=0.002,
    #             epochs=12
    #         ),
    #         log_model=True # default로 True 
    # ]




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
