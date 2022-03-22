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

runner = dict(type='EpochBasedRunner', max_epochs=12) # epoch 설정 
