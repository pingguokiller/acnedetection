# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)  # 0.00125 * B  # proportional to the batchsize: 0.002: 2   0.08:8
optimizer_config = dict(grad_clip=None)
# learning policy
base_epoch_num = 1 # 1 16
lr_config = dict(
    policy='step',  # 优化策略
    warmup='linear',  # 初始的学习率增加的策略，linear为线性增加
    warmup_iters=50,  # 在初始的500次迭代中学习率逐渐增加 # 50
    warmup_ratio=0.001,  # 起始的学习率 # 0.001
    step=[8*base_epoch_num, 11*base_epoch_num], # [8, 11]*[1, 16]  8*16, 11*16
)  
# total_epochs = 100  # 最大epoch数
runner = dict(type='EpochBasedRunner', max_epochs=15*base_epoch_num) # 15 15*16
